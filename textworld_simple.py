import asyncio
import atexit
import os
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import textworld
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock


REWARD_TYPES = ["dense", "balanced", "sparse"]
GOAL_TYPES = ["detailed", "brief", "none"]
TRAIN_SEEDS = list(range(1, 8))   # seeds 1-7
TEST_SEEDS = list(range(8, 11))   # seeds 8-10
MAX_STEPS = 50

# Cache of compiled game files keyed by (rewards, goal, seed, split).
# Compilation runs in a ProcessPoolExecutor because textworld's tatsu-based
# parser has module-level state that isn't thread-safe. Each worker process
# has isolated parser state, so compilations parallelise safely. Once a game
# is compiled, its file path and metadata are cached in-process and reused
# by subsequent sessions with the same settings — no subprocess call needed.
_GAME_CACHE: dict[Tuple[str, str, int, bool], Tuple[str, list, str]] = {}
_COMPILE_POOL: ProcessPoolExecutor | None = None
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "textworld_simple_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# textworld.start + reset touch the module-level tatsu parser, which isn't
# thread-safe. Serialize those calls across all sessions in this process.
_LOAD_LOCK = threading.Lock()

# Per-key async locks that dedupe concurrent compiles for the same cache_key.
# Without this, two concurrent cache-miss setups would both submit compile jobs
# that write to the same output path and clobber each other.
_COMPILE_LOCKS: dict[Tuple[str, str, int, bool], "asyncio.Lock"] = {}


def _get_compile_pool() -> ProcessPoolExecutor:
    """Lazy-init the process pool so it's only created in the main process."""
    global _COMPILE_POOL
    if _COMPILE_POOL is None:
        _COMPILE_POOL = ProcessPoolExecutor(max_workers=os.cpu_count() or 4)
    return _COMPILE_POOL


@atexit.register
def _shutdown_compile_pool() -> None:
    if _COMPILE_POOL is not None:
        _COMPILE_POOL.shutdown(wait=False, cancel_futures=True)


def _compile_game_in_subprocess(
    rewards: str, goal: str, seed: int, test: bool, cache_subdir: str, task_id: str,
) -> Tuple[str, list, str]:
    """Compile a textworld game in a worker process. Must be picklable (top-level)."""
    # Re-import inside the worker (each process has its own module state)
    import os
    import textworld
    from textworld import GameOptions
    from textworld.challenges.tw_simple.simple import make as make_simple_game
    from textworld.generator import compile_game

    settings = {"rewards": rewards, "goal": goal, "test": test}
    options = GameOptions()
    options.seeds = seed
    os.makedirs(cache_subdir, exist_ok=True)
    options.path = os.path.join(cache_subdir, f"tw_simple_{task_id}")

    game = make_simple_game(settings, options)
    walkthrough = game.metadata.get("walkthrough", [])
    objective = game.objective or ""
    game_file = compile_game(game, options)
    return game_file, walkthrough, objective


class TaskSpec(BaseModel):
    id: str
    rewards: str
    goal: str
    seed: int
    split: str = ""


class SendCommandParams(BaseModel, extra="forbid"):
    """Send a text command to the TextWorld game."""
    command: str


class TextWorldSimple(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)
        self._env = None
        self._game_state = None
        self._walkthrough = []
        self._game_objective = ""
        self._max_score = 0
        self._step_count = 0
        self._max_steps = MAX_STEPS
        self._game_done = False
        self._prev_score = 0

    def _load_game_file(self, game_file: str) -> None:
        """Open a compiled game file. Serialized because textworld.start/reset
        share module-level tatsu parser state that races under threads."""
        infos = textworld.EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            score=True,
            max_score=True,
        )
        with _LOAD_LOCK:
            self._env = textworld.start(game_file, infos)
            self._game_state = self._env.reset()
        self._max_score = self._game_state.max_score

    async def setup(self) -> None:
        """Compile the game (in a worker process) and open the game file."""
        if self._env is not None:
            return

        cache_key = (
            self.validated.rewards,
            self.validated.goal,
            self.validated.seed,
            self.validated.split == "test",
        )

        if cache_key not in _GAME_CACHE:
            # setdefault is atomic within a single event loop, so concurrent
            # callers for the same key observe the same Lock instance.
            lock = _COMPILE_LOCKS.setdefault(cache_key, asyncio.Lock())
            async with lock:
                if cache_key not in _GAME_CACHE:
                    loop = asyncio.get_running_loop()
                    cache_subdir = os.path.join(_CACHE_DIR, f"tw_{self.validated.id}")
                    _GAME_CACHE[cache_key] = await loop.run_in_executor(
                        _get_compile_pool(),
                        _compile_game_in_subprocess,
                        self.validated.rewards,
                        self.validated.goal,
                        self.validated.seed,
                        self.validated.split == "test",
                        cache_subdir,
                        self.validated.id,
                    )

        game_file, walkthrough, objective = _GAME_CACHE[cache_key]
        self._walkthrough = walkthrough
        self._game_objective = objective

        await asyncio.to_thread(self._load_game_file, game_file)

    async def get_prompt(self) -> List[TextBlock]:
        await self.setup()
        initial_obs = self._game_state.feedback

        objective_section = ""
        if self._game_objective:
            objective_section = f"\nObjective: {self._game_objective}\n"

        prompt = f"""You are playing a text-based adventure game set in a house. You interact with the game by sending text commands.
{objective_section}
Available commands include:
- "look" - examine your surroundings
- "inventory" - check what you're carrying
- "examine <object>" - look at something closely
- "open <container>" - open a container or door
- "close <container>" - close a container or door
- "take <item>" or "take <item> from <container>" - pick up an item
- "drop <item>" - drop an item
- "put <item> on <surface>" - place an item on a surface
- "go <direction>" - move (north, south, east, west)
- "unlock <door> with <key>" - unlock a door
- "eat <food>" - eat a food item

Use the send_command tool to interact with the game. You have a maximum of {self._max_steps} steps.

--- GAME START ---
{initial_obs}"""

        return [TextBlock(text=prompt)]

    @tool
    async def send_command(self, params: SendCommandParams) -> ToolOutput:
        """Send a text command to the game and receive the result."""
        if self._game_done:
            return ToolOutput(
                blocks=[TextBlock(text="The game is already over.")],
                metadata={"error": "game_over"},
                reward=0.0,
                finished=True,
            )

        self._step_count += 1
        game_state, _cumulative_score, done = await asyncio.to_thread(self._env.step, params.command)
        self._game_state = game_state

        current_score = game_state.score
        step_delta = current_score - self._prev_score
        self._prev_score = current_score

        observation = game_state.feedback

        if done:
            self._game_done = True
            max_score = self._max_score if self._max_score > 0 else 1
            final_reward = current_score / max_score

            won = current_score >= max_score
            result_text = f"{observation}\n\nFinal score: {current_score}/{max_score} in {self._step_count} steps."

            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata={
                    "score": current_score,
                    "max_score": max_score,
                    "steps": self._step_count,
                    "won": won,
                },
                reward=final_reward,
                finished=True,
            )

        if self._step_count >= self._max_steps:
            self._game_done = True
            max_score = self._max_score if self._max_score > 0 else 1
            final_reward = current_score / max_score

            result_text = f"{observation}\n\nMax steps ({self._max_steps}) reached. Game over.\nFinal score: {current_score}/{max_score}"

            return ToolOutput(
                blocks=[TextBlock(text=result_text)],
                metadata={
                    "score": current_score,
                    "max_score": max_score,
                    "steps": self._step_count,
                    "won": False,
                    "max_steps_exceeded": True,
                },
                reward=final_reward,
                finished=True,
            )

        max_score = self._max_score if self._max_score > 0 else 1
        step_reward = step_delta / max_score

        result_text = f"{observation}\n\nScore: {current_score}/{max_score} | Steps: {self._step_count}/{self._max_steps}"

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "score": current_score,
                "max_score": max_score,
                "step_delta": step_delta,
                "steps": self._step_count,
            },
            reward=step_reward,
            finished=False,
        )

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ["train", "test"]:
            raise ValueError(f"Unknown split: {split}")

        seeds = TRAIN_SEEDS if split == "train" else TEST_SEEDS

        tasks = []
        for rewards in REWARD_TYPES:
            for goal in GOAL_TYPES:
                for seed in seeds:
                    task_id = f"{rewards}_{goal}_{seed}"
                    tasks.append({
                        "id": task_id,
                        "rewards": rewards,
                        "goal": goal,
                        "seed": seed,
                        "split": split,
                    })

        return tasks

    def __del__(self):
        try:
            if hasattr(self, "_env") and self._env is not None:
                self._env.close()
        except Exception:
            pass
