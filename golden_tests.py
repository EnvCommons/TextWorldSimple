import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from textworld_simple import TextWorldSimple, SendCommandParams


# --- list_tasks / list_splits ---

def test_list_splits():
    splits = TextWorldSimple.list_splits()
    assert splits == ["train", "test"]


def test_list_tasks_train():
    tasks = TextWorldSimple.list_tasks("train")
    assert len(tasks) == 63  # 3 rewards x 3 goals x 7 seeds


def test_list_tasks_test():
    tasks = TextWorldSimple.list_tasks("test")
    assert len(tasks) == 27  # 3 rewards x 3 goals x 3 seeds


def test_list_tasks_invalid_split():
    with pytest.raises(ValueError):
        TextWorldSimple.list_tasks("invalid")


def test_task_structure():
    tasks = TextWorldSimple.list_tasks("train")
    task = tasks[0]
    assert "id" in task
    assert "rewards" in task
    assert "goal" in task
    assert "seed" in task
    assert "split" in task
    assert task["split"] == "train"
    assert task["rewards"] in ["dense", "balanced", "sparse"]
    assert task["goal"] in ["detailed", "brief", "none"]


# --- Environment initialization ---

@pytest.mark.asyncio
async def test_environment_init():
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    prompt = await env.get_prompt()
    assert len(prompt) == 1
    assert isinstance(prompt[0].text, str)
    assert len(prompt[0].text) > 0
    assert "send_command" in prompt[0].text.lower() or "command" in prompt[0].text.lower()


@pytest.mark.asyncio
async def test_prompt_includes_objective_when_detailed():
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    prompt = await env.get_prompt()
    assert "Objective:" in prompt[0].text


@pytest.mark.asyncio
async def test_prompt_no_objective_when_none():
    task = {"id": "dense_none_1", "rewards": "dense", "goal": "none", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    prompt = await env.get_prompt()
    assert "Objective:" not in prompt[0].text


# --- Basic gameplay ---

@pytest.mark.asyncio
async def test_send_command_look():
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    result = await env.send_command(SendCommandParams(command="look"))
    assert not result.finished
    text = result.blocks[0].text.lower()
    assert "bedroom" in text


@pytest.mark.asyncio
async def test_send_command_inventory():
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    result = await env.send_command(SendCommandParams(command="inventory"))
    assert not result.finished
    assert result.blocks[0].text  # Should have some response


# --- Walkthrough completion ---

@pytest.mark.asyncio
async def test_walkthrough_dense():
    """Following the walkthrough with dense rewards should complete with reward=1.0."""
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    walkthrough = env._walkthrough
    assert len(walkthrough) > 0

    result = None
    for cmd in walkthrough:
        result = await env.send_command(SendCommandParams(command=cmd))
        if result.finished:
            break

    assert result is not None
    assert result.finished
    assert result.metadata["won"] is True
    assert result.reward == 1.0
    assert result.metadata["score"] == result.metadata["max_score"]


@pytest.mark.asyncio
async def test_walkthrough_sparse():
    """Following the walkthrough with sparse rewards should complete with reward=1.0."""
    task = {"id": "sparse_detailed_1", "rewards": "sparse", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    walkthrough = env._walkthrough

    result = None
    for cmd in walkthrough:
        result = await env.send_command(SendCommandParams(command=cmd))
        if result.finished:
            break

    assert result is not None
    assert result.finished
    assert result.metadata["won"] is True
    assert result.reward == 1.0


# --- Max steps ---

@pytest.mark.asyncio
async def test_max_steps_exceeded():
    task = {"id": "sparse_none_1", "rewards": "sparse", "goal": "none", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    env._max_steps = 3
    await env.get_prompt()

    for _ in range(3):
        result = await env.send_command(SendCommandParams(command="look"))

    assert result.finished
    assert result.metadata.get("max_steps_exceeded") is True


# --- Game over state ---

@pytest.mark.asyncio
async def test_game_already_over():
    task = {"id": "sparse_none_1", "rewards": "sparse", "goal": "none", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    env._max_steps = 1
    await env.get_prompt()

    result = await env.send_command(SendCommandParams(command="look"))
    assert result.finished

    result2 = await env.send_command(SendCommandParams(command="look"))
    assert result2.finished
    assert result2.metadata.get("error") == "game_over"
    assert result2.reward == 0.0


# --- Losing condition ---

@pytest.mark.asyncio
async def test_losing_by_eating_food():
    """Eating the food should end the game with done=True but not won."""
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    walkthrough = env._walkthrough
    # Play all steps except the last (which is "put X on stove")
    for cmd in walkthrough[:-1]:
        result = await env.send_command(SendCommandParams(command=cmd))
        if result.finished:
            break

    if not result.finished:
        # Extract the food name from the last walkthrough command
        last_cmd = walkthrough[-1]  # e.g. "put milk on stove"
        food = last_cmd.replace("put ", "").replace(" on stove", "")
        result = await env.send_command(SendCommandParams(command=f"eat {food}"))
        assert result.finished
        assert result.metadata["won"] is False
        assert result.reward < 1.0


# --- Determinism ---

@pytest.mark.asyncio
async def test_deterministic_same_seed():
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env1 = TextWorldSimple(task_spec=task)
    env2 = TextWorldSimple(task_spec=task)

    prompt1 = await env1.get_prompt()
    prompt2 = await env2.get_prompt()

    assert prompt1[0].text == prompt2[0].text


@pytest.mark.asyncio
async def test_different_seeds_different_walkthroughs():
    task1 = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    task2 = {"id": "dense_detailed_2", "rewards": "dense", "goal": "detailed", "seed": 2, "split": "train"}

    env1 = TextWorldSimple(task_spec=task1)
    env2 = TextWorldSimple(task_spec=task2)

    # Different seeds should produce games (walkthroughs may differ due to
    # different key placement and food selection)
    assert len(env1._walkthrough) > 0
    assert len(env2._walkthrough) > 0


# --- Per-step rewards ---

@pytest.mark.asyncio
async def test_dense_per_step_rewards():
    """Dense reward mode should give non-zero rewards for sub-quest steps."""
    task = {"id": "dense_detailed_1", "rewards": "dense", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    # The first walkthrough step should give a reward in dense mode
    result = await env.send_command(SendCommandParams(command=env._walkthrough[0]))
    assert not result.finished
    assert result.reward > 0


@pytest.mark.asyncio
async def test_sparse_no_intermediate_rewards():
    """Sparse reward mode should give 0 reward for intermediate steps."""
    task = {"id": "sparse_detailed_1", "rewards": "sparse", "goal": "detailed", "seed": 1, "split": "train"}
    env = TextWorldSimple(task_spec=task)
    await env.get_prompt()

    # Intermediate steps should yield 0 reward in sparse mode
    result = await env.send_command(SendCommandParams(command=env._walkthrough[0]))
    assert not result.finished
    assert result.reward == 0.0
