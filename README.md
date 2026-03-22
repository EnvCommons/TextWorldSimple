# TextWorld Simple

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/textworld_simple)

## Description

TextWorld Simple is an environment for evaluating language model agents on text-based adventure games using Microsoft's [TextWorld](https://github.com/microsoft/TextWorld) framework. Agents must navigate a 6-room house, find a key to escape a locked bedroom, locate a food item, and cook it on the stove -- all through natural language text commands.

## Capabilities

- Text-based game navigation and room exploration
- Object interaction (open, take, put, unlock)
- Spatial reasoning across a multi-room environment
- Goal-directed planning under varying information conditions
- Handling varying reward densities (dense, balanced, sparse)

## Compute Requirements

TextWorld Simple has minimal compute requirements. Games are generated and played locally with no sandbox or external services needed.

## License

[MIT](https://opensource.org/license/mit) (matching TextWorld's license).

## Tasks

There are 90 tasks across 9 difficulty configurations and 2 splits.

| Split | Configurations | Seeds | Tasks |
|-------|---------------|-------|-------|
| train | 9 (3 rewards x 3 goals) | 7 (seeds 1-7) | 63 |
| test | 9 (3 rewards x 3 goals) | 3 (seeds 8-10) | 27 |

**Reward density settings:**
- **Dense**: Sub-quest rewards for each milestone (opening containers, getting key, unlocking door, reaching kitchen, visiting rooms, picking up food, cooking). Max score varies (typically 8 points).
- **Balanced**: Milestone rewards at key stages (reaching kitchen, opening doors, getting food, cooking). Max score is 3 points.
- **Sparse**: Only the final goal (put food on stove) gives reward. Max score is 1 point.

**Goal description settings:**
- **Detailed**: Full TextWorld-generated objective describing the quest step by step.
- **Brief**: A one-line hint (e.g., "The dinner is almost ready! It's only missing a grilled milk.").
- **None**: No objective description provided; the agent must explore to discover the goal.

Each task uses a different random seed controlling food item selection, key placement, and food name shuffling. The room layout is fixed across all tasks.

## Reward Structure

This is a verifiable reward environment with configurable density. Per-step rewards are the score delta normalized by the maximum possible score:

$$\text{reward}_{\text{step}} = \frac{\text{score}_{t} - \text{score}_{t-1}}{\text{max\_score}}$$

The final reward upon game completion is the fraction of total points earned:

$$\text{reward}_{\text{final}} = \frac{\text{score}_{\text{final}}}{\text{max\_score}}$$

Rewards range from 0.0 to 1.0. A score of 1.0 means the agent successfully completed the cooking quest.

We do not use LLM graders for this task.

## Data

Games are procedurally generated at runtime by the TextWorld library using deterministic seeds. No external data files are required.

## Tools

Agents are given a single tool:

- **`send_command`**: Send a natural language text command to the game (e.g., `"open chest drawer"`, `"take old key from antique trunk"`, `"go east"`, `"put milk on stove"`). Returns the game's text response, current score, and step count.

## Time Horizon

TextWorld Simple is a multi-turn environment. The optimal solution requires 8-12 commands depending on the seed. A maximum of 50 steps is allowed before the game is terminated.

## Other Environment Requirements

There are no further environment requirements. TextWorld Simple works out of the box without any secrets or API keys.

## Safety

Agents in TextWorld Simple interact only with a procedurally generated text game. The environment does not present safety risks, as agents have no access to external systems, the internet, or real-world resources.

## Citations

```bibtex
@inproceedings{cote2019textworld,
  title={TextWorld: A Learning Environment for Text-Based Games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and Tay, Wendy and Trischler, Adam},
  booktitle={Computer Games: 7th Workshop, CGW 2018},
  pages={41--75},
  year={2019},
  publisher={Springer}
}
```
