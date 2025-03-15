
# Tree of Thoughts (ToT) Rollout Pipeline for RL Fine-tuning

## Overview

This repository implements a **Tree of Thoughts (ToT)** based agent designed to generate high-quality interaction trajectories in complex environments. These trajectories are intended for **reinforcement learning (RL) fine-tuning** of large language models (LLMs), enabling more efficient and thoughtful decision-making.

The system leverages **multi-step reasoning** and **branching exploration**, powered by LLM APIs, to produce diverse and rich trajectories suitable for offline RL datasets.

---

## Directory Structure

```
.
├── llm_agent_tot.py        # Core ToT agent logic for generating thought-action trajectories
├── run_agent_tot.py        # Main entry point for running ToT agent and saving rollouts
├── util.py                 # Utility functions (random seed, prompt formatting)
├── prompt/                 # Directory for LLM prompting templates (to be provided)
└── README.md               # This README file
```

---

## Purpose

This pipeline is built to **roll out high-quality interaction trajectories** for downstream **RL fine-tuning** of LLM-based agents.  
It enables:
- Complex **reasoning-action** trajectory generation.
- Flexible **branching exploration** to gather diverse behavioral data.
- Rich **thought-action-reward** sequences suitable for RL objective optimization.

---

## Dependencies

- Python 3.8+
- Gym (environment interaction)
- Transformers (random seed control)
- PyTorch (random seed control)
- Requests (API calls)

### Install dependencies:
```bash
pip install gym torch transformers requests
```

---

## Usage

### 1. Run the ToT Agent to Generate Trajectories

```bash
python run_agent_tot.py --mode test --temperature 0.7 --top_p 0.9 --max_batch_size 1 --max_gen_len 512
```

#### Parameters:
| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode (default: `test`) | test |
| `--temperature` | Sampling temperature for LLM API | 0.5 |
| `--top_p` | Top-p nucleus sampling for LLM API | 0.5 |
| `--max_batch_size` | Maximum batch size for LLM API | 1 |
| `--max_gen_len` | Maximum token length for LLM output | 512 |

### 2. Output

- All generated trajectories are saved in `tot_agent_results.json`.
- Each trajectory contains:
  - Observation sequences.
  - Thought and action sequences.
  - Log probabilities (if available).
  - Cumulative rewards.
  
### Example output format:
```json
[
  {
    "episode": 0,
    "paths": [
      {
        "trajectory": [
          {
            "observation": "...",
            "thought_history": [...],
            "actions_history": [...],
            "current_thought": "...",
            "current_thought_logp": [...],
            "current_action": "...",
            "current_action_logp": [...],
            "cumulative_reward": 1.0
          },
          ...
        ],
        "final_reward": 1.0
      }
    ]
  }
]
```

---

## Components

### `llm_agent_tot.py`
- **APIAgent_TOT**: Main agent class implementing the ToT reasoning mechanism.
  - Recursive trajectory tree construction.
  - API-based thought and action generation.
  - Environment cloning for independent exploration.
- **extract_paths**: Extract linear trajectories from ToT tree for RL datasets.

### `run_agent_tot.py`
- Run script for rolling out ToT trajectories in a WebShop-like `WebAgentTextEnv`.
- Handles configuration, agent-environment interaction, and result saving.

### `util.py`
- `set_random_seed(seed)`: Ensures reproducibility.
- `refine_prompt(prompt, **kwargs)`: Template filling for dynamic prompting.

---

## Customization

### 1. **LLM API Endpoint**
Set your LLM API endpoint in `run_agent_tot.py` and `llm_agent_tot.py`:
```python
api_url = "http://your-llm-api"
```

### 2. **Prompt Templates**
Provide your custom thought and action prompting templates under `prompt/` directory:
- `prompt_system_thought`
- `prompt_user_thought`
- `prompt_system_action`
- `prompt_user_action`

### 3. **Environment**
Replace `'WebAgentTextEnv-v0'` with any custom Gym-compatible environment supporting:
- `reset()`
- `step(action)`
- Optionally: `get_state()` and `set_state()` for efficient environment cloning.

---

## Goal of the Dataset

The generated `tot_agent_results.json` file serves as:
- **Offline RL dataset** for fine-tuning agents to imitate or optimize ToT-based reasoning.
- **Supervised learning data** for LLM behavior cloning.
- **Evaluation corpus** for testing LLM planning capabilities.

---

## Notes

- **Branching factor** and **depth** of ToT reasoning can be adjusted in `APIAgent_TOT` initialization.
- Random seed setting (`set_random_seed`) is available but commented out for default stochasticity; uncomment for deterministic runs.
- Make sure the prompting templates are well-aligned with the task to ensure meaningful thoughts and actions.

---

## Contact & Contribution

Feel free to contribute by submitting pull requests or raising issues.
