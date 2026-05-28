# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# Install with vLLM rollout backend:
uv pip install -e ".[test,vllm]"
# Or with SGLang rollout backend:
uv pip install -e ".[test,sglang]"

uv pip install pre-commit hydra-core
pre-commit install
```

### Linting

```bash
# Staged changes only
pre-commit run

# All files
pre-commit run --all-files

# Specific hooks only
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
pre-commit run autogen-trainer-cfg --all-files
```

### Testing

```bash
# Run a single test file
python -m pytest tests/trainer/ppo/test_xxx.py -v

# Run a specific test
python -m pytest tests/trainer/ppo/test_xxx.py::test_name -v

# Run tests matching a pattern
python -m pytest tests/ -k "ppo" -v

# CPU-only tests (no GPU required)
python -m pytest tests/trainer/ tests/workers/ tests/tools/ tests/utils/ \
    tests/models/ tests/single_controller/ tests/checkpoint_engine/ \
    tests/experimental/ tests/special_distributed/ tests/special_standalone/ -v

# GPU tests (require CUDA)
python -m pytest tests/special_e2e/ -v
```

### Running a training job

```bash
# PPO with FSDP + vLLM
python -m verl.trainer.main_ppo \
    --config-path=verl/trainer/config \
    --config-name=ppo_trainer

# GRPO with FSDP
bash examples/grpo_trainer/run_qwen2_5_7b_fsdp.sh

# SFT training
python -m verl.trainer.sft_trainer \
    --config-path=verl/trainer/config \
    --config-name=sft_trainer_engine
```

### Regenerating auto-generated trainer configs

When modifying YAML configs under `verl/trainer/config/`, run:

```bash
bash scripts/generate_trainer_config.sh
```

This regenerates `_generated_ppo_trainer.yaml`, `_generated_ppo_megatron_trainer.yaml`, etc.

---

## Architecture

verl is a distributed RL training library for LLMs. It uses a **hybrid-controller programming model** where an actor generates responses (rollout), a critic evaluates them, and training signals flow back to update the actor — all coordinated via a Ray-based single controller.

### Top-level layout

| Directory | Purpose |
|---|---|
| `verl/trainer/` | Training orchestration (main entry points, PPO logic, configs) |
| `verl/workers/` | Distributed workers (actor/critic training, rollout generation) |
| `verl/single_controller/` | Ray-based resource allocation and worker group management |
| `verl/models/` | Model registry, HuggingFace ↔ Megatron bridge |
| `verl/tools/` | Tool-calling schemas for multi-turn agent loops |
| `verl/experimental/` | Features being upstreamed (agent_loop, reward_loop, fully_async_policy, etc.) |
| `verl/checkpoint_engine/` | Checkpoint I/O backends (NCCL, Mooncake, NIXL, etc.) |
| `examples/` | Standalone training scripts per algorithm (PPO, GRPO, SFT, GSPO, DAPO, etc.) |
| `recipe/` | Git submodule: verl-recipe (community-contributed algorithms) |
| `tests/` | Mirrors `verl/` structure; `special_e2e/` for GPU integration tests |
| `scripts/` | Utility scripts (config generation, model conversion, diagnostics) |

### Key abstractions

**`DataProto`** (`verl/protocol.py`) — The universal data container wrapping a `TensorDict`. All data flowing between workers (inputs, logprobs, rewards, advantages) travels as `DataProto` objects. Supports padding/unpadding and device transfer.

**`single_controller`** (`verl/single_controller/`) — A thin abstraction over Ray for allocating GPU resources and managing worker groups. Key classes:
- `ResourcePool` / `RayResourcePool` — tracks per-node GPU allocations
- `WorkerGroup` / `RayWorkerGroup` — manages a set of workers and dispatches method calls to them
- `create_colocated_worker_cls` — fuses multiple worker roles onto the same GPU set

**`BaseEngine`** (`verl/workers/engine/base.py`) — Abstract interface for training backends. Implementations:
- `verl/workers/engine/fsdp/` — FSDP / FSDP2 backend
- `verl/workers/engine/megatron/` — Megatron-LM backend
- `verl/workers/engine/torchtitan/` — TorchTitan backend
- `verl/workers/engine/veomni/` — VeOmni backend

**Rollout** (`verl/workers/rollout/`) — Response generation backends:
- `vllm_rollout/` — vLLM-based rollout (async server or inline)
- `sglang_rollout/` — SGLang-based rollout
- `hf_rollout.py` — HuggingFace Transformers rollout
- `trtllm_rollout/` — TensorRT-LLM rollout
- `llm_server.py` — LLMServerManager for managing distributed inference servers

**`engine_workers.py`** (`verl/workers/engine_workers.py`) — Ray workers that wrap each engine type (actor, critic, ref, reward model). They own the model, optimizer, and LR scheduler, and execute train/forward/save/load operations dispatched by the trainer.

### Training dataflow (PPO)

1. **Rollout** — Actor generates responses for prompts via the rollout engine
2. **Reward** — Reward model or verifiable reward function scores each response
3. **Advantage** — GAE (or other advantage estimator) computes advantages from rewards and critic values
4. **Actor update** — PPO loss with optional KL penalty against reference policy
5. **Critic update** — MSE regression on value targets (optional; GRPO/RLOO skip this)

The trainer (`verl/trainer/ppo/ray_trainer.py`) orchestrates this loop, dispatching work to engine workers through the single_controller / WorkerGroup APIs.

### Two trainer modes

- **`main_ppo.py`** — Legacy async trainer. Deprecated, will be replaced by `main_ppo_sync.py`.
- **`main_ppo_sync.py`** — Synchronous trainer with TransferQueue-based zero-copy data transfer, ReplayBuffer support, and multiple agent-loop outputs per prompt. This is the preferred path going forward.

### Configuration system

Uses Hydra with YAML configs under `verl/trainer/config/`. The structure:
- `ppo_trainer.yaml` — top-level config (composes per-component configs)
- `config/` — Python dataclass definitions for typed config access
- `actor/`, `critic/`, `ref/`, `rollout/`, `reward/`, `data/`, `model/`, `engine/`, `optim/`, `algorithm/` — per-component YAML configs
- `_generated_*.yaml` — auto-generated full configs (run `scripts/generate_trainer_config.sh` to regenerate)

Configs are validated at startup via `verl/utils/config.py`.

### Adding a new algorithm

1. Create a new example script under `examples/<algo>_trainer/` with a shell script that calls `verl.trainer.main_ppo` or `verl.trainer.main_ppo_sync` with appropriate Hydra overrides
2. If the algorithm needs custom advantage estimation or loss functions, add them to `verl/trainer/ppo/core_algos.py`
3. If it needs custom reward computation, add to `verl/trainer/ppo/reward.py`

---

## Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

### 1. Contribution Policy (Mandatory)

#### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

#### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

#### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

#### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

### 2. Development Workflow

#### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

#### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

### Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
