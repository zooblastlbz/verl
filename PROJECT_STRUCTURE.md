# verl 项目结构分析报告

## 概述

**verl** (Volcano Engine Reinforcement Learning) 是字节跳动 Seed 团队发起的分布式 RL 训练库，专为大语言模型（LLM）的强化学习训练设计。项目基于论文 [HybridFlow](https://arxiv.org/abs/2409.19256v2)，使用 Ray 作为分布式计算框架，支持多种训练后端和推理引擎，提供灵活的混合控制器编程模型。

---

## 1. 项目顶层结构

```
verl/
├── verl/                    # 主 Python 包（核心库）
├── examples/                # 各算法的训练脚本示例
├── tests/                   # 测试套件
├── docs/                    # 文档
├── docker/                  # Docker 镜像构建
├── recipe/                  # Git 子模块（社区贡献算法）
├── scripts/                 # 工具脚本
├── setup.py                 # 安装脚本（fallback）
├── pyproject.toml           # 项目元数据与工具配置
├── requirements.txt         # 开发依赖
├── requirements-test.txt    # 测试依赖
├── requirements-npu.txt     # NPU（昇腾）依赖
├── CLAUDE.md               # AI Agent 协作指南
├── CONTRIBUTING.md         # 贡献指南
└── README.md               # 项目说明
```

---

## 2. 核心包 `verl/` 结构详解

### 2.1 入口文件

| 文件 | 功能 |
|---|---|
| `__init__.py` | 包入口，导出 `DataProto`、`__version__`；应用 NPU 补丁 |
| `protocol.py` | **数据传输协议**：定义 `DataProto` 类，是系统内所有数据的统一交换载体（~1346行） |
| `base_config.py` | 配置基类：冻结的 dataclass，实现 Mapping 接口 |

### 2.2 各子模块详解

```
verl/
├── trainer/           # 训练入口与核心算法
├── workers/           # 分布式工作器
├── single_controller/ # 分布式编排框架
├── models/            # 模型定义与注册
├── experimental/      # 实验性功能
├── checkpoint_engine/ # 权重同步引擎
├── tools/             # 工具调用框架
├── utils/             # 共享工具库
├── model_merger/      # 检查点合并器
└── third_party/       # 第三方补丁
```

---

## 3. `verl/trainer/` — 训练入口与核心算法

### 3.1 目录结构

```
trainer/
├── main_ppo.py          # [已弃用] PPO 入口（异步模式）
├── main_ppo_sync.py     # 新的同步 PPO 入口（推荐使用）
├── main_eval.py         # 评估入口
├── main_generation_server.py  # 独立生成服务器
├── sft_trainer.py       # SFT 训练器（torchrun）
├── sft_trainer_ray.py   # [实验性] SFT 训练器（Ray）
├── constants_ppo.py     # PPO Ray 运行时环境常量
├── runtime_env.yaml     # Ray 运行时环境配置
│
├── ppo/                 # PPO 核心实现
│   ├── ray_trainer.py   # RayPPOTrainer（主训练循环）
│   ├── core_algos.py    # PPO 损失函数、KL 控制器、优势估计
│   ├── reward.py        # 奖励提取与处理
│   ├── metric_utils.py  # 训练指标计算
│   ├── utils.py         # Role 枚举、辅助函数
│   ├── padding_utils.py # 填充工具
│   ├── prefix_grouper_utils.py  # GRPO 前缀分组
│   └── rollout_corr_helper.py   # Rollout 修正辅助
│
├── distillation/        # 在线蒸馏
└── config/              # Hydra/YAML 配置系统（见第 9 节）
```

### 3.2 训练器架构

**两种训练模式：**

| 模式 | 文件 | 特点 |
|---|---|---|
| 异步 PPO | `main_ppo.py` | 旧版，已在 v0.10.0 废弃 |
| 同步 PPO | `main_ppo_sync.py` | 新版，TransferQueue 零拷贝传输、ReplayBuffer 支持、每个 prompt 支持多个 agent-loop 输出 |

**启动流程：**

1. Hydra 加载配置（`ppo_trainer.yaml` 组合所有子配置）
2. 初始化 Ray 集群
3. 创建 `TaskRunner` Ray Actor
4. 在 TaskRunner 中实例化 `RayPPOTrainer`
5. `RayPPOTrainer` 使用 `RayWorkerGroup` 编排各模型的训练

---

## 4. `verl/workers/` — 分布式工作器

### 4.1 概述

```
workers/
├── engine_workers.py    # TrainingWorker — 承载模型引擎的核心工作器
├── config/              # 工作器配置 dataclass
├── engine/              # 训练引擎后端
├── rollout/             # 推理/生成后端
├── reward_manager/      # 奖励计算管理器
└── utils/               # 损失函数与填充工具
```

### 4.2 训练引擎后端 (`engine/`)

`BaseEngine` 是训练引擎的抽象基类，定义了完整的训练接口：

```
engine/
├── base.py              # BaseEngine + EngineRegistry（装饰器注册模式）
├── utils.py             # 共享引擎工具
├── fsdp/                # PyTorch FSDP/FSDP2
├── megatron/            # NVIDIA Megatron-LM
├── torchtitan/          # Meta TorchTitan
├── veomni/              # VeOmni
├── automodel/           # 通用 HF AutoModel
└── mindspeed/           # 华为昇腾 MindSpeed
```

| 后端 | 适用硬件 | 说明 |
|---|---|---|
| **FSDP** | NVIDIA/AMD | PyTorch 原生 FSDP/FSDP2 |
| **Megatron** | NVIDIA | 支持大规模 MoE（如 DeepSeek-671B） |
| **TorchTitan** | NVIDIA | Meta 的 Titan 框架 |
| **VeOmni** | 多种 | Volcengine 自研引擎 |
| **AutoModel** | 通用 | 通用 HuggingFace 模型 |
| **Mindspeed** | Ascend NPU | 华为昇腾 NPU |

每个后端提供多种变体：`Engine`（基础）、`EngineWithLMHead`（语言模型头）、`EngineWithValueHead`（价值模型头，仅 Megatron/VeOmni/Mindspeed）。

### 4.3 Rollout 推理后端 (`rollout/`)

`BaseRollout` 定义了三个核心抽象方法：
- `resume(tags)` — 恢复权重或 KV 缓存
- `update_weights(weights_generator)` — 从训练器更新模型权重
- `release()` — 释放 GPU 内存

```
rollout/
├── base.py              # BaseRollout ABC + 注册表
├── hf_rollout.py        # HuggingFace Transformers 生成
├── llm_server.py        # LLM 服务器管理器
├── schema.py            # 请求/响应 schema
├── tokenizer.py         # 分词器工具
├── utils.py             # 异步迭代器工具
├── naive/               # 最简实现
├── vllm_rollout/        # vLLM 后端
├── sglang_rollout/      # SGLang 后端
└── trtllm_rollout/      # TensorRT-LLM 后端
```

| 后端 | 文件 | 特点 |
|---|---|---|
| **vLLM** | `vllm_async_server.py` | Ray Actor 异步服务器 + 分桶权重传输 |
| **SGLang** | `sglang_rollout.py` | 支持 Prefill-Decode 分离 |
| **TRT-LLM** | `trtllm_async_server.py` | NVIDIA TensorRT 优化推理 |
| **HF** | `hf_rollout.py` | 直接 HuggingFace 模型生成 |
| **Naive** | `naive/` | 基础推理实现 |

### 4.4 奖励管理器 (`reward_manager/`)

```
reward_manager/
├── abstract.py          # AbstractRewardManager 抽象基类
├── registry.py          # 注册表
├── naive.py             # 简易奖励计算
├── batch.py             # 批量奖励计算
├── dapo.py              # DAPO 专用
└── prime.py             # PRIME 专用
```

### 4.5 TrainingWorker (`engine_workers.py`)

`TrainingWorker` 是连接训练编排与模型引擎的关键桥梁。它继承 `Worker`（分布式基础）和 `DistProfilerExtension`（性能剖析），负责：
1. 初始化分布式进程组（Ray 方式）
2. 通过 `EngineRegistry` 创建对应的引擎实例
3. 提供高层方法：`init_model()`、`train_batch()`、`infer_batch()`、`save_checkpoint()`、`load_checkpoint()`

---

## 5. `verl/single_controller/` — 分布式编排

这是对 Ray 的一层轻量封装，解耦了驱动端与工作器执行。

```
single_controller/
├── __init__.py              # 重导出
├── base/                    # 平台无关的抽象
│   ├── worker.py            # Worker 基类
│   ├── worker_group.py      # WorkerGroup + ResourcePool
│   └── decorator.py         # @register 装饰器（调度/执行模式）
└── ray/                     # Ray 具体实现
    └── base.py              # RayWorkerGroup、ResourcePoolManager、create_colocated_worker_cls
```

**核心类：**

| 类 | 功能 |
|---|---|
| `Worker` | 分布式工作器基类，处理环境变量、设备配置、融合注册 |
| `WorkerGroup` | 管理一组 Worker，分派方法调用 |
| `ResourcePool` | 跟踪多节点 GPU 分配 |
| `RayWorkerGroup` | WorkerGroup 的 Ray 实现 |
| `create_colocated_worker_cls` | 融合多个 Worker 角色到同一组 GPU |

**调度模式**（定义在 `decorator.py`）：

- `RANK_ZERO` — 仅在 rank 0 执行
- `ONE_TO_ALL` — 从一个 rank 广播到所有
- `ALL_TO_ALL` — 全收集/分散
- `DP_COMPUTE_PROTO` — 使用 `DataProto` 进行数据并行计算
- `DP_COMPUTE_METRIC` — 指标聚合
- `DIRECT_ROLLOUT_METHOD` — vLLM 外部执行器专用

---

## 6. `verl/protocol.py` — 数据传输协议

`DataProto` 是系统的**数据交换中枢**，是一个包含三个核心字段的 dataclass：

| 字段 | 类型 | 用途 |
|---|---|---|
| `batch` | `TensorDict` | 批量张量数据（观测、动作、奖励、对数概率等） |
| `non_tensor_batch` | `dict[str, np.ndarray]` | 非张量元数据 |
| `meta_info` | `dict` | 指标、配置覆盖等元信息 |

关键操作：`chunk`（分块）、`concat`（拼接）、`split`（分割）、`select`（选择）、`union`（合并）、`repeat`（重复）、`make_iterator`（创建 DataLoader）、`padding`（填充/解填充）、`reorder`（重排）。

还定义了 `DataProtoFuture` 类用于异步分布式执行。

---

## 7. `verl/models/` — 模型定义

```
models/
├── registry.py              # ModelRegistry：HF 架构 → Megatron 实现映射
├── weight_loader_registry.py # 权重加载注册
├── mcore/                   # Megatron Core 模型桥接
│   ├── bridge.py            # HF → mcore 桥接
│   ├── mbridge.py           # Megatron Bridge
│   ├── config_converter.py  # HF 配置 → mcore 配置转换
│   ├── loader.py            # 权重加载
│   ├── saver.py             # 权重保存
│   ├── model_forward.py     # 前向传播
│   ├── model_initializer.py # 模型初始化
│   ├── patch.py             # 模型补丁
│   ├── registry.py          # 模型注册表
│   └── weight_converter.py  # 权重格式转换
└── transformers/            # HuggingFace 模型适配器
    ├── dense_common.py      # 通用稠密模型工具
    ├── llama.py             # Llama 配置
    ├── qwen2.py             # Qwen2 配置
    ├── qwen2_vl.py          # Qwen2-VL 视觉语言模型
    ├── qwen3_5.py           # Qwen3.5
    ├── qwen3_vl.py          # Qwen3-VL
    ├── glm4v.py             # GLM-4V
    ├── kimi_vl.py           # Kimi VL
    ├── apertus.py           # Apertus
    ├── monkey_patch.py      # Transformer 猴子补丁
    ├── npu_patch.py         # 昇腾 NPU 补丁
    └── tiled_mlp.py         # Tiled MLP 实现
```

支持通过 `ModelRegistry` 注册的架构：`LlamaForCausalLM`、`Qwen2ForCausalLM`、`MistralForCausalLM`、`ApertusForCausalLM`。

---

## 8. `verl/checkpoint_engine/` — 权重同步引擎

同步训练后端与推理后端之间的模型权重。

```
checkpoint_engine/
├── base.py                             # 管理器 + 注册表 + 工作器
├── nccl_checkpoint_engine.py           # NCCL（all_gather + broadcast）
├── hccl_checkpoint_engine.py           # HCCL（昇腾 NPU）
├── nixl_checkpoint_engine.py           # NIXL（多传输后端）
├── kimi_checkpoint_engine.py           # Mooncake + NCCL/HCCL
└── mooncake_checkpoint_engine.py       # Mooncake Transfer Engine
```

| 后端 | 通信库 | 拓扑 | 弹性 | 适用场景 |
|---|---|---|---|---|
| naive | torch.distributed | all_gather | 无 | 训练+推理同组GPU |
| nccl | NCCL | all_gather+broadcast | 低 | 分离部署、固定集群 |
| hccl | HCCL | all_gather+broadcast | 低 | 昇腾 NPU |
| nixl | NIXL | all_gather+ring p2p | 高 | 弹性推理 |
| kimi_ckpt | Mooncake+NCCL/HCCL | p2p+broadcast | 低 | 分离部署、每次保存检查点 |
| mooncake | Mooncake TE | all_gather+ring p2p | 高 | 分离部署、固定集群 |

三个统一 API：
- `send_weights` — 从生成器获取命名张量并流式发送
- `receive_weights` — 返回张量生成器
- `get_weights` — 从本地缓存产出命名张量

---

## 9. 配置系统 (`verl/trainer/config/`)

使用 **Hydra** + **YAML** 实现分层组合配置。主文件 `ppo_trainer.yaml` 通过 `defaults` 列表组合各组件配置：

```yaml
defaults:
  - model_engine: dp              # 选择后端：dp/megatron/torchtitan/veomni
  - actor@actor_rollout_ref.actor: ${model_engine}_actor
  - ref@actor_rollout_ref.ref: ${model_engine}_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - critic@critic: ${model_engine}_critic
  - reward@reward: reward
  - data@data: legacy_data
  - algorithm@algorithm.rollout_correction: rollout_correction
  - distillation@distillation: distillation
```

切换 `model_engine` 即自动选择对应后端的全部配置。

```
config/
├── ppo_trainer.yaml                # 主 PPO 配置
├── ppo_megatron_trainer.yaml       # Megatron 专用
├── _generated_ppo_trainer.yaml     # 自动生成的完整配置
├── _generated_ppo_megatron_trainer.yaml
├── _generated_ppo_torchtitan_trainer.yaml
├── _generated_ppo_veomni_trainer.yaml
├── sft_trainer_engine.yaml         # SFT 配置
├── config.py                       # 配置 dataclass
├── algorithm.py                    # 算法配置
├── algorithm.py                    # AlgoConfig、KLControlConfig
├── actor/                          # Actor 配置（各后端）
├── critic/                         # Critic 配置
├── ref/                            # 参考模型配置
├── rollout/                        # Rollout 配置
├── reward/                         # 奖励配置
├── data/                           # 数据配置
├── engine/                         # 引擎配置
├── optim/                          # 优化器配置
├── model_engine/                   # 模型引擎配置
├── model/                          # 模型配置
├── profiler/                       # 性能剖析配置
├── npu_profile/                    # NPU 性能剖析配置
├── distillation/                   # 蒸馏配置
└── algorithm/                      # 算法配置（rollout correction）
```

---

## 10. `verl/experimental/` — 实验性功能

```
experimental/
├── agent_loop/              # Agent 循环框架
│   ├── agent_loop.py        # 基础 agent loop
│   ├── single_turn_agent_loop.py  # 单轮 agent
│   ├── tool_agent_loop.py   # 工具使用 agent
│   ├── tool_parser.py       # 工具调用解析
│   └── utils.py
├── fully_async_policy/      # 完全异步 PPO（AReaL 风格）
│   ├── fully_async_main.py  # 入口
│   ├── fully_async_trainer.py  # 异步训练器
│   ├── fully_async_rollouter.py  # 异步 Rollouter
│   ├── message_queue.py     # 消息队列（样本流）
│   ├── detach_utils.py      # 异步模型分离工具
│   ├── config/              # FSDP 和 Megatron 配置
│   └── shell/               # ~25 个 shell 脚本
├── one_step_off_policy/     # 一步离线异步训练
├── reward_loop/             # 外部奖励循环
├── separation/              # Prefill-Decode 分离
└── teacher_loop/            # 教师模型循环（蒸馏）
```

**fully_async_policy** 是最重要的实验模块，通过完全解耦生成与训练，在 128 GPU、Qwen2.5-7B 上实现 **2.35x-2.67x 性能提升**。

---

## 11. `verl/models/` — 模型定义与注册

### 11.1 模型目录

```
models/
├── registry.py              # ModelRegistry：HF 架构 -> Megatron 实现映射
├── weight_loader_registry.py # 权重加载注册
├── mcore/                   # Megatron Core 模型桥接
│   ├── bridge.py            # HF -> mcore 桥接
│   ├── mbridge.py           # Megatron Bridge
│   ├── config_converter.py  # HF 配置 -> mcore 配置转换
│   ├── loader.py            # 权重加载
│   ├── saver.py             # 权重保存
│   ├── model_forward.py     # 前向传播
│   ├── model_initializer.py # 模型初始化
│   ├── patch.py             # 模型补丁
│   ├── registry.py          # 模型注册表
│   └── weight_converter.py  # 权重格式转换
└── transformers/            # HuggingFace 模型适配器
    ├── dense_common.py      # 通用稠密模型工具
    ├── llama.py             # Llama 配置
    ├── qwen2.py             # Qwen2 配置
    ├── qwen2_vl.py          # Qwen2-VL 视觉语言模型
    ├── qwen3_5.py           # Qwen3.5
    ├── qwen3_vl.py          # Qwen3-VL
    ├── glm4v.py             # GLM-4V
    ├── kimi_vl.py           # Kimi VL
    ├── apertus.py           # Apertus
    ├── monkey_patch.py      # Transformer 猴子补丁
    ├── npu_patch.py         # 昇腾 NPU 补丁
    └── tiled_mlp.py         # Tiled MLP 实现
```

支持通过 `ModelRegistry` 注册的架构：`LlamaForCausalLM`、`Qwen2ForCausalLM`、`MistralForCausalLM`、`ApertusForCausalLM`。

---

## 12. `verl/tools/` — 工具调用框架

```
tools/
├── base_tool.py       # BaseTool 抽象类
├── function_tool.py   # @function_tool 装饰器
├── schemas.py         # OpenAI 兼容的 JSON Schema
└── tool_registry.py   # 工具注册表
```

用于多轮 agent loop 场景，模型在 rollout 过程中可以调用外部工具。

---

## 13. 支持的算法（`examples/`）

| 算法目录 | 算法名称 | 说明 |
|---|---|---|
| `ppo_trainer/` | PPO | 基础算法 |
| `grpo_trainer/` | GRPO | 组相对策略优化 |
| `gspo_trainer/` | GSPO | 组采样策略优化 |
| `reinforce_plus_plus_trainer/` | Reinforce++ | 改进的 REINFORCE |
| `remax_trainer/` | ReMax | 基于最大奖励的策略梯度 |
| `rloo_trainer/` | RLOO | 留一法策略优化 |
| `dppo_trainer/` | DPPO | 分布式 PPO |
| `gdpo_trainer/` | GDPO | 组直接偏好优化 |
| `gmpo_trainer/` | GMPO | 组多任务策略优化 |
| `gpg_trainer/` | GPG | 组策略梯度 |
| `sapo_trainer/` | SAPO | 结构化自适应策略优化 |
| `cispo_trainer/` | CISPO | 对比迭代策略优化 |
| `otb_trainer/` | OTB | 在线微调 |
| `mtp_trainer/` | MTP | 多 Token 预测 |
| `on_policy_distillation_trainer/` | 在线蒸馏 | 训练时蒸馏 |
| `sft/` | SFT | 监督微调 |

---

## 14. 数据流全景图

```
配置入口: verl/trainer/config/ (Hydra YAML)
     │
     ▼
main_ppo.py / main_ppo_sync.py (Ray 初始化)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  single_controller (RayWorkerGroup 创建与管理)        │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │           RayPPOTrainer (训练循环)            │    │
│  │                                              │    │
│  │  ① Rollout ──→ ② Reward ──→ ③ Advantage     │    │
│  │  ④ Actor Update ←── ⑤ Critic Update         │    │
│  └──────┬──────────────────────────┬───────────┘    │
│         │                          │                 │
│         ▼                          ▼                 │
│  ┌──────────────┐         ┌──────────────┐         │
│  │ TrainingWorker│         │ BaseRollout  │         │
│  │ (engine/)    │◄────────│ (rollout/)   │          │
│  │ FSDP/MCore/  │ 权重同步 │ vLLM/SGLang/ │         │
│  │ TorchTitan/  │checkpoint│ TRTLLM/HF    │         │
│  │ VeOmni/Auto  │ _engine  │              │          │
│  └──────────────┘         └──────────────┘         │
│         │                                            │
│         ▼                                            │
│  ┌──────────────────┐                               │
│  │ DataProto        │  (统一数据交换协议)            │
│  │ batch + meta_info│                               │
│  └──────────────────┘                               │
└──────────────────────────────────────────────────────┘
```

**PPO 训练五步循环：**

1. **Rollout** — Actor 通过推理引擎生成回复
2. **Reward** — 奖励模型或可验证奖励函数评分
3. **Advantage** — GAE 或其他优势估计器计算优势
4. **Actor Update** — PPO 损失 + 可选的 KL 惩罚
5. **Critic Update** — 对价值目标的 MSE 回归（GRPO/RLOO 等算法可跳过）

---

## 15. `verl/utils/` — 共享工具库

```
utils/
├── dataset/              # 数据集加载器（RL、RM、多轮SFT、视觉）
├── reward_score/         # 奖励函数实现（gsm8k、math_verify、geo3k等）
├── checkpoint/           # 检查点处理器
├── profiler/             # 性能剖析
├── megatron/             # Megatron 工具集
├── vllm/                 # vLLM 工具/补丁
├── sglang/               # SGLang FP8 工具
├── trtllm/               # TRTLLM FP8 工具
├── veomni/               # VeOmni 路由重放
├── modelopt/             # NVIDIA ModelOpt（QAT、量化）
├── qat/                  # 量化感知训练
├── kernel/               # 自定义 CUDA 内核（FP8、交叉熵）
├── logger/               # 日志聚合器
├── debug/                # 调试工具
├── experimental/         # 实验性张量函数
├── tensordict_utils.py   # TensorDict 操作
├── torch_functional.py   # 自定义 torch 函数
├── config.py             # OmegaConf ↔ dataclass 转换
├── distributed.py        # 分布式进程组初始化
├── device.py             # 设备检测
├── seqlen_balancing.py   # 序列长度均衡
└── chat_template.py      # 聊天模板处理
```

---

## 16. 测试结构 (`tests/`)

测试目录镜像 `verl/` 包结构：

| 目录 | 用途 |
|---|---|
| `tests/trainer/` | 训练器、PPO 核心算法、指标 |
| `tests/workers/` | 工作器配置、奖励管理、rollout |
| `tests/models/` | 引擎测试、transformer 测试、内核测试 |
| `tests/single_controller/` | 装饰器、工作器组、数据传递 |
| `tests/checkpoint_engine/` | 权重同步验证 |
| `tests/utils/` | 数据集、检查点、Megatron 工具 |
| `tests/experimental/` | Agent loop、fully_async_policy |
| `tests/tools/` | 函数工具测试 |
| `tests/special_distributed/` | 分布式测试（需多GPU） |
| `tests/special_e2e/` | 端到端测试（需GPU） |
| `tests/special_sanity/` | 代码质量检查 |
| `tests/special_standalone/` | 独立测试（内存缓冲区） |
| `tests/special_npu/` | NPU 专用 CI 测试 |

文件命名规范：`*_on_cpu.py`（CPU测试）、`*_on_gpu.py`（需GPU）、`special_distributed`（需多GPU分布式环境）。

---

## 17. 技术栈总结

| 层级 | 技术选择 |
|---|---|
| 分布式框架 | Ray |
| 配置管理 | Hydra + OmegaConf |
| 训练后端 | FSDP/FSDP2、Megatron-LM、TorchTitan、VeOmni |
| 推理后端 | vLLM、SGLang、TensorRT-LLM、HuggingFace |
| 模型格式 | HuggingFace Transformers、Megatron Core |
| 数据容器 | TensorDict + DataProto |
| 检查点同步 | NCCL、HCCL、NIXL、Mooncake |
| 代码规范 | Ruff、mypy、pre-commit |
| 性能剖析 | PyTorch Profiler、NPU Profiler、NVTX |
| 实验追踪 | Weights & Biases、TensorBoard |

---

*本报告生成于 2026-05-28，基于 verl 项目当前 `main` 分支代码分析。*
