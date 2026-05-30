#!/usr/bin/env bash
# On-policy distillation | multi-teacher | Qwen3-Omni thinker-only student | vLLM/vLLM-Omni rollout | FSDP training

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Thinking}
TEXT_TEACHER_MODEL=${TEXT_TEACHER_MODEL:-Qwen/Qwen3-32B}
OMNI_TEACHER_MODEL=${OMNI_TEACHER_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Thinking}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}
USE_VLLM_OMNI=${USE_VLLM_OMNI:-False}
OMNI_SCHEDULER_CLS=${OMNI_SCHEDULER_CLS:-vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Per-teacher replicas; total teacher GPUs = sum(num_replicas) * teacher_tp
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS_TEXT=${TEACHER_NUM_REPLICAS_TEXT:-1}
TEACHER_NUM_REPLICAS_OMNI=${TEACHER_NUM_REPLICAS_OMNI:-1}
teacher_tp=${TEACHER_TP:-2}
TEACHER_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_TEXT + TEACHER_NUM_REPLICAS_OMNI) * teacher_tp ))

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}
USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-False}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_mopd_qwen3_omni}
experiment_name=${EXPERIMENT_NAME:-qwen3_omni_thinker_mopd_vllm_fsdp}
# ---- end user-adjustable ----

text_train=${TEXT_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
text_test=${TEXT_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
omni_train=${OMNI_TRAIN_FILE:-$HOME/data/qwen3_omni/train.parquet}
omni_test=${OMNI_TEST_FILE:-$HOME/data/qwen3_omni/test.parquet}

TEXT_DATA_SOURCE=${TEXT_DATA_SOURCE:-openai/gsm8k}
OMNI_DATA_SOURCE=${OMNI_DATA_SOURCE:-qwen3_omni}

train_files="['$text_train', '$omni_train']"
val_files="['$text_test', '$omni_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.image_key=images
    data.video_key=videos
    data.audio_key=audios
    +data.mm_processor_kwargs.use_audio_in_video=${USE_AUDIO_IN_VIDEO}
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}
    actor_rollout_ref.model.load_thinker_only=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.enable_audio_output=False
)

if [ "${USE_VLLM_OMNI}" = True ]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.omni=True
        +actor_rollout_ref.rollout.engine_kwargs.vllm.model_stage=thinker
        +actor_rollout_ref.rollout.engine_kwargs.vllm.model_arch=Qwen3OmniMoeForConditionalGeneration
        +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_type=ar
        +actor_rollout_ref.rollout.engine_kwargs.vllm.scheduler_cls=${OMNI_SCHEDULER_CLS}
        +actor_rollout_ref.rollout.engine_kwargs.vllm.engine_output_type=text
        +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_config_name=thinker_config
    )
fi

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Multi-teacher: route by each sample's `data_source` value.
# The Omni teacher is inference-only, but we still disable audio output so vLLM does not load talker/code2wav.
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    # --- text teacher ---
    +distillation.teacher_models.text.key="$TEXT_DATA_SOURCE"
    +distillation.teacher_models.text.model_path="$TEXT_TEACHER_MODEL"
    +distillation.teacher_models.text.num_replicas=${TEACHER_NUM_REPLICAS_TEXT}
    +distillation.teacher_models.text.inference.name=vllm
    +distillation.teacher_models.text.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.text.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.text.inference.max_model_len=${max_num_tokens}
    # --- omni teacher ---
    +distillation.teacher_models.omni.key="$OMNI_DATA_SOURCE"
    +distillation.teacher_models.omni.model_path="$OMNI_TEACHER_MODEL"
    +distillation.teacher_models.omni.num_replicas=${TEACHER_NUM_REPLICAS_OMNI}
    +distillation.teacher_models.omni.inference.name=vllm
    +distillation.teacher_models.omni.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.omni.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.omni.inference.max_model_len=${max_num_tokens}
    +distillation.teacher_models.omni.inference.engine_kwargs.vllm.hf_overrides.enable_audio_output=False
    # --- loss ---
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

if [ "${USE_VLLM_OMNI}" = True ]; then
    EXTRA+=(
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.omni=True
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.model_stage=thinker
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.model_arch=Qwen3OmniMoeForConditionalGeneration
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.worker_type=ar
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.scheduler_cls=${OMNI_SCHEDULER_CLS}
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.engine_output_type=text
        +distillation.teacher_models.omni.inference.engine_kwargs.vllm.hf_config_name=thinker_config
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
