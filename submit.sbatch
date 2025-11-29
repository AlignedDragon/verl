#!/bin/bash
#SBATCH --job-name=verl            # A name for your job. Visible in squeue.
#SBATCH --account=infra01          # The account you are charged for the job
#SBATCH --nodes=1                    # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1          # Tasks (processes) per node
#SBATCH --time=05:00:00              # HH:MM:SS, set a time limit for this job (here 10min)
#SBATCH --partition=normal           # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --environment=verl  # the environment to use (See [4/7])
#SBATCH --output=/users/msayfiddinov/scratch/verl_logs/node/%j.out            # log file for stdout, prints, et cetera
#SBATCH --error=/users/msayfiddinov/scratch/verl_logs/node/%j.err             # log file for stderr, errors

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail

# Job info
echo "==========================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "==========================="

# Navigate to verl directory
cd /workspace/src/verl

# Check GPU availability
echo "Checking GPU..."
nvidia-smi

# Setting home dirs for hf, and torch
export HF_HOME=$HOME/scratch/cacheX/huggingface_cache
mkdir -p $HF_HOME
export TORCH_HOME=$HOME/scratch/cacheX/torch_cache
mkdir -p $TORCH_HOME
echo "HF_HOME and TORCH_HOME are now set"

# Check if verl is installed
python -c "import verl; print(f'verl version: {verl.__version__}')" || echo "verl import check"

# Run PPO training
echo "Starting PPO training..."

set -x


python3 -m verl.trainer.main_ppo \
   data.train_files=$HOME/scratch/verl_data/gsm8k/train.parquet \
   data.val_files=$HOME/scratch/verl_data/gsm8k/test.parquet \
   data.train_batch_size=1024 \
   data.max_prompt_length=512 \
   data.max_response_length=512 \
   actor_rollout_ref.model.path=Qwen/Qwen2-0.5B-Instruct \
   actor_rollout_ref.actor.optim.lr=1e-6 \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.actor.ppo_mini_batch_size=256 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
   actor_rollout_ref.actor.fsdp_config.param_offload=False \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   actor_rollout_ref.rollout.name=sglang \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   critic.optim.lr=1e-5 \
   critic.model.use_remove_padding=True \
   critic.model.path=Qwen/Qwen2-0.5B-Instruct \
   critic.model.enable_gradient_checkpointing=True \
   critic.ppo_micro_batch_size_per_gpu=32 \
   critic.model.fsdp_config.param_offload=False \
   critic.model.fsdp_config.optimizer_offload=False \
   algorithm.kl_ctrl.kl_coef=0.001 \
   trainer.critic_warmup=0 \
   trainer.logger='["console","wandb"]' \
   trainer.project_name='verl_example_gsm8k' \
   trainer.experiment_name='qwen_function_rm' \
   trainer.n_gpus_per_node=4 \
   trainer.nnodes=1 \
   trainer.save_freq=-1 \
   trainer.test_freq=1 \
   trainer.total_epochs=15 $@

#end  
echo "==========================="
echo "Run has finished at $(date)"
echo "==========================="
