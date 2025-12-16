#!/bin/bash
#SBATCH --job-name=verl            # A name for your job. Visible in squeue.
#SBATCH --account=infra01          # The account you are charged for the job
#SBATCH --nodes=1                    # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1          # Tasks (processes) per node
#SBATCH --time=05:00:00              # HH:MM:SS, set a time limit for this job (here 10min)
#SBATCH --partition=normal           # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --environment=jahongir  # the environment to use (See [4/7])
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

# Check GPU availability
echo "Checking GPU..."
nvidia-smi

# Setting home dirs for hf, torch, and wandb
export HF_HOME=$HOME/scratch/cacheX/huggingface_cache
mkdir -p $HF_HOME
export TORCH_HOME=$HOME/scratch/cacheX/torch_cache
mkdir -p $TORCH_HOME
export WANDB_DIR=$HOME/scratch/cacheX/wandb
mkdir -p $WANDB_DIR
echo "HF, WANDB, and TORCH dirs are set"

# installing dependencies
pip install -q qwen_vl_utils mathruler math-verify weave gql==3.5.3
pip install -q -e /users/msayfiddinov/shared/users/msayfiddi/verl_utils/decord/python
export LD_LIBRARY_PATH=/users/msayfiddinov/shared/users/msayfiddi/verl_utils/ffmpeg4/lib:$LD_LIBRARY_PATH

# # building datasets
# python $HOME/verl/data/generate_random_rotations.py --output_dir $HOME/scratch/verl_data/rotations
# python $HOME/verl/data/generate_random_sizes.py --output_dir $HOME/scratch/verl_data/sizes
# python3 $HOME/verl/examples/data_preprocess/custom_multiturn_w_tool_agent_rotate.py  --local_dir="$HOME/scratch/verl_data/custom_multiturn_w_tool_rotate"

# training
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-3B-Instruct')"
unset ROCR_VISIBLE_DEVICES
unset SSL_CERT_FILE
# otherwise it causes the following error: "Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set."

bash examples/sglang_multiturn/custom/grpo_multiturn_rotate.sh

#end  
echo "==========================="
echo "Run has finished at $(date)"
echo "==========================="
