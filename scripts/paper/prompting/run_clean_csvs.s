#!/bin/bash
#SBATCH --job-name=clean_csvs
#SBATCH --output=clean_csvs_%j.out
#SBATCH --error=clean_csvs_%j.err
#SBATCH --mem=16GB
#SBATCH --time=0-03:00:00
#SBATCH --partition=all

# Load python module
module load python3

# Set the path to your conda environment's python
PYTHON_ENV=~/miniforge3/envs/LLMfinetune/bin/python

# Set the base path where your target directories are located
# CHANGE THIS to your actual path
BASE_PATH="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/paper/pmh_method/methods/prompting/train_test/stage3"

# Path to the clean_csvs.py script
# CHANGE THIS to the actual path of your script
SCRIPT_PATH="/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/src/prompt/clean_csvs.py"

echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "======================================"
echo ""

# Run the Python script
$PYTHON_ENV $SCRIPT_PATH $BASE_PATH

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"