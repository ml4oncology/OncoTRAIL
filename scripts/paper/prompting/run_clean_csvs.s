#!/bin/bash
#SBATCH --job-name=clean_csvs
#SBATCH --output=log_files/clean_csvs_%j.out
#SBATCH --error=log_files/clean_csvs_%j.err
#SBATCH --mem=4GB
#SBATCH --time=0-03:00:00
#SBATCH --partition=all

# Load python module
module load python3

# Set the path to your conda environment's python
PYTHON_ENV=~/miniforge3/envs/OncoTRAIL/bin/python

# Path to the clean_csvs.py script
SCRIPT_PATH="/cluster/home/t127556uhn/gitrepo/2024/OncoTRAIL/src/prompt/clean_csvs.py"

# Parse the stage argument
stage=$1

if [[ "$stage" == "stage1" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage1
elif [[ "$stage" == "stage2" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage2
elif [[ "$stage" == "stage3" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage3
elif [[ "$stage" == "train" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/train
elif [[ "$stage" == "test" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/test
elif [[ "$stage" == "inference" ]]; then
    BASE_PATH=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/inference
else
    echo "Error: unknown stage '$stage'"
    echo "Valid stages are: stage1, stage2, stage3, train, test, inference"
    exit 1
fi

echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "======================================"
echo ""
echo "Stage:     $stage"
echo "BASE_PATH: $BASE_PATH"
echo ""

# Run the Python script
$PYTHON_ENV $SCRIPT_PATH $BASE_PATH

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"