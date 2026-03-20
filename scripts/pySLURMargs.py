#!/usr/bin/python -u
import tempfile
import os
import sys
import time

"""Python script to submit jobs to the cluster with
   SLURM workload manager. Best used in conjunction
   with a bash script. The script below prints slurm
   commands to the terminal to submit the job.

   Inputs: 
   user_name: user name log-in for the cluster
   memory: memory needed for the job (in GB)
   conda_env: path to the conda environment
   n_GPU: number of GPU's requested. if 0, run using CPU
   mcmd: python script to be ran on the cluster and all 
         its input arguments
"""

# determine host
host = os.uname()[1]

# get temp file
fname = tempfile.mkstemp()
fname = fname[1]
ffname = fname.split('/')
ffname = ffname[-1]

# get system parameters
pwd = os.getcwd()
user_name = sys.argv[1]
memory = sys.argv[2]
conda_env = sys.argv[3]
n_GPU = sys.argv[4]
run_time = sys.argv[5]

# Parse optional args
specific_node = None
extra_strings = ''
group_name = 'grantgroup_gpu'  # Default
gpu_type = None                # NEW
arg_start = 6

# Check for specific node
if len(sys.argv) > arg_start and sys.argv[arg_start].startswith("node"):
    specific_node = sys.argv[arg_start]
    arg_start += 1

# Check for extra SLURM strings
if len(sys.argv) > arg_start and sys.argv[arg_start].startswith("extras="):
    extra_strings = sys.argv[arg_start][len("extras="):].replace("\\n", "\n")
    arg_start += 1

# Check for custom SLURM account
if len(sys.argv) > arg_start and sys.argv[arg_start].startswith("group="):
    group_name = sys.argv[arg_start][len("group="):]
    arg_start += 1

# Check for GPU type
if len(sys.argv) > arg_start and sys.argv[arg_start].startswith("gpu_type="):
    gpu_type = sys.argv[arg_start][len("gpu_type="):]
    arg_start += 1

# Remaining args are the command to run
mcmd = sys.argv[arg_start:]

os.environ['PATH'] = ('/usr/local/slurm/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/cluster/home/' 
                      + user_name + 
                      '/.local/bin:/cluster/home/' 
                      + user_name + 
                      '/bin')

# create a log directory
try:
    os.mkdir( pwd + '/log_files' )
except:
    pass

fp = open(fname, 'w')
fp.write('#!/bin/bash\n\n')
fp.write('#SBATCH -o ' + pwd + '/log_files/log_' + ffname + '.txt\n')
fp.write('#SBATCH -D ' + pwd + '\n')
fp.write('#SBATCH -J py\n')
fp.write('#SBATCH --get-user-env\n')
fp.write('#SBATCH --ntasks=1\n')
fp.write('#SBATCH --mem=' + memory + 'GB\n')
fp.write('#SBATCH --time=' + run_time + '\n')
if int(n_GPU) > 0:
    if specific_node == 'node159':
        fp.write('#SBATCH --partition=gpu_grantgroup\n')
    else:
        fp.write('#SBATCH --partition=gpu\n')
    fp.write('#SBATCH --account=' + group_name + '\n')

    if gpu_type:
        fp.write(f'#SBATCH --gres=gpu:{gpu_type}:{n_GPU}\n')
    else:
        fp.write(f'#SBATCH --gres=gpu:{n_GPU}\n')
# if run time is less than 4 hrs and memory is less than 8, run on 'short' partition
# elif int(run_time.split('-')[-1].split(':')[0]) <= 4 and int(memory) <= 8:
#     fp.write('#SBATCH -p short'+'\n')
else:
    fp.write('#SBATCH -p all'+'\n')
if specific_node:
    fp.write(f'#SBATCH --nodelist={specific_node}\n')
fp.write('\n')

fp.write('module load python3\n')
if extra_strings:
    fp.write(extra_strings + '\n\n')
fp.write('\n')
for i in range(len(mcmd)):
    if(i + 1 == len(mcmd)):
        fp.write( conda_env + ' -u ' + mcmd[i] + '\n\n')
    else:
        fp.write( conda_env + ' -u ' + mcmd[i] + '&\n\n')
fp.close()

fp = open(fname, 'r')
print('--- BEGIN SLURM ---')
for l in fp:
    print(l),
print('--- END SLURM ---')
time.sleep(3)
os.system('/usr/local/slurm/bin/sbatch ' + fname)
