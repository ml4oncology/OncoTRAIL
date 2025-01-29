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
mcmd = sys.argv[6:]

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

ADDSLURM = ''
if(host == 'voyager'):
    # change home to master
    pwd = '/master' + pwd
    ADDSLURM += '#SBATCH --cpus-per-task=2\n'
    ADDSLURM += 'export OMP_NUM_THREADS=2\n'
if(host == 'mac-login-amd'):
    ADDSLURM += '#SBATCH --partition=bdz'

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
    fp.write('#SBATCH --partition=gpu\n')
    fp.write('#SBATCH --account=gliugroup_gpu\n')
    fp.write('#SBATCH --gres=gpu:'+ n_GPU +'\n')
    # fp.write('#SBATCH --gpus=p100:'+ n_GPU +'\n')
    # fp.write('#SBATCH --gres=gpu:1' +'\n')
    # fp.write('#SBATCH -C "gpu32g"' +'\n')
else:
    fp.write('#SBATCH -p all')
fp.write(ADDSLURM)
fp.write('\n')

fp.write('module load python3\n')
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
