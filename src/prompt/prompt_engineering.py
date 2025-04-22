import argparse
import submitit
from datetime import datetime
import os
import numpy as np
from ml_common.util import load_table
import pandas as pd
from llm_notes_classification.prompt.helper import prompt_llm

def launch(cfg):
    """Use submitit to launch jobs in the SLURM cluster

    References: 
    - https://www.unitary.ai/articles/intro-to-multi-node-machine-learning-2-using-slurm
    - https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
    """

    # few shot examples not implemented yet
    # if cfg['n_few_shot'] != 0:
    #     raise NotImplementedError("few shot examples not implemented yet")

    save_dir = cfg['save_dir']

    # Initialize the executor, which is the submission interface
    log_file_save = f"{save_dir}/log_files/{datetime.now().replace(microsecond=0)}"
    os.makedirs(log_file_save, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_file_save)

    # Specify the Slurm parameters
    # TODO: put this in another config file
    executor.update_parameters(  
        # slurm_account="gliugroup_gpu",      
        slurm_partition="gpu",
        nodes=1, # Each job in the job array gets one node
        mem_gb=cfg['memory'], # Each job gets 4GB of memory
        timeout_min=cfg['n_hours'] * 60, # Limit the job running time to 2 days
        slurm_gpus_per_node=1, # Each node should use 1 GPU
        gres="gpu:1",              # Request 1 GPU
        slurm_additional_parameters={
            "account": "gliugroup_gpu",
        }
    )

    if cfg['gpu_constraint'] == 1:
        executor.update_parameters(constraint="gpu32g")

    # load parameters for splitting dataframe
    n_partitions = cfg['n_partitions']
    data_dir = cfg['data_dir']
    df_name = cfg['file_name']
    
    # read dataframe
    df = load_table(f'{data_dir}/{df_name}')

    if '.parquet.gzip' in df_name:
        file_name_no_ext = os.path.splitext(df_name)[0]
        file_name_no_ext = os.path.splitext(file_name_no_ext)[0]
    else:
        file_name_no_ext = os.path.splitext(df_name)[0]

    # create save_dir
    param_string = f"{file_name_no_ext}_{cfg['LLM_name']}_{cfg['quant_level']}_{cfg['start_date']}_{cfg['end_date']}"
    param_string = f"{param_string}_{cfg['random_sampling']}_{cfg['n_few_shot']}_{cfg['numeric_proba']}"
    param_string = f"{param_string}_{cfg['prompt_num']}_{cfg['top_k']}_{cfg['min_p']}_{cfg['top_p']}"
    param_string = f"{param_string}_{cfg['temperature']}"
    save_dir = f"{save_dir}/{param_string}"
    # data_dir = f"{data_dir}/{param_string}/data_partitions/"
    data_dir = f"{data_dir}/data_partitions/{file_name_no_ext}"

    os.makedirs(f"{data_dir}", exist_ok=True)

    # list of targets
    list_of_targets = cfg['target_names'].split(",")

    if cfg['n_few_shot'] != 0:
        for target in list_of_targets:
            # find indices where target is not -1
            df_few_shot = df.loc[df['treatment_date']<cfg['few_shot_date']].loc[df['note_summary'].notna()].loc[df[target] != -1].copy()
            # sample a few examples from df_few_shot
            df_few_shot_sample_pos = (
                df_few_shot.loc[df_few_shot[target] == 1]
                .sample(n=round(cfg['n_few_shot']/2), replace=False, random_state=42)
                )
            df_few_shot_sample_neg = (
                df_few_shot.loc[df_few_shot[target] == 0]
                .sample(n=cfg['n_few_shot']-df_few_shot_sample_pos.shape[0], replace=False, random_state=42)
                )
            
            # fix this later if there is not enough samples
            df_few_shot_sample = (pd.concat([df_few_shot_sample_pos, df_few_shot_sample_neg])
                                    .reset_index(drop=True)
                                    )
            few_shot_examples_fname = f"few_shot_{target}_nfewshot_{cfg['n_few_shot']}_{cfg['few_shot_date']}.csv"
            if not os.path.isfile(f'{data_dir}/{few_shot_examples_fname}'):
                df_few_shot_sample[['mrn','treatment_date','note','note_summary',target]].to_csv(f'{data_dir}/{few_shot_examples_fname}', 
                                                        index=False)

            # create a new key
            cfg['few_shot_dir'] = data_dir

            # possibly delete few_shot_file_path from the variable inputs
            # what if there are not enough positive examples?

    # restrict the data frame to time period
    df = df[df['treatment_date'].between(cfg['start_date'], cfg['end_date'])].copy()

    partition_list = np.array_split(df.index, n_partitions)

    # if random sampling, change targets of discarded rows to -1
    if cfg['random_sampling'] == 1:
        n_class_samples = 300
        for target in list_of_targets:
            # randomly select n_class_samples indices
            df_positive = df[df[target] == 1].copy()
            positive_idxs = df_positive.sample(n=np.min([n_class_samples, df_positive.shape[0]]), 
                                               replace=False, 
                                               random_state=42).index
            
            df_negative = df[df[target] == 0].copy()
            n_neg_samples = n_class_samples
            if df_positive.shape[0] < n_class_samples:
                n_neg_samples = 2*n_class_samples - df_positive.shape[0]
            negative_idxs = df_negative.sample(n=n_neg_samples, 
                                               replace=False, 
                                               random_state=42).index
            
            # take the union of positive and negative indices
            idxs = np.union1d(positive_idxs, negative_idxs)

            # for the rest of the indices, set the target to -1
            non_idxs = np.setdiff1d(df.index, idxs)

            df.loc[non_idxs, target] = -1

        # re-arrange data frame so that the number of queries across parts is roughly the same
        non_negatives = (df[list_of_targets] != -1).astype(int)
        df['non_negative_count'] = non_negatives.sum(axis=1)
        df.sort_values('non_negative_count', ascending=False, inplace=True)
        group_sums = [0] * n_partitions  # Keep track of the sum of non_negative_count in each group
        group_assignments = []  # To store group assignment for each row

        # greedily assign rows to groups
        for idx, row in df.iterrows():
            # Find the group with the minimum sum of non_negative_count
            min_group_index = np.argmin(group_sums)
            
            # Assign the row to that group
            group_assignments.append(min_group_index)
            
            # Update the sum of the assigned group
            group_sums[min_group_index] += row['non_negative_count']

        df['group'] = group_assignments
        partition_list = [[] for _ in range(n_partitions)]

        for idx, group in zip(df.index, df['group']):
            partition_list[group].append(idx)
        
        # drop 'group' and 'non_negative_count' columns from df
        df.drop(columns=['group', 'non_negative_count'], inplace=True)

    cfg.pop('file_name')
    cfg.pop('data_dir')
    cfg.pop('save_dir')

    cfgs = []

    data_path_partitions = f'{data_dir}/randomsampling{cfg["random_sampling"]}_{cfg["start_date"]}_{cfg["end_date"]}'
    os.makedirs(data_path_partitions, exist_ok=True)

    for partition_id, idxs in enumerate(partition_list):
        partition_path = f'{data_path_partitions}/{partition_id}_{df_name}'
        # if not os.path.isfile(partition_path):
        if partition_path.endswith('.csv'):
            df.loc[idxs].reset_index(drop=True).to_csv(partition_path, index=False)
        elif partition_path.endswith(('.parquet','.parquet.gzip')):
            df.loc[idxs].reset_index(drop=True).to_parquet(partition_path, compression='gzip', index=False)

        cfgs.append(dict(data_dir=f'{data_path_partitions}', 
                         file_name=f'{partition_id}_{df_name}',
                         save_dir=save_dir, **cfg))

    # Submit your function and inputs as a job array
    jobs = executor.map_array(prompt_llm, cfgs)

    # Monitor jobs to keep track of completed jobs
    submitit.helpers.monitor_jobs(jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("file_name", help="notes file name", type=str)  # notes file name
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("start_date", help="start date", type=str)  # start date
    parser.add_argument("end_date", help="end date", type=str)  # end date
    parser.add_argument("few_shot_date", help="date cut-off for few shot examples", type=str) # date cut off for few shot examples
    parser.add_argument("random_sampling", help="random sampling", type=int)  # random sampling
    parser.add_argument("n_few_shot", help="number of few shot examples", type=int)  # number of few shot
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("quant_level", help="quantization level", type=str)  # quantization level
    parser.add_argument("num_samples", help="number of samples", type=int)  # number of samples
    parser.add_argument("numeric_proba", help="numerical probability", type=int)  # numeric probability?
    parser.add_argument("prompt_file_dir", help="directory where json files are stored", type=str)  # directory of prompt json files
    parser.add_argument("prompt_num", help="prompt number", type=int)  # prompt number
    parser.add_argument("llama_cpp", help="llama_cpp", type=int)  # llama_cpp
    parser.add_argument("top_k", help="top k", type=float)  # top k
    parser.add_argument("min_p", help="min p", type=float)  # min p
    parser.add_argument("top_p", help="top p", type=float)  # top p
    parser.add_argument("temperature", help="temperature", type=float)  # temperature
    parser.add_argument('target_names', type=str, help='Comma-separated list of targets') # targets
    parser.add_argument("n_partitions", help="number of partitions", type=int)  # number of partitions
    parser.add_argument("n_hours", help="number of hours", type=int)  # number of hours
    parser.add_argument("memory", help="memory of each node", type=int)  # memory of each node
    parser.add_argument("gpu_constraint", help="gpu constraint", type=int)  # constraint on gpu
    cfg = vars(parser.parse_args())
    launch(cfg)