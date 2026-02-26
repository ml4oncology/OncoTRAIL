import argparse
import submitit
from datetime import datetime
import os
import numpy as np
# from ml_common.util import load_table
import pandas as pd
# from oncotrail.prompt.helper import prompt_llm
# from oncotrail.prompt.run_prompting import LLMPromptRunner
import sys

def launch(cfg):
    """Use submitit to launch jobs in the SLURM cluster

    References: 
    - https://www.unitary.ai/articles/intro-to-multi-node-machine-learning-2-using-slurm
    - https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
    """

    save_dir = cfg['save_dir']

    # Initialize the executor, which is the submission interface
    log_file_save = f"{save_dir}/log_files/{datetime.now().replace(microsecond=0)}"
    os.makedirs(log_file_save, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_file_save)

    # load parameters for splitting dataframe
    n_partitions = cfg['n_partitions']
    data_dir = cfg['data_dir']
    df_name = cfg['file_name']

    file_name_no_ext = os.path.splitext(df_name)[0]
    if '.parquet.gzip' in df_name:       
        file_name_no_ext = os.path.splitext(file_name_no_ext)[0]
 
    # create save_dir
    param_string = f"{file_name_no_ext}_{cfg['LLM_name']}_{cfg['quant_level']}_{cfg['start_date']}_{cfg['end_date']}"
    param_string = f"{param_string}_{cfg['random_sampling']}_{cfg['n_few_shot']}_{cfg['numeric_proba']}"
    param_string = f"{param_string}_{cfg['prompt_num']}_{cfg['top_k']}_{cfg['min_p']}_{cfg['top_p']}"
    param_string = f"{param_string}_{cfg['temperature']}"
    save_dir = f"{save_dir}/{param_string}"
    data_dir = f"{data_dir}/data_partitions/{file_name_no_ext}"

    if cfg['n_few_shot'] != 0:
        # create a new key
        if cfg['mode'] == "train":
            cfg['few_shot_dir'] = data_dir
        elif cfg['mode'] == "inference":
            cfg['few_shot_dir'] = cfg['few_shot_train_dir']

    cfg.pop('file_name')
    cfg.pop('data_dir')
    cfg.pop('save_dir')

    data_path_partitions = f'{data_dir}/randomsampling{cfg["random_sampling"]}_{cfg["start_date"]}_{cfg["end_date"]}'
    # check if data_path_partitions exists if not error
    if not os.path.exists(data_path_partitions):
        raise ValueError(f"Data path partitions {data_path_partitions} does not exist")

    # edit this depending on use_vllm value

    # Specify the Slurm parameters
    # TODO: put this in another config file
    if cfg['use_vllm'] == 0 or cfg['vllm_mode'] == 'online':
        if cfg['use_vllm'] == 0:

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
            
            if cfg['llama_cpp_mode'] == 'sequential':
                
                if cfg['gpu_constraint'] == 1:
                    executor.update_parameters(constraint="gpu32g")
                
                def run_llm_prompt(cfg: dict):
                    from oncotrail.prompt.local_runner_sequential import LocalLLMRunnerSequential 
                    runner = LocalLLMRunnerSequential(cfg)
                    runner.run()
            else:
                
                executor.update_parameters(
                    mem_gb=64,
                    slurm_additional_parameters={
                        "account": "grantgroup_gpu",
                        "nodelist": "node159",
                    }
                )

                def run_llm_prompt(cfg: dict):
                    from oncotrail.prompt.local_runner_parallel import LocalLLMRunnerParallel 
                    runner = LocalLLMRunnerParallel(cfg)
                    runner.run()

        else:
            executor.update_parameters(  
                # slurm_account="gliugroup_gpu",      
                slurm_partition="all",
                nodes=1, # Each job in the job array gets one node
                mem_gb=cfg['memory'], # Each job gets 4GB of memory
                timeout_min=cfg['n_hours'] * 60, # Limit the job running time to 2 days
            )
            
            def run_llm_prompt(cfg: dict):
                from oncotrail.prompt.vllm_runner_online import VLLMRunnerOnline
                runner = VLLMRunnerOnline(cfg)
                runner.run()

        cfgs = []

        # for partition_id, idxs in enumerate(partition_list):
        for partition_id in range(n_partitions):

            cfgs.append(dict(data_dir=f'{data_path_partitions}', 
                            file_name=f'{partition_id}_{df_name}',
                            save_dir=save_dir, **cfg))

        # Submit your function and inputs as a job array
        # jobs = executor.map_array(prompt_llm, cfgs)
        jobs = executor.map_array(run_llm_prompt, cfgs)

        # Monitor jobs to keep track of completed jobs
        submitit.helpers.monitor_jobs(jobs)

    else:
        
        # in data_path_partitions, merge all *_{df_name}.csv into one dataframe
        df = pd.DataFrame()
        for partition_id in range(n_partitions):
            df_partition = pd.read_csv(f'{data_path_partitions}/{partition_id}_{df_name}')
            df = pd.concat([df, df_partition], ignore_index=True)
        
        df = df.reset_index(drop=True)

        # if {data_path_partitions}/merged_{df_name} exists, skip saving
        if not os.path.exists(f'{data_path_partitions}/merged_{df_name}'):
            df.to_csv(f'{data_path_partitions}/merged_{df_name}', index=False)

        cfg['data_dir'] = data_path_partitions
        cfg['file_name'] = f'merged_{df_name}'
        cfg['save_dir'] = save_dir

        from oncotrail.prompt.vllm_runner_offline import VLLMRunnerOffline
        runner = VLLMRunnerOffline(cfg)
        runner.run()

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
    parser.add_argument("tokenizer_path", help="name of tokenizer", type=str)  # name of tokenizer
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

    # Mode arguments
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="mode: train or inference (default: train)")
    parser.add_argument("--few_shot_train_dir", type=str, help="directory for few shot examples from the training set")

    # New llama_cpp-related arguments
    parser.add_argument("--llama_cpp_mode", type=str, choices=["sequential", "parallel"], default="sequential", 
                        help="llama cpp mode: sequential or parallel (default: sequential)")

    # New vLLM-related arguments
    parser.add_argument("use_vllm", type=int, choices=[0, 1], help="whether to use vLLM (0 or 1)")
    parser.add_argument("--base_url", type=str, default=None, help="base URL for vLLM server")
    parser.add_argument("--vllm_model_name", type=str, default=None, help="model name on vLLM server")
    parser.add_argument("--vllm_mode", type=str, choices=["online", "offline"], default="online", 
                        help="vLLM mode: 'online' for server API or 'offline' for local inference (default: online)")
    
    # New shapley-related arguments
    parser.add_argument("add_tabularML_prediction", type=int, choices=[0, 1, 2, 3], help="whether to add shapley (1), linear regression coefficient (2), or (3) tabular prediction or not (0)")
    parser.add_argument("--shapley_path", type=str, default=None, help="path for shapley coefficients")
    parser.add_argument("--log_reg_path", type=str, default=None, help="path for logistic regression coefficients")

    cfg = vars(parser.parse_args())

    if cfg["use_vllm"] == 1:
        if cfg["vllm_mode"] == "online":
            if cfg["base_url"] is None or cfg["vllm_model_name"] is None:
                print("Error: --base_url and --vllm_model_name must be provided when use_vllm is 1.")
                sys.exit(1)

    if cfg["add_tabularML_prediction"] > 0 and cfg["shapley_path"] is None:
        print("Error: --shapley_path must be provided when add_tabularML_prediction is 1 or 2.")
        sys.exit(1)

    if cfg["add_tabularML_prediction"] == 2 and cfg["log_reg_path"] is None:
        print("Error: --log_reg_path must be provided when add_tabularML_prediction is 2.")
        sys.exit(1)    

    launch(cfg)