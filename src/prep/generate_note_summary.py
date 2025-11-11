import numpy as np
import pandas as pd
import os
import ast
import argparse
import sys
from llama_cpp import Llama
import torch
from pathlib import Path
import submitit
from datetime import datetime
from ml_common.util import load_table
import logging
logging.basicConfig(
    level=logging.INFO,         # Log level (you can adjust it to INFO, DEBUG, etc.)
    stream=sys.stdout 
)
logger = logging.getLogger(__name__)

# Empty cuda cache
torch.cuda.empty_cache()


def generate_note_summary(cfg: dict):

    data_dir = cfg["data_dir"]
    file_name = cfg["file_name"]
    LLM_path = cfg["LLM_path"]  
    LLM_name = cfg["LLM_name"]
    save_dir = cfg["save_dir"]

    logger.info(f"{save_dir}")

    # create save folder
    os.makedirs(f'{save_dir}', exist_ok=True)

    if 'Gemma' in LLM_name:
        chat_format = "gemma"
    elif 'Qwen' in LLM_name or 'QwQ' in LLM_name:
        chat_format = "chatml"
    else:
        chat_format = "llama-2"
    
    response_format = {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"Summary": {"type": "string"}},
            "required": ["Summary"],
        },
    }

    # load the clinical notes file
    clinical_notes_df = load_table(f"{data_dir}/{file_name}")

    logger.info(f"loaded file {file_name}")

    for _, row in clinical_notes_df.iterrows():

        note = row["note"]
        mrn = row["mrn"]
        treatment_date = row["treatment_date"]

        logger.info(f"MRN: {mrn}\n")

        # check if file already exists, otherwise, skip!
        if os.path.isfile(
            f"{save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_summary.csv"
        ):
            continue
        
        system_instructions = """
            Please summarize the following clinical note, retaining all key information 
            relevant for predicting the risk of adverse events for this patient. Focus on the patient's medical history, 
            current chemotherapy regimen, laboratory values, symptoms, previous adverse events, and any other pertinent 
            information that could impact the risk of the following outcomes: death, emergency department visit, worsening 
            laboratory values according to the CTCAE (Common Terminology Criteria for Adverse Events), or changes in the 
            Edmonton Symptom Assessment System score. The summary should be concise yet include any details that are important 
            for understanding the patient's current health status and potential risks.
        """

        messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": note},
            ]

        results = []

        try:
            llm = Llama(model_path=LLM_path, n_gpu_layers=-1, main_gpu=0,
                chat_format=chat_format, seed=42, n_ctx=8192, flash_attn=False)

            sequences = llm.create_chat_completion(messages=messages, 
                                    response_format=response_format, 
                                    max_tokens=1500)
            logger.info("Generated LLM response.")
            llm = None

            raw_string = sequences['choices'][0]['message']['content']
            start_idx = raw_string.find("{")
            end_idx = raw_string.find("}", start_idx)

            result = ast.literal_eval(raw_string[start_idx : end_idx + 1])

            if "Summary" not in result:
                result["Summary"] = None
            
            result['Raw'] = raw_string

            # llm._sampler.close()
            # llm.close()

        except:
            result = {"Summary": None, "Raw": None}
                
        results.append(result)

        results_df = pd.DataFrame(results)
        results_df['mrn'] = mrn
        results_df['treatment_date'] = treatment_date
        results_df.to_csv(
            f"{save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_summary.csv"
        )

def launch(cfg):
    """Use submitit to launch jobs in the SLURM cluster

    References: 
    - https://www.unitary.ai/articles/intro-to-multi-node-machine-learning-2-using-slurm
    - https://github.com/facebookincubator/submitit/blob/main/docs/examples.md
    """
    # Initialize the executor, which is the submission interface
    executor = submitit.AutoExecutor(folder=f"log_files/{datetime.now().replace(microsecond=0)}")

    # Specify the Slurm parameters
    # TODO: put this in another config file
    executor.update_parameters(  
        # slurm_account="gliugroup_gpu",      
        slurm_partition="gpu",
        nodes=1, # Each job in the job array gets one node
        mem_gb=cfg['memory'], # Each job gets 4GB of memory
        timeout_min=cfg['n_hours'] * 60, # Limit the job running time to 2 days
        slurm_gpus_per_node=1, # Each node should use 1 GPU
        slurm_additional_parameters={
            "account": "gliugroup_gpu",
        }
    )

    # Split the data into n partitions
    n_partitions = cfg['n_partitions']
    data_dir = cfg['data_dir']
    df_name = cfg['file_name']
    os.makedirs(f'{data_dir}/data_partitions_summarize/', exist_ok=True)

    # read dataframe
    df = load_table(f'{data_dir}/{df_name}')

    cfg.pop('file_name')
    cfg.pop('data_dir')

    cfgs = []

    for partition_id, idxs in enumerate(np.array_split(df.index, n_partitions)):
        partition_path = f'{data_dir}/data_partitions_summarize/{partition_id}_{df_name}'
        if partition_path.endswith('.csv'):
            df.loc[idxs].reset_index(drop=True).to_csv(partition_path, index=False)
        elif partition_path.endswith(('.parquet','.parquet.gzip')):
            df.loc[idxs].reset_index(drop=True).to_parquet(partition_path, compression='gzip', index=False)

        cfgs.append(dict(data_dir=f'{data_dir}/data_partitions_summarize', file_name=f'{partition_id}_{df_name}', **cfg))

    # Submit your function and inputs as a job array
    jobs = executor.map_array(generate_note_summary, cfgs)

    # Monitor jobs to keep track of completed jobs
    submitit.helpers.monitor_jobs(jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("file_name", help="data file name", type=str)  # data file name
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("n_partitions", help="number of partitions", type = int) # number of partitions
    parser.add_argument("n_hours", help = "number of hours", type = int) # number of hours
    parser.add_argument("memory", help = "memory of each node", type = int) # memory of each node
    cfg = vars(parser.parse_args())
    launch(cfg)