import numpy as np
import pandas as pd
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
import torch
from pathlib import Path
import submitit
from datetime import datetime
from ml_common.util import load_table
from llama_cpp import Llama
import logging
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)
logger = logging.getLogger(__name__)

# Empty cuda cache
torch.cuda.empty_cache()


def get_prompt(llm_name):
    target_prompt = "die within the next year"
    base_prompt = (
        "You are a highly experienced and extremely competent medical oncologist from the world-renowned Princess Margaret Cancer Centre in Toronto, Ontario. "
        + "Your task is to predict the probability that a patient undergoing systemic therapy for cancer will "
        + target_prompt
        + " based on the clinical note below. "
    )
    if "Instruct" in llm_name:
        base_prompt = [{"role": "system", "content": base_prompt}]
    return base_prompt


def get_quant_config():
    """Get quantization configurations for QLoRA - Quantized Low-Rank Adaptation

    Ref: https://github.com/artidoro/qlora
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    return quant_config


def generate_note_embedding(cfg: dict):

    data_dir = cfg['data_dir']
    file_name = cfg['file_name']
    LLM_path = cfg['LLM_path']
    LLM_name = cfg['LLM_name']
    save_dir = cfg['save_dir']
    prepend = cfg['prepend']
    llama_cpp = cfg['llama_cpp']

    notes_col_name = 'note'

    # load the clinical notes file
    clinical_notes = load_table(f'{data_dir}/{file_name}')

    embedding_dict = {}
    embeddings_list = []

    # obtain the unique notes and convert it to list
    notes_list = clinical_notes[notes_col_name].unique().tolist()

    if llama_cpp == 0:
        # define label maps
        id2label = {0: "Negative", 1: "Positive"}
        label2id = {"Negative": 0, "Positive": 1}

        # set up LLM -- generate classification model from model_checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(
            LLM_path,
            num_labels=2,
            quantization_config=get_quant_config(),
            id2label=id2label,
            label2id=label2id,
            device_map="auto",
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        # set up tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_path)
        if tokenizer.padding_side is None:
            tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "right"

        def decoder_text_to_embedding(text):
            tokenized_texts = tokenizer(text, truncation=True, return_tensors="pt")

            with torch.no_grad():
                transformer_outputs = model.model(
                    **tokenized_texts, output_hidden_states=True
                )

            hidden_states = transformer_outputs[0]

            input_ids = tokenized_texts["input_ids"]
            sequence_lengths = (
                torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]

            return hidden_states[:, sequence_lengths].cpu().numpy()[0]

        def longformer_text_to_embedding(text):
            tokenized_texts = tokenizer(text, truncation=True, return_tensors="pt")

            with torch.no_grad():
                transformer_outputs = model.longformer(
                    **tokenized_texts, output_hidden_states=True
                )

            hidden_states = transformer_outputs[0]

            return hidden_states[:, 0, :].cpu().numpy()[0]

        if LLM_name in ["Mistral", "BioMistral", "Llama3-8B", "Llama3-8B-Instruct"]:
            text_to_embedding = decoder_text_to_embedding
        elif LLM_name in ["ClinicalLongformer"]:
            text_to_embedding = longformer_text_to_embedding
        else:
            raise Exception("Not implemented yet.")

    elif llama_cpp == 1:
        if 'Gemma' in LLM_name:
            chat_format = "gemma"
        elif 'Qwen' in LLM_name or 'QwQ' in LLM_name:
            chat_format = "chatml"
        else:
            chat_format = "llama-2"

        llm = Llama(model_path=LLM_path, n_gpu_layers=-1, main_gpu=0,
                    chat_format=chat_format, seed=42, n_ctx=8192, flash_attn=False,
                    embedding=True)

    if prepend:
        # get prompt
        instr_prompt = get_prompt(LLM_name)
        if "Instruct" in LLM_name and llama_cpp == 0:
            instr_prompt = tokenizer.apply_chat_template(instr_prompt, tokenize=False)
        elif "Instruct" in LLM_name and llama_cpp == 1:
            raise Exception("Not implemented yet.")
        notes_list = [instr_prompt + "\n" + note for note in notes_list]

    for ctr, note in enumerate(notes_list):
        if llama_cpp == 0:
            embeddings_list.append(text_to_embedding(note))
        elif llama_cpp == 1:
            embedding = llm.create_embedding(note)
            embeddings_list.append(np.array(embedding['data'][0]['embedding'][-1]))
        logger.info(f"Processing note {ctr}")

        if ctr % 500 == 0:
            torch.cuda.empty_cache()

    embeddings = np.array(embeddings_list)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embedding_dict["embeddings"] = embeddings

    if '.parquet.gzip' in file_name:
        file_name = os.path.splitext(file_name)[0]
        file_name = os.path.splitext(file_name)[0]
    else:
        file_name = os.path.splitext(file_name)[0]

    np.savez(
        f"{save_dir}/{file_name}_{LLM_name}_embedding.npz", **embedding_dict
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
        constraint="gpu16g",
        slurm_additional_parameters={
            "account": "gliugroup_gpu",
        }
    )

    # Split the data into n partitions
    n_partitions = cfg['n_partitions']
    data_dir = cfg['data_dir']
    df_name = cfg['file_name']
    os.makedirs(f'{data_dir}/data_partitions_embedding/', exist_ok=True)

    # read dataframe
    df = load_table(f'{data_dir}/{df_name}')

    # keep only the notes column and the note_index column
    df = df[['note', 'note_index']].copy()

    # remove duplicates
    df = df.drop_duplicates(subset=['note'])

    cfg.pop('file_name')
    cfg.pop('data_dir')

    cfgs = []

    for partition_id, idxs in enumerate(np.array_split(df.index, n_partitions)):
        partition_path = f'{data_dir}/data_partitions_embedding/{partition_id}_{df_name}'
        if partition_path.endswith('.csv'):
            df.loc[idxs].reset_index(drop=True).to_csv(partition_path, index=False)
        elif partition_path.endswith(('.parquet','.parquet.gzip')):
            df.loc[idxs].reset_index(drop=True).to_parquet(partition_path, compression='gzip', index=False)

        cfgs.append(dict(data_dir=f'{data_dir}/data_partitions_embedding', file_name=f'{partition_id}_{df_name}', **cfg))

    # Submit your function and inputs as a job array
    jobs = executor.map_array(generate_note_embedding, cfgs)

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
    parser.add_argument("llama_cpp", help = "use llama_cpp", type = int) # use lama cpp
    parser.add_argument(
        "-p", "--prepend", help="prepend instructions", type=int, default=0
    )  # prepend note with instructions
    cfg = vars(parser.parse_args())
    launch(cfg)