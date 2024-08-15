import os
# # Set environment variable
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import pandas as pd
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import torch
from pathlib import Path

# Empty cuda cache
torch.cuda.empty_cache()

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

def gen_prompt(target_name):

    # base_prompt =  ("You are a highly experienced and extremely competent medical oncologist from the world-renowned Princess Margaret Cancer Centre in Toronto, Ontario. " + 
    #                 "Your task is to predict the probability that a patient is a male based on the de-identified clinical note below. " +
    #                  "First, concisely explain your reasoning as a response to 'Reason:'. Subsequently, provide a numerical value only as a response to 'Probability:'.")

    if target_name == 'target_male':
        str_to_add = 'male'
    elif target_name == 'target_female':
        str_to_add = 'female'

    base_prompt =  (f"Given the de-identified clinical note of a patient below, your task is to predict the probability that the patient is a {str_to_add}. " +
                     "First, concisely explain your reasoning as a response to 'Reason:'. Subsequently, provide a numerical value only as a response to 'Probability:'.")

    return base_prompt

def extract_probability(prompt_response):
    # find Probability:
    prob_idx = prompt_response.index("Probability:") + len("Probability:")
     
    # convert string to float  
    prob_value = float(prompt_response[prob_idx:].replace('\n','').strip())

    if prob_value > 1:
        prob_value = prob_value/100
    
    return prob_value

def process_prompting(data_path, LLM_path, LLM_name, save_dir, target_name):

    print(data_path)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        LLM_path,
        device_map="auto",
        quantization_config=get_quant_config()
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        temperature=1
    )

    # extract file name
    file_name = os.path.basename(data_path)
    file_name = Path(file_name).stem

    # load the clinical notes file
    clinical_notes = pd.read_csv(data_path, index_col=False)

    # convert note data to list
    notes_list = clinical_notes["note"].tolist()

    # append "Probability:" and "Reason:" to each note
    notes_list = [note + '\nReason:\nProbability:' for note in notes_list]

    # generate the prompt
    system_instructions = gen_prompt(target_name)

    llm_response = []
    probability_val = []

    for count, note in enumerate(notes_list):
        torch.manual_seed(0)
        print(count, len(note))
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note }
        ]

        sequences = pipe(
                    messages,
                    max_new_tokens=250,
                    do_sample=False,
                    return_full_text=False,
                    top_k=1
                )
        seq = sequences[0]

        try:
            probability_val.append(extract_probability(seq['generated_text']))
            llm_response.append(seq['generated_text'])
        
        except:
            probability_val.append(-1)
            llm_response.append(seq['generated_text'])

        if count % 5 == 0:
            torch.cuda.empty_cache()
    
    clinical_notes[f'llm_response_{LLM_name}_{target_name}'] = llm_response
    clinical_notes[f'probability_{LLM_name}_{target_name}'] = probability_val

    clinical_notes.to_csv(f'{save_dir}/{file_name}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data file path", type=str)  # data file path
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("target_name", help="target name", type=str)  # target name
    args = parser.parse_args()

    process_prompting(args.data_path, args.LLM_path, args.LLM_name, args.save_dir, args.target_name)