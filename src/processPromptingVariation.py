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

def gen_prompt(target_string):

    additional_info = ''
    if target_string == 'target_ED_visit':
        target_prompt = "visit the emergency department within the next 30 days"
    elif target_string == 'target_death_in_365d':
        target_prompt = "die within the next year"
    elif target_string == 'target_death_in_30d':
        target_prompt = "die within the next 30 days"
    elif 'esas' in target_string:
        if 'well_being' in target_string:
            target_string = target_string.replace('well_being','well-being')
        
        if 'shortness_of_breath' in target_string:
            target_string = target_string.replace('shortness_of_breath','shortness of breath')

        # extract the target
        esas_target = target_string.split('_')[2]
        esas_change_value = target_string.split('_')[3][0]

        target_prompt = f"experience a {esas_change_value} point change in the ESAS score for {esas_target}"

        # extract the point change

        additional_info = ("The ESAS score refers to the Edmonton Symptom Assessment System. " +
                            "It's a clinical tool used to assess the severity of common symptoms " + 
                            "experienced by patients with cancer and other advanced illnesses. Patients rate the " +
                            "severity of each symptom on a scale from 0 to 10, with 0 indicating no symptom " +
                            "and 10 indicating the worst possible severity. This assessment helps healthcare " +
                            "providers manage symptoms and improve quality of life for patients.")
        
    else:
        raise Exception("Not implemented yet.")        

    base_prompt =  ("You are a highly experienced and extremely competent medical oncologist from the world-renowned Princess Margaret Cancer Centre in Toronto, Ontario. " + 
                    "Your task is to predict the probability that a patient undergoing systemic therapy for cancer will " + target_prompt + " based on the clinical note below. "
                    + additional_info + " First, concisely explain your reasoning as a response to 'Reason:'. Subsequently, provide a numerical value only as a response to 'Probability:'.")
                    
    return base_prompt

def extract_probability(prompt_response):
    # find Probability:
    prob_idx = prompt_response.index("Probability:") + len("Probability:")
     
    # convert string to float  
    prob_value = float(prompt_response[prob_idx:].replace('\n','').strip())

    return prob_value

def process_prompting(data_path, LLM_path, LLM_name, save_dir, target_name):

    torch.manual_seed(0)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        LLM_path,
        device_map="auto",
        quantization_config=get_quant_config()
    )
    # load_in_8bit=True,
    # quantization_config=get_quant_config()

    tokenizer = AutoTokenizer.from_pretrained(LLM_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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

    system_instructions = gen_prompt(target_name)

    llm_response = []
    probability_val = []

    for count, note in enumerate(notes_list):
        
        print(count, len(note))
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note }
        ]

        sequences = pipe(
                    messages,
                    max_new_tokens=250,
                    do_sample=True,
                    return_full_text=False
                )
        seq = sequences[0]
        llm_response.append(seq['generated_text'])
        probability_val.append(extract_probability(seq['generated_text']))

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