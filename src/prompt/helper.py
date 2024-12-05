import os
import ast
import json
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
from ml_common.util import load_table

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


def prompt_huggingface(cfg: dict):

    data_dir = cfg["data_dir"]
    file_name = cfg["file_name"]
    LLM_path = cfg["LLM_path"]  
    LLM_name = cfg["LLM_name"]
    save_dir = cfg["save_dir"]
    num_samples = cfg["num_samples"]
    target_list = cfg['target_names'].split(",")
    numeric_proba = cfg["numeric_proba"]
    prompt_file_dir = cfg["prompt_file_dir"]
    prompt_num = cfg["prompt_num"]
    temperature = cfg["temperature"]
    min_p = cfg["min_p"]
    top_k = cfg["top_k"]
    top_p = cfg["top_p"]
    
    if cfg['quant_level'] != 4:
        raise ValueError("Quantization level must be 4 for now")

    # create save folder
    os.makedirs(f'{save_dir}', exist_ok=True)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        LLM_path, device_map="auto", quantization_config=get_quant_config()
    )
    # load_in_8bit=True,
    # quantization_config=get_quant_config()

    tokenizer = AutoTokenizer.from_pretrained(LLM_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # get note data
    # load the clinical notes file
    clinical_notes_df = load_table(f"{data_dir}/{file_name}")
    prompt_file_list = [f'{prompt_file_dir}/prompt_list_{x}_numeric-proba{numeric_proba}.json' for x in target_list]

    logger.info(f"{file_name}")

    for _, row in clinical_notes_df.iterrows():

        note = row["note"]
        mrn = row["mrn"]
        treatment_date = row["treatment_date"]

        logger.info(f"MRN: {mrn}\n")

        for idx in range(len(target_list)):
            
            target_name = target_list[idx]
            json_file = prompt_file_list[idx]

            if row[target_name] == -1:
                continue

            logger.info(f"Target: {target_name}\n")
            
            torch.manual_seed(0)
            target_name_nospace = target_name.replace("_", "-")

            # check if file already exists, otherwise, skip!
            if os.path.isfile(
                f"{save_dir}/mrn{mrn}_trtdate{treatment_date}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            ):
                continue

            # open prompt file and extract prompt
            with open(json_file, "r") as file:
                prompt_dict = json.load(file)
            system_instructions = prompt_dict[f"{prompt_num}"]
            # convert treatment date in yyyy-mm-dd to MMM dd, yyyy format
            date_object = datetime.strptime(treatment_date, "%Y-%m-%d")
            treatment_date_str = date_object.strftime("%b %d, %Y")
            # replace the treatment date in system_instructions
            system_instructions = system_instructions.replace(
                "<TREATMENT DATE>", treatment_date_str
            )

            if "Llama" in LLM_name:
                messages = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": note},
                ]
            else:
                messages = [
                    {"role": "user", "content": f"{system_instructions}\n\n{note}"},
                ]

            results = []
            for count in range(num_samples):
                logger.info(count)

                sequences = pipe(
                    messages, max_new_tokens=250, 
                    do_sample=True, 
                    return_full_text=False, 
                    min_p = min_p, 
                    temperature = temperature,
                    top_k = top_k,
                    top_p = top_p
                )

                seq = sequences[0]

                try:
                    temp_string = seq["generated_text"]

                    start_idx = temp_string.find("{")
                    end_idx = temp_string.find("}", start_idx)

                    result = ast.literal_eval(temp_string[start_idx : end_idx + 1])

                    if "Probability" not in result:
                        result["Probability"] = None
                    if "Reason" not in result:
                        result["Reason"] = None
                    
                    result['Raw'] = temp_string

                except:
                    result = {"Reason": None, "Probability": None, "Raw": seq["generated_text"]}

                # result[target_name] = row[target_name]

                results.append(result)

                # if count % 5 == 0:
                #     torch.cuda.empty_cache()

            results_df = pd.DataFrame(results)
            results_df['mrn'] = mrn
            results_df['treatment_date'] = treatment_date
            results_df.to_csv(
                f"{save_dir}/mrn{mrn}_trtdate{treatment_date}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            )