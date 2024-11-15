import os
import ast
import re

# # Set environment variable
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json
import pandas as pd
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)
from src.config import target_list

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


def processPromptEngineeringMultiTargets(
    data_dir,
    LLM_path,
    LLM_name,
    save_dir,
    file_name,
    json_path,
    prompt_num,
    num_samples,
    numeric_proba,
    minp=0
):
    
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
    data_path = f"{data_dir}/{file_name}"
    clinical_notes = pd.read_csv(data_path, index_col=False)
    prompt_file_list = [f'promptList_{x}_numeric-proba{numeric_proba}.json' for x in target_list]

    print(f"{file_name}")

    for _, row in clinical_notes.iterrows():

        note = row["note"]
        mrn = row["mrn"]

        print("MRN: {mrn}\n")

        for idx in range(len(target_list)):
            
            target_name = target_list[idx]
            json_file = prompt_file_list[idx]

            print("Target: {target}\n")
            
            torch.manual_seed(0)
            target_name_nospace = target_name.replace("_", "-")

            # check if file already exists, otherwise, skip!
            if os.path.isfile(
                f"{save_dir}/mrn{mrn}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            ):
                continue

            # open prompt file and extract prompt
            with open(f'{json_path}/{json_file}', "r") as file:
                prompt_dict = json.load(file)
            system_instructions = prompt_dict[f"{prompt_num}"]

            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": note},
            ]

            results = []
            for count in range(num_samples):
                print(count)

                if minp == 0:
                    sequences = pipe(
                        messages, max_new_tokens=250, do_sample=True, return_full_text=False
                    )
                else:
                    sequences = pipe(
                        messages, max_new_tokens=250, do_sample=True, return_full_text=False, min_p = 0.1, temperature = 1.5
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

                results.append(result)

                if count % 5 == 0:
                    torch.cuda.empty_cache()

            results_df = pd.DataFrame(results)
            results_df['mrn'] = mrn
            results_df.to_csv(
                f"{save_dir}/mrn{mrn}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("file_name", help="file name", type=str)  # file name
    parser.add_argument("json_path", help="path to json file", type=str)  # path to json file
    parser.add_argument("prompt_num", help="prompt number", type=int)  # prompt number
    parser.add_argument("num_samples", help="number of samples", type=int)  # number of samples
    parser.add_argument("numeric_proba", help="numerical probability", type=int)  # numeric probability?
    parser.add_argument("-minp", "--minp", help="use min p and good temperature?", type=int, default=0) # use minp and good temperature value?
    args = parser.parse_args()

    processPromptEngineeringMultiTargets(
        args.data_dir,
        args.LLM_path,
        args.LLM_name,
        args.save_dir,
        args.file_name,
        args.json_path,
        args.prompt_num,
        args.num_samples,
        args.numeric_proba,
        args.minp
    )
