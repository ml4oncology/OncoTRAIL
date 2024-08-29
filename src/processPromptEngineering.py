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

# Empty cuda cache
torch.cuda.empty_cache()


def fix_quotes(input_string):
    # Pattern to match single-quoted strings where single quotes are not properly enclosed
    pattern = re.compile(r"(:\s*)'([^']*'[^']*)'")

    # Replace single quotes around values with double quotes
    transformed_string = pattern.sub(r'\1"\2"', input_string)

    return transformed_string


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


def processPromptEngineering(
    data_dir,
    LLM_path,
    LLM_name,
    save_dir,
    target_name,
    file_name,
    patient_num,
    json_file,
    prompt_num,
    num_samples,
):
    torch.manual_seed(0)
    target_name_nospace = target_name.replace("_", "-")
    file_name_notemplate = file_name.replace("_template", "")
    print(
        f"{file_name_notemplate}_{target_name_nospace}_{LLM_name}_patient{patient_num}_prompt{prompt_num}"
    )

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

    # load the clinical notes file
    data_path = f"{data_dir}/{file_name}_patient{patient_num}.csv"
    clinical_notes = pd.read_csv(data_path, index_col=False)

    # get note data
    note = clinical_notes["note"].tolist()[0]

    # open prompt file and extract prompt
    with open(json_file, "r") as file:
        prompt_dict = json.load(file)
    system_instructions = prompt_dict[f"{prompt_num}"]

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": note},
    ]

    results = []
    for count in range(num_samples):
        print(count)

        sequences = pipe(
            messages, max_new_tokens=250, do_sample=True, return_full_text=False
        )

        seq = sequences[0]
        # print(seq['generated_text'])
        # result = json.loads(seq['generated_text'])
        # result = ast.literal_eval(fix_quotes(seq['generated_text']))
        try:
            temp_string = seq["generated_text"]

            start_idx = temp_string.find("{")
            end_idx = temp_string.find("}", start_idx)

            result = ast.literal_eval(temp_string[start_idx : end_idx + 1])

            if "Probability" not in result:
                result["Probability"] = None
            if "Reason" not in result:
                result["Reason"] = None
        except:
            result = {"Reason": None, "Probability": None}

        results.append(result)

        if count % 5 == 0:
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"{save_dir}/{file_name_notemplate}_{target_name_nospace}_{LLM_name}_patient{patient_num}_prompt{prompt_num}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("target_name", help="target name", type=str)  # target name
    parser.add_argument("file_name", help="file name", type=str)  # file name
    parser.add_argument(
        "patient_num", help="patient number", type=int
    )  # patient number
    parser.add_argument("json_file", help="json file path", type=str)  # json file path
    parser.add_argument("prompt_num", help="prompt number", type=int)  # prompt number
    parser.add_argument("num_samples", help="number", type=int)  # prompt number
    args = parser.parse_args()

    processPromptEngineering(
        args.data_dir,
        args.LLM_path,
        args.LLM_name,
        args.save_dir,
        args.target_name,
        args.file_name,
        args.patient_num,
        args.json_file,
        args.prompt_num,
        args.num_samples,
    )
