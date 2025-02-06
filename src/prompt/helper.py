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
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)
from datetime import datetime
logger = logging.getLogger(__name__)
from ml_common.util import load_table
from llama_cpp import Llama

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

# TO-DO: merge this with llama_cpp
def prompt_llm(cfg: dict):

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
    llama_cpp = cfg["llama_cpp"]

    logger.info(f"{save_dir}")

    # llm parameters
    llm_params = {}
    if cfg['temperature'] != -1.0:
        llm_params['temperature'] = cfg["temperature"]
    if cfg['min_p'] != -1.0:
        llm_params['min_p'] = cfg["min_p"]
    if cfg['top_k'] != -1.0:
        llm_params['top_k'] = round(cfg["top_k"])
    if cfg['top_p'] != -1.0:
        llm_params['top_p'] = cfg["top_p"]

    n_few_shot = cfg["n_few_shot"]

    # dictionary map
    # event_map = {
        # "target_hemoglobin_grade2plus": "worsening anemia within 30 days",
        # "target_hemoglobin_grade3plus": "worsening anemia within 30 days",
        # "target_neutrophil_grade2plus": "worsening neutrophil count within 30 days",
        # "target_neutrophil_grade3plus": "worsening neutrophil count within 30 days",
        # "target_platelet_grade2plus": "worsening platelet count within 30 days",
        # "target_platelet_grade3plus": "worsening platelet count within 30 days",
        # "target_AKI_grade2plus": "acute kidney injury within 30 days",
        # "target_ALT_grade2plus": "increasing alanine aminotransferase within 30 days",
        # "target_ALT_grade3plus": "increasing alanine aminotransferase within 30 days",
        # "target_AST_grade2plus": "increasing aspartate aminotransferase within 30 days",
        # "target_AST_grade3plus": "increasing aspartate aminotransferase within 30 days",
        # "target_bilirubin_grade2plus": "increasing blood bilirubin within 30 days",
        # "target_bilirubin_grade3plus": "increasing blood bilirubin within 30 days",
        # "target_esas_pain_3pt_change": "worsening pain within 30 days",
        # "target_esas_tiredness_3pt_change": "worsening tiredness within 30 days",
        # "target_esas_nausea_3pt_change": "worsening nausea within 30 days",
        # "target_esas_depression_3pt_change": "worsening depression within 30 days",
        # "target_esas_anxiety_3pt_change": "worsening anxiety within 30 days",
        # "target_esas_drowsiness_3pt_change": "worsening drowsiness within 30 days",
        # "target_esas_appetite_3pt_change": "worsening appetite within 30 days",
        # "target_esas_well_being_3pt_change": "worsening well-being within 30 days",
        # "target_esas_shortness_of_breath_3pt_change": "worsening shortness of breath within 30 days",
        # "target_death_in_30d": "death within 30 days",
        # "target_death_in_365d": "death within 1 year",
        # "target_ED_visit": "visit the emergency department within the next 30 days"
        # }
    
    if llama_cpp == 1 and cfg['quant_level'] != 'NA':
        raise ValueError("Quantization level must be NA for llama cpp")

    elif llama_cpp == 0 and cfg['quant_level'] != 4:
        raise ValueError("Quantization level must be 4 for now")

    # create save folder
    os.makedirs(f'{save_dir}', exist_ok=True)

    # load model
    if llama_cpp == 0:
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
    else:
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
                "properties": {"Reason": {"type": "string"}, 
                               "Probability": {"type": "float"}},
                "required": ["Reason", "Probability"],
            },
        }

        # llm = Llama(model_path=LLM_path, n_gpu_layers=-1, main_gpu=0,
        #             chat_format=chat_format, seed=42, n_ctx=8192, flash_attn=True)

    # get note data
    # load the clinical notes file
    clinical_notes_df = load_table(f"{data_dir}/{file_name}")
    prompt_file_list = [f'{prompt_file_dir}/prompt_list_{x}_numeric-proba{numeric_proba}.json' for x in target_list]

    # if n_few_shot > 0:
        # if condition on ED visit/ other targets
        # df_to_concat['preface'] =  ['Note {}:\n'.format(i + 1) for i in range(len(df_to_concat))]
        # df_to_concat['target_string'] = df_to_concat['preface'] + '{' + df_to_concat['note'] + '}' + '\n' + 'Event occurrence: ' + df_to_concat[target].astype(str)
    
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
            
            target_name_nospace = target_name.replace("_", "-")

            # check if file already exists, otherwise, skip!
            if os.path.isfile(
                f"{save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            ):
                continue

            if llama_cpp == 0:
                torch.manual_seed(0)
            else:
                llm = Llama(model_path=LLM_path, n_gpu_layers=-1, main_gpu=0,
                    chat_format=chat_format, seed=42, n_ctx=8192, flash_attn=False)

            # open prompt file and extract prompt
            with open(json_file, "r") as file:
                prompt_dict = json.load(file)
            system_instructions = prompt_dict[f"{prompt_num}"]
            # convert treatment date in yyyy-mm-dd to MMM dd, yyyy format
            date_object = datetime.strptime(treatment_date[:10], "%Y-%m-%d")
            treatment_date_str = date_object.strftime("%b %d, %Y")
            # replace the treatment date in system_instructions
            system_instructions = system_instructions.replace(
                "<TREATMENT DATE>", treatment_date_str
            )

            # prepare prompt
            # if huggingface
            if "Llama" in LLM_name or llama_cpp == 1:
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

                # generate llm response
                if llama_cpp == 0:
                    try:
                        sequences = pipe(
                            messages, max_new_tokens=250, 
                            do_sample=True, 
                            return_full_text=False, 
                            **llm_params
                        )

                        seq = sequences[0]
                        raw_string = seq["generated_text"]
                    except:
                        raw_string = None

                else:
                    try:
                        sequences = llm.create_chat_completion(messages=messages, 
                                                response_format=response_format, 
                                                max_tokens=250, 
                                                **llm_params)
                        raw_string = sequences['choices'][0]['message']['content']
                    except:
                        raw_string = None
                        
                try:

                    start_idx = raw_string.find("{")
                    end_idx = raw_string.find("}", start_idx)

                    result = ast.literal_eval(raw_string[start_idx : end_idx + 1])

                    if "Probability" not in result:
                        result["Probability"] = None
                    if "Reason" not in result:
                        result["Reason"] = None

                    if result["Probability"] is not None and result["Probability"] > 1:
                        result["Probability"] = result["Probability"] / 100 
                    
                    result['Raw'] = raw_string

                except:
                    result = {"Reason": None, "Probability": None, "Raw": raw_string}

                result[target_name] = row[target_name]

                results.append(result)

            results_df = pd.DataFrame(results)
            results_df['mrn'] = mrn
            results_df['treatment_date'] = treatment_date
            results_df.to_csv(
                f"{save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            )

            if llama_cpp == 1:
                try:
                    llm._sampler.close()
                    llm.close()
                except:
                    llm = None