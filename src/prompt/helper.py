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
from ml_common.constants import CANCER_CODE_MAP
import numpy as np
from openai import OpenAI

# Empty cuda cache
torch.cuda.empty_cache()

# note on few-shot prompting
# https://www.reddit.com/r/LocalLLaMA/comments/1at0zat/does_this_version_of_fewshot_learning_make_sense/

# this is for applying the chat template to the messages

from typing import List, Dict
from llama_cpp.llama_chat_format import (
    Jinja2ChatFormatter,
    ChatFormatterResponse,
    # ChatML (used for llama-3 BOS token)
    CHATML_BOS_TOKEN,
    # Existing templates
    CHATML_CHAT_TEMPLATE,
    CHATML_EOS_TOKEN,
    # Llama-3 template
    LLAMA3_INSTRUCT_CHAT_TEMPLATE,
)

# Llama-2 pieces (unchanged)
LLAMA2_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "{{ system_message }}\n"
    "<</SYS>>\n\n"
    "{{ user_message }} [/INST]"
)
LLAMA2_BOS_TOKEN = "<s>"
LLAMA2_EOS_TOKEN = "[/INST]"
LLAMA2_GEN_PROMPT = "\n\n<s>[INST] <assistant>"

def get_formatter(chat_format: str) -> Jinja2ChatFormatter:
    if chat_format == "chatml":
        tmpl, bos, eos = CHATML_CHAT_TEMPLATE, CHATML_BOS_TOKEN, CHATML_EOS_TOKEN
    elif chat_format == "llama-2":
        tmpl, bos, eos = LLAMA2_CHAT_TEMPLATE, LLAMA2_BOS_TOKEN, LLAMA2_EOS_TOKEN
    elif chat_format == "llama-3":
        # Use the built-in llama-3 instruct template (has its own add_generation_prompt block) :contentReference[oaicite:1]{index=1}
        tmpl = LLAMA3_INSTRUCT_CHAT_TEMPLATE
        bos = CHATML_BOS_TOKEN      # "<s>"
        eos = "<|eot_id|>"           # matches the "<|eot_id|>" marker in the template
    else:
        raise ValueError(f"Unsupported chat_format: {chat_format}")

    # We pass add_generation_prompt=True so Jinja will inject the assistant-start block where appropriate
    return Jinja2ChatFormatter(
        template=tmpl,
        bos_token=bos,
        eos_token=eos,
        add_generation_prompt=True,
    )

def apply_chat_template(
    messages: List[Dict[str, str]],
    chat_format: str,
    add_generation_prompt: bool = False,
) -> str:
    """
    Formats messages exactly like HF's apply_chat_template:
     - llama-2: manually stitches on the assistant prompt if requested
     - llama-3: uses the template's own add_generation_prompt logic
     - others: delegates entirely to the Jinja2ChatFormatter
    """
    formatter = get_formatter(chat_format)

    # llama-2 still needs manual stitching
    if chat_format == "llama-2":
        system_message = ""
        user_message = ""
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_message = msg["content"]

        prompt = formatter._environment.render(
            system_message=system_message,
            user_message=user_message,
            bos_token=formatter.bos_token,
            eos_token=formatter.eos_token,
            add_generation_prompt=False,
        )
        if add_generation_prompt:
            prompt += LLAMA2_GEN_PROMPT
        return prompt

    # For llama-3 and all others—let Jinja handle add_generation_prompt
    response: ChatFormatterResponse = formatter(
        messages=messages,
        add_generation_prompt=add_generation_prompt
    )
    return response.prompt

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

def alternate_rows(df, target_col):
    # Interleave the rows so that the target values alternate between 0 and 1
    # Split the dataframe based on 'target' values
    df_1 = df[df[target_col] == 1].reset_index(drop=True)
    df_0 = df[df[target_col] == 0].reset_index(drop=True)

    # Interleave the rows
    min_len = min(len(df_1), len(df_0))
    interleaved = pd.concat([df_1.iloc[:min_len], df_0.iloc[:min_len]], axis=1).stack().reset_index(drop=True)

    # Convert interleaved Series back to DataFrame
    interleaved_df = pd.DataFrame(interleaved.values.reshape(-1, df.shape[1]), columns=df.columns)

    # Append remaining rows properly using pd.concat()
    remaining = pd.concat([df_1.iloc[min_len:], df_0.iloc[min_len:]])

    # Return final DataFrame with reset index
    return pd.concat([interleaved_df, remaining], ignore_index=True)

def process_llm_output(raw_string, return_val):

    try:
        start_idx = raw_string.find("{")
        end_idx = raw_string.find("}", start_idx)

        result = ast.literal_eval(raw_string[start_idx : end_idx + 1])

        if return_val == 'proba':
            if "Probability" not in result:
                result["Probability"] = None
            if result["Probability"] is not None and result["Probability"] > 1:
                result["Probability"] = result["Probability"] / 100
        else:
            if "Prediction" not in result:
                result["Prediction"] = None
            if result["Prediction"] is not None and result["Prediction"] not in (0, 1):
                result["Prediction"] = 0 if abs(result["Prediction"] - 0) < abs(result["Prediction"] - 1) else 1

        if "Reason" not in result:
            result["Reason"] = None

        result['Raw'] = raw_string

    except:
        if return_val == 'proba':
            result = {"Reason": None, "Probability": None, "Raw": raw_string}
        else:
            result = {"Reason": None, "Prediction": None, "Raw": raw_string}

    return result

def clean_col_name(str_name):
    if str_name == 'num_prior_ED_visits_within_5_years':
        clean_name = 'number of prior ED visits within 5 years'
    elif str_name == 'days_since_prev_ED_visit':
        clean_name = 'days since previous ED visit'
    elif '%_ideal_dose_given' in str_name:
        clean_name = str_name.replace('_',' ').replace('given ', 'planned of ')
    elif str_name == 'female':
        clean_name = 'female (Y/N)'
    elif ('morphology' in str_name or 'cancer_site' in str_name) and 'other' not in str_name:
        # get last part of the string
        code = str_name.split('_')[-1]
        str_name = str_name.replace(code, CANCER_CODE_MAP[code])
        clean_name = str_name.replace('_', ' ')
        clean_name = clean_name.replace('cancer site', 'cancer site:').replace('morphology', 'morphology:')
    else:
        clean_name = str_name.replace('_',' ')
    
    clean_name = clean_name.replace('regimen', 'regimen: ')
    clean_name = clean_name.replace('intent', 'intent: ')

    return clean_name

def format_shap_values(df):
    max_name_len = max(df["var_names"].str.len().max(), len("Feature"))
    shap_col_width = 15
    corr_col_width = 15

    intro = (
        "Below are features with the highest mean absolute Shapley values "
        "from our tabular model predicting risk of this target. The correlation coefficient "
        "indicates the direction of the effect."
    )
    header = f"{'Feature':<{max_name_len}} {'Mean |SHAP|':>{shap_col_width}} {'Corr Coeff':>{corr_col_width}}"
    lines = [
        f"{row['var_names']:<{max_name_len}} {row['shap_values']:>{shap_col_width}.4f} {row['corr_coeff']:+{corr_col_width}.2f}"
        for _, row in df.iterrows()
    ]
    return "\n".join([intro, header] + lines)

def format_logistic_coefficients(df):
    max_name_len = max(df["var_names"].str.len().max(), len("Feature"))
    coef_col_width = 15

    intro = (
        "Below are features with the largest absolute logistic regression coefficients "
        "from our tabular model predicting risk of this target. A positive coefficient "
        "suggests that higher values of the feature are associated with increased risk, "
        "while a negative coefficient suggests decreased risk."
    )
    header = f"{'Feature':<{max_name_len}} {'LR Coefficient':>{coef_col_width}}"
    lines = [
        f"{row['var_names']:<{max_name_len}} {row['lr_values']:+{coef_col_width}.4f}"
        for _, row in df.iterrows()
    ]
    return "\n".join([intro, header] + lines)

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
    
    use_vllm = cfg["use_vllm"]
    if use_vllm == 1:
        base_url = cfg["base_url"]
        model_name = cfg["vllm_model_name"]

    add_shapley = cfg["add_shapley"]
    if add_shapley >= 1:
        shapley_path = cfg["shapley_path"]
    if add_shapley == 2:
        lr_path = cfg["log_reg_path"]

    # TO-DO:
    # edit prompt_engineering to include vllm input
    # and vllm host and model name
    # then need to update scripts on prompt engineering
    # client completions: how do you fix the seed and llm hyperparameters?
    # this is important if we plan to query the same note multiple times

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
    if n_few_shot > 0:
        few_shot_dir = cfg["few_shot_dir"]
    
    max_tokens = 250

    # check if memory is enough for flash attention
    avail_mem = torch.cuda.mem_get_info()[1]/100000
    use_flash_attn = False
    if avail_mem > 120000:
        use_flash_attn = True

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

    # get note data
    # load the clinical notes file
    clinical_notes_df = load_table(f"{data_dir}/{file_name}")
    prompt_file_list = [f'{prompt_file_dir}/prompt_list_{x}_numeric-proba{numeric_proba}.json' for x in target_list]

    # check whether the prompt contains 'Probability:' or 'Prediction:'
    first_prompt_file = prompt_file_list[0]
    with open(first_prompt_file, "r") as file:
        prompt_dict = json.load(file)
        system_instructions = prompt_dict["0"]
        # logger.info(f"sys instructions: {system_instructions}\n")
        if '"Probability":' in system_instructions:
            return_val = 'proba'
        elif '"Prediction":' in system_instructions:
            return_val = 'pred'
        else:
            raise NotImplementedError("To be added later.")

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
        n_ctx_length = 8192
        if 'Gemma' in LLM_name:
            chat_format = "gemma"
        elif 'Qwen' in LLM_name or 'QwQ' in LLM_name:
            chat_format = "chatml"
        elif 'Llama' in LLM_name:
            chat_format = "llama-3"
        else:
            chat_format = "llama-2"
        
        if return_val == "proba":
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"Reason": {"type": "string"}, 
                                "Probability": {"type": "float"}},
                    "required": ["Reason", "Probability"],
                },
            }
        else:
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"Reason": {"type": "string"}, 
                                "Prediction": {"type": "int"}},
                    "required": ["Reason", "Prediction"],
                },
            }

        # llm = Llama(model_path=LLM_path, n_gpu_layers=-1, main_gpu=0,
        #             chat_format=chat_format, seed=42, n_ctx=8192, flash_attn=True)

    # if n_few_shot > 0:
        # if condition on ED visit/ other targets
        # df_to_concat['preface'] =  ['Note {}:\n'.format(i + 1) for i in range(len(df_to_concat))]
        # df_to_concat['target_string'] = df_to_concat['preface'] + '{' + df_to_concat['note'] + '}' + '\n' + 'Event occurrence: ' + df_to_concat[target].astype(str)
    
    logger.info(f"{file_name}")

    # max_tokens = 2000
    # response_format = {}

    if use_vllm == 1:
        messages_list = []
        mrn_list = []
        treatment_date_list = []
        target_name_list = []
        target_val_list = []
        if n_few_shot > 0: n_examples_added_list = []

        # create extra body
        extra_body = {}
        if "top_k" in llm_params:
            extra_body["top_k"] = llm_params["top_k"]
            del llm_params["top_k"]
        if "min_p" in llm_params:
            extra_body["min_p"] = llm_params["min_p"]
            del llm_params["min_p"]
        
        llm_params['seed'] = 42
        
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
            
            if n_few_shot > 0:
                # load few shot file examples
                few_shot_fname = f"few_shot_{target_name}_nfewshot_{cfg['n_few_shot']}_{cfg['few_shot_date']}.csv"
                df_few_shot = pd.read_csv(f'{few_shot_dir}/{few_shot_fname}')

                # delete possibly duplicate example
                df_few_shot = df_few_shot.loc[df_few_shot['note'] != note].copy()

                # re-arrange so there is an alternating +/-
                df_few_shot = alternate_rows(df_few_shot, target_name)

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
                    chat_format=chat_format, seed=42, n_ctx=n_ctx_length, flash_attn=use_flash_attn)
               
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

            # if few-shot, add examples so as not to exceed limit
            if n_few_shot > 0:

                # assert that llama_cpp is equal to 1
                assert llama_cpp == 1, "Not implemented for hugging face models yet."

                # get tokenized length of note
                note_tokens = len(llm.tokenize(note.encode('utf-8')))

                # get tokenized length of system instructions
                system_instructions_tokens = len(llm.tokenize(system_instructions.encode('utf-8')))

                few_shot_phrase = (' Below are examples of clinical note summaries along with whether ' + 
                 'or not the adverse event occurred (1 = Yes, 0 = No). These examples are for reference, ' +
                 'but your prediction should be a probability rather than a binary classification.\n' + 
                 'Examples:\n')
                
                # get tokenized length of additional instructions
                few_shot_phrase_tokens = len(llm.tokenize(few_shot_phrase.encode('utf-8')))

                # get tokenized length of first note to be added
                first_note_tokens = len(llm.tokenize(df_few_shot.iloc[0]['note_summary'].encode('utf-8')))

                initial_tokens = (note_tokens + system_instructions_tokens + 
                                few_shot_phrase_tokens + first_note_tokens)

                n_examples_added = 0
                if initial_tokens < n_ctx_length:
                    
                    system_instructions = system_instructions + few_shot_phrase
                    
                    for idx_fewshot, row_fewshot in df_few_shot.iterrows():

                        if idx_fewshot + 1 > n_few_shot: break

                        note_summary = row_fewshot["note_summary"]
                        summary_target_val = row_fewshot[target_name]

                        # prepare new example
                        new_example = f'Clinical Note Summary: {note_summary}\n'
                        new_example += f'Outcome: {summary_target_val}\n\n'

                        example_tokens = len(llm.tokenize((system_instructions + new_example).encode('utf-8')))
                        if example_tokens + note_tokens < (n_ctx_length - max_tokens):
                            system_instructions += new_example
                        else:
                            idx_fewshot = idx_fewshot - 1
                            break

                    n_examples_added = idx_fewshot + 1

                # logger.info(f"system instructions: {system_instructions}\n")
            
            # add shapley values
            if add_shapley == 1:
                npz_file = np.load(shapley_path)
                var_names = npz_file['var_names']
                shapley_val_cols = [val for val in var_names if val not in ['visit_month_sin', 'visit_month_cos'] and 'is_missing' not in val]
                shap_values = npz_file['shap_values_test']
                ave_shap_values = np.mean(np.abs(shap_values), axis=0)
                df_shapley = pd.DataFrame({"var_names": var_names, "shap_values": ave_shap_values, "corr_coeff": npz_file['corr_coeff']})
                df_shapley = df_shapley.loc[df_shapley['var_names'].isin(shapley_val_cols)]
                df_shapley.sort_values(by="shap_values", ascending=False, inplace=True)
                df_shapley = df_shapley.reset_index(drop=True)
                df_shapley['var_names'] = df_shapley['var_names'].apply(clean_col_name)
                df_shapley = df_shapley[~np.isclose(df_shapley['shap_values'], 0)]
                shapley_string = format_shap_values(df_shapley)

                system_instructions = system_instructions + '\n' + shapley_string
                
                logger.info(f"system instructions: {system_instructions}\n")
            
            elif add_shapley == 2:
                shapley_npz_file = np.load(shapley_path)
                var_names = shapley_npz_file['var_names']
                shapley_val_cols = [val for val in var_names if val not in ['visit_month_sin', 'visit_month_cos'] and 'is_missing' not in val]
                lr_npz_file = np.load(lr_path)
                lr_values = lr_npz_file['coefficients']
                df_LR = pd.DataFrame({"var_names": var_names, "lr_values": lr_values})
                df_LR = df_LR.loc[df_LR['var_names'].isin(shapley_val_cols)]
                df_LR.sort_values(by="lr_values", key=np.abs, ascending=False, inplace=True)
                df_LR = df_LR.reset_index(drop=True)
                df_LR['var_names'] = df_LR['var_names'].apply(clean_col_name)
                df_LR = df_LR[~np.isclose(df_LR['lr_values'], 0)]
                LR_string = format_logistic_coefficients(df_LR)

                system_instructions = system_instructions + '\n' + LR_string

                logger.info(f"system instructions: {system_instructions}\n")

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

            if use_vllm == 1:
                messages_list.append(messages)
                mrn_list.append(mrn)
                treatment_date_list.append(treatment_date)
                target_name_list.append(target_name)
                target_val_list.append(row[target_name])
                # check loop for target name

                if n_few_shot > 0:
                    n_examples_added_list.append(n_examples_added)
                
                if llama_cpp == 1:
                    try:
                        llm._sampler.close()
                        llm.close()
                    except:
                        llm = None

            if use_vllm == 0:
                results = []
                for count in range(num_samples):
                    logger.info(count)

                    # generate llm response
                    if llama_cpp == 0:
                        try:
                            sequences = pipe(
                                messages, max_new_tokens=max_tokens, 
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
                            if len(response_format) == 0:
                                sequences = llm.create_chat_completion(messages=messages,  
                                                    max_tokens=max_tokens, 
                                                    **llm_params)
                            else:
                                sequences = llm.create_chat_completion(messages=messages, 
                                                    response_format=response_format, 
                                                    max_tokens=max_tokens, 
                                                    **llm_params)
                            raw_string = sequences['choices'][0]['message']['content']
                        except:
                            raw_string = None
                            
                    result = process_llm_output(raw_string, return_val)
                    result[target_name] = row[target_name]

                    results.append(result)

                results_df = pd.DataFrame(results)
                results_df['mrn'] = mrn
                results_df['treatment_date'] = treatment_date
                if n_few_shot > 0:
                    results_df['n_few_shot_added'] = n_examples_added

                results_df.to_csv(
                    f"{save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
                )

                if llama_cpp == 1:
                    try:
                        llm._sampler.close()
                        llm.close()
                    except:
                        llm = None

    # process vllm queries here
    if use_vllm == 1:

        logger.info(f"Running vllm\n")

        # process the messages list
        apply_chat_template_list = []
        for msg in messages_list:
            apply_chat_template_list.append(apply_chat_template(msg, chat_format, add_generation_prompt=True))

        # deal with repeated prompting here
        repeated_apply_chat_template_list = [item for item in apply_chat_template_list for _ in range(num_samples)]

        # CALL VLLM here
        vllm_output = []
        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=7200) 
        batch_response = client.completions.create(model=model_name, 
                                                    prompt=repeated_apply_chat_template_list, 
                                                    max_tokens=max_tokens, extra_body=extra_body, 
                                                    **llm_params) 
                                                    # request_timeout = 60 (1 minute)
                                                    # OPENAI_REQUEST_TIMEOUT=60
        vllm_output = [choice.text.strip() for choice in batch_response.choices]

        # loop over each mrn and save the dataframe
        for idx, mrn_val in enumerate(mrn_list):
            results = []
            vllm_output_mrn = vllm_output[:num_samples]
            vllm_output = vllm_output[num_samples:]
            for raw_string in vllm_output_mrn:
                result = process_llm_output(raw_string, return_val)
                result[target_name_list[idx]] = target_val_list[idx]
                results.append(result)
            results_df = pd.DataFrame(results)
            results_df['mrn'] = mrn_val
            results_df['treatment_date'] = treatment_date_list[idx]
            if n_few_shot > 0:
                results_df['n_few_shot_added'] = n_examples_added_list[idx]

            target_name_nospace = target_name_list[idx].replace("_", "-")
            results_df.to_csv(
                f"{save_dir}/mrn{mrn_val}_trtdate{treatment_date_list[idx][:10]}_{target_name_nospace}_{LLM_name}_prompt{prompt_num}.csv"
            )

# need to record mrn
# treatment_date
# n_examples_added