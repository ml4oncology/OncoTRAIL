import os
import ast
import json
import pandas as pd
from transformers import AutoTokenizer
import logging
import numpy as np
from datetime import datetime
from ml_common.util import load_table
from ml_common.constants import CANCER_CODE_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_col_name(str_name):
    if str_name == 'num_prior_ED_visits_within_5_years':
        clean_name = 'number of prior ED visits within 5 years'
    elif str_name == 'days_since_prev_ED_visit':
        clean_name = 'days since previous ED visit'
    elif '%_ideal_dose_given' in str_name:
        clean_name = str_name.replace('_', ' ').replace('given ', 'planned of ')
    elif str_name == 'female':
        clean_name = 'female (Y/N)'
    elif ('morphology' in str_name or 'cancer_site' in str_name) and 'other' not in str_name:
        code = str_name.split('_')[-1]
        str_name = str_name.replace(code, CANCER_CODE_MAP.get(code, code))
        clean_name = str_name.replace('_', ' ').replace('cancer site', 'cancer site:').replace('morphology', 'morphology:')
    else:
        clean_name = str_name.replace('_', ' ')
    return clean_name.replace('regimen', 'regimen: ').replace('intent', 'intent: ')

def load_shapley_df(shapley_path):
    npz_file = np.load(shapley_path)
    var_names = npz_file['var_names']
    shap_values = npz_file['shap_values_test']

    shapley_val_cols = [
        val for val in var_names
        if val not in ['visit_month_sin', 'visit_month_cos'] and 'is_missing' not in val
    ]

    ave_shap_values = np.mean(np.abs(shap_values), axis=0)
    df_shapley = pd.DataFrame({
        "var_names": var_names,
        "shap_values": ave_shap_values,
        "corr_coeff": npz_file['corr_coeff']
    })

    df_shapley = df_shapley[df_shapley['var_names'].isin(shapley_val_cols)]
    df_shapley.sort_values(by="shap_values", ascending=False, inplace=True)
    df_shapley = df_shapley.reset_index(drop=True)
    df_shapley['var_names'] = df_shapley['var_names'].apply(clean_col_name)
    df_shapley = df_shapley[~np.isclose(df_shapley['shap_values'], 0)]

    return df_shapley

def load_logistic_df(prediction_path, log_reg_path):
    prediction_npz_file = np.load(prediction_path)
    lr_npz_file = np.load(log_reg_path)

    var_names = prediction_npz_file['var_names']
    shapley_val_cols = [
        val for val in var_names
        if val not in ['visit_month_sin', 'visit_month_cos'] and 'is_missing' not in val
    ]

    lr_values = lr_npz_file['coefficients']
    df = pd.DataFrame({"var_names": var_names, "lr_values": lr_values})
    df = df[df['var_names'].isin(shapley_val_cols)]
    df.sort_values(by="lr_values", key=np.abs, ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    df['var_names'] = df['var_names'].apply(clean_col_name)
    df = df[~np.isclose(df['lr_values'], 0)]

    return df

class PromptingConfig:
    """Configuration class to encapsulate settings."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg["data_dir"]
        self.file_name = cfg["file_name"]
        self.LLM_path = cfg["LLM_path"]
        self.LLM_name = cfg["LLM_name"]
        self.save_dir = cfg["save_dir"]
        self.num_samples = cfg["num_samples"]
        self.target_list = cfg['target_names'].split(",")
        self.numeric_proba = cfg["numeric_proba"]
        self.prompt_file_dir = cfg["prompt_file_dir"]
        self.prompt_num = cfg["prompt_num"]
        self.use_vllm = cfg["use_vllm"] == 1
        self.tokenizer_path = cfg["tokenizer_path"]
        self.temperature = cfg.get("temperature", -1.0)
        self.min_p = cfg.get("min_p", -1.0)
        self.top_k = round(cfg.get("top_k", -1.0))
        self.top_p = cfg.get("top_p", -1.0)
        self.n_few_shot = cfg.get("n_few_shot", 0)
        self.few_shot_dir = cfg.get("few_shot_dir")
        self.few_shot_date = cfg.get("few_shot_date")
        self.add_tabularML_prediction = cfg.get("add_tabularML_prediction", 0)
        self.shapley_path = cfg.get("shapley_path")
        self.log_reg_path = cfg.get("log_reg_path")
        self.base_url = cfg.get("base_url")
        self.vllm_model_name = cfg.get("vllm_model_name")
        self.llm_params = self._extract_llm_params()
        self.extra_body = self._extract_extra_body() if self.use_vllm else {}

    def _extract_llm_params(self):
        llm_params = {}
        if self.temperature != -1.0:
            llm_params['temperature'] = self.temperature
        if self.min_p != -1.0:
            llm_params['min_p'] = self.min_p
        if self.top_k != -1.0:
            llm_params['top_k'] = self.top_k
        if self.top_p != -1.0:
            llm_params['top_p'] = self.top_p
        if self.use_vllm:
            llm_params['seed'] = 42
        return llm_params
    
    def _extract_extra_body(self):
        extra_body = {}
        for key in ["top_k", "min_p"]:
            if key in self.llm_params:
                extra_body[key] = self.llm_params[key]
                del self.llm_params[key]
        return extra_body
    
    def response_format(self, return_val):
        if return_val == "proba":
            return {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"Reason": {"type": "string"}, 
                                "Probability": {"type": "float"}},
                    "required": ["Reason", "Probability"],
                },
            }
        elif return_val == "pred":
            return {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"Reason": {"type": "string"}, 
                                "Prediction": {"type": "int"}},
                    "required": ["Reason", "Prediction"],
                },
            }
        else:
            raise ValueError(f"Unsupported return_val: {return_val}")


class PromptingUtils:
    @staticmethod
    def process_llm_output(raw_string, return_val):
        try:
            start_idx = raw_string.find("{")
            end_idx = raw_string.find("}", start_idx)
            result = ast.literal_eval(raw_string[start_idx: end_idx + 1])
            if return_val == 'proba':
                result["Probability"] = result.get("Probability", None)
                if result["Probability"] and result["Probability"] > 1:
                    result["Probability"] /= 100
            else:
                result["Prediction"] = result.get("Prediction", None)
                if result["Prediction"] is not None and result["Prediction"] not in (0, 1):
                    result["Prediction"] = 0 if abs(result["Prediction"] - 0) < abs(result["Prediction"] - 1) else 1
            result["Reason"] = result.get("Reason", None)
            result["Raw"] = raw_string
        except Exception:
            result = {
                "Reason": None,
                "Raw": raw_string,
            }
            if return_val == 'proba':
                result["Probability"] = None
            else:
                result["Prediction"] = None
        return result

    @staticmethod
    def alternate_rows(df, target_col):
        # Interleave the rows so that the target values alternate between 0 and 1
        df_1 = df[df[target_col] == 1].reset_index(drop=True)
        df_0 = df[df[target_col] == 0].reset_index(drop=True)
        # Interleave the rows
        min_len = min(len(df_1), len(df_0))
        interleaved = pd.concat([df_1.iloc[:min_len], df_0.iloc[:min_len]], axis=1).stack().reset_index(drop=True)
        # Convert interleaved Series back to DataFrame
        interleaved_df = pd.DataFrame(interleaved.values.reshape(-1, df.shape[1]), columns=df.columns)
        # Append remaining rows properly using pd.concat()
        remaining = pd.concat([df_1.iloc[min_len:], df_0.iloc[min_len:]])
        return pd.concat([interleaved_df, remaining], ignore_index=True)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def format_tabularMLprediction(prediction):
        tabular_ml_string = (
            f"Based on structured EHR data—covering demographics, cancer characteristics, "
            f"treatment details, lab tests, acute care history, and patient-reported symptoms—a "
            f"machine learning model predicted the risk of this adverse event as {prediction:.2f}. "
            f"You must take this prediction into account when forming your own assessment."
        )
        return tabular_ml_string

class BaseLLMRunner:
    """Base class with shared functionality for both local and vLLM runners."""
    
    def __init__(self, cfg: dict):
        self.config = PromptingConfig(cfg)
        self.utils = PromptingUtils()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.n_ctx_length = 8192
        self.max_tokens = 250
        self.prompt_file_list = []
        self.clinical_notes_df = None
        self.return_val = None
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _load_shapley_string(self):
        df_shapley = load_shapley_df(self.config.shapley_path)
        return self.utils.format_shap_values(df_shapley)

    def _load_logistic_string(self):
        df_logistic = load_logistic_df(self.config.shapley_path, self.config.log_reg_path)
        return self.utils.format_logistic_coefficients(df_logistic)
    
    def _load_tabularMLprediction_string(self, mrn):
        npz_file = np.load(self.config.shapley_path)
        mrn_pred_pairs = [
            ('mrn_train', 'train_pred'),
            ('mrn_valid', 'val_pred'),
            ('mrn_test', 'test_pred'),
            ('mrn_eval', 'eval_pred'),
        ]

        for mrn_key, pred_key in mrn_pred_pairs:
            mrns = npz_file[mrn_key]
            preds = npz_file[pred_key]

            # Check if target_mrn is in this array
            indices = np.where(mrns == mrn)[0]
            if len(indices) == 1:
                prediction = preds[indices.item()]
                return self.utils.format_tabularMLprediction(prediction)
            elif len(indices) > 1:
                raise ValueError(f"MRN {mrn} appears multiple times in {mrn_key}.")
            

    def load_prompt_files(self):
        self.clinical_notes_df = load_table(f"{self.config.data_dir}/{self.config.file_name}")
        self.prompt_file_list = [
            f"{self.config.prompt_file_dir}/prompt_list_{target}_numeric-proba{self.config.numeric_proba}.json"
            for target in self.config.target_list
        ]

        # check whether the prompt contains 'Probability:' or 'Prediction:'
        with open(self.prompt_file_list[0], "r") as file:
            prompt_dict = json.load(file)
            example_prompt = prompt_dict["0"]
            if '"Probability":' in example_prompt:
                self.return_val = 'proba'
            elif '"Prediction":' in example_prompt:
                self.return_val = 'pred'
            else:
                raise NotImplementedError("Prompt must contain 'Probability' or 'Prediction'.")

    def add_few_shot(self, system_instructions, note, target_name):
        fname = f"few_shot_{target_name}_nfewshot_{self.config.n_few_shot}_{self.config.few_shot_date}.csv"
        df_few = pd.read_csv(f"{self.config.few_shot_dir}/{fname}")
        # delete possibly duplicate example
        df_few = df_few[df_few["note"] != note].copy()
        df_few = self.utils.alternate_rows(df_few, target_name)

        few_shot_phrase = (
            " Below are examples of clinical note summaries along with whether "
            "or not the adverse event occurred (1 = Yes, 0 = No). These examples are for reference, "
            "but your prediction should be a probability rather than a binary classification.\n"
            "Examples:\n"
        )
        note_tokens = len(self.tokenizer.tokenize(note))
        system_tokens = len(self.tokenizer.tokenize(system_instructions))
        few_shot_phrase_tokens = len(self.tokenizer.tokenize(few_shot_phrase))
        first_example_tokens = len(self.tokenizer.tokenize(df_few.iloc[0]["note_summary"]))
        initial_tokens = note_tokens + system_tokens + few_shot_phrase_tokens + first_example_tokens
        n_added = 0

        # Only proceed if initial token budget allows
        if initial_tokens < self.n_ctx_length:
            system_instructions += few_shot_phrase

            for idx, row in df_few.iterrows():
                if idx + 1 > self.config.n_few_shot:
                    break

                example = (
                    f"Clinical Note Summary: {row['note_summary']}\n"
                    f"Outcome: {row[target_name]}\n\n"
                )
                example_tokens = len(self.tokenizer.tokenize(system_instructions + example))

                if example_tokens + note_tokens < (self.n_ctx_length - self.max_tokens):
                    system_instructions += example
                    n_added += 1
                else:
                    break

        return system_instructions, n_added

    def prepare_system_instructions(self, target_name, treatment_date, note, mrn):
        """Prepare system instructions for a given target and note."""
        with open(self.prompt_file_list[self.config.target_list.index(target_name)], "r") as f:
            prompt_dict = json.load(f)
            system_instructions = prompt_dict[f"{self.config.prompt_num}"]

        # convert treatment date in yyyy-mm-dd to MMM dd, yyyy format
        treatment_date_str = datetime.strptime(treatment_date[:10], "%Y-%m-%d").strftime("%b %d, %Y")
        # replace the treatment date in system_instructions
        system_instructions = system_instructions.replace("<TREATMENT DATE>", treatment_date_str)

        if self.config.n_few_shot > 0:
            system_instructions, n_examples_added = self.add_few_shot(system_instructions, note, target_name)
        else:
            n_examples_added = 0

        if self.config.add_tabularML_prediction == 1:
            system_instructions += "\n" + self._load_shapley_string()
        elif self.config.add_tabularML_prediction == 2:
            system_instructions += "\n" + self._load_logistic_string()
        elif self.config.add_tabularML_prediction == 3:
            system_instructions += "\n" + self._load_tabularMLprediction_string(mrn)

        return system_instructions, n_examples_added

    def run(self):
        logger.info(f"{self.config.save_dir}")
        self.load_prompt_files()
        logger.info(f"{self.config.file_name}")
        for _, row in self.clinical_notes_df.iterrows():
            self.process_patient_row(row)

    def process_patient_row(self, row):
        note, mrn, treatment_date = row["note"], row["mrn"], row["treatment_date"]
        logger.info(f"MRN: {mrn}")
        for idx, target_name in enumerate(self.config.target_list):
            if row[target_name] == -1:
                continue
            logger.info(f"Target: {target_name}")
            self.process_note_for_target(note, mrn, treatment_date, target_name, idx, row[target_name])

    def process_note_for_target(self, note, mrn, treatment_date, target_name, idx, target_val):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")