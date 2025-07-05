import argparse
import math
# needed as this function doesn't like it when the lm_head has its size changed
from unsloth import tokenizer_utils
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import logging
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)
from datetime import datetime
logger = logging.getLogger(__name__)
major_version, minor_version = torch.cuda.get_device_capability()
logger.info(f"Major: {major_version}, Minor: {minor_version}")
import datasets
from trl import SFTTrainer
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, TrainerCallback
from typing import Tuple
import warnings
from typing import Any, Dict, List, Union
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from sklearn.metrics import log_loss
from peft import LoftQConfig
import re
import gc
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
os.environ["FLASH_ATTENTION_FORCE"] = "True"
from threading import Thread
import pynvml
import psutil
import time

CTCAE_constants = {
    'hemoglobin': {
        'grade2plus': 100,
        'grade3plus': 80
    },
    'neutrophil': {
        'grade2plus': 1.5,
        'grade3plus': 1.0
    },
    'platelet': {
        'grade2plus': 75,
        'grade3plus': 50
    },
    'AKI': {
        'grade2plus': 1.5,
        'grade3plus': 3.0,
        'ULN': 353.68
    },
    'ALT': {
        'grade2plus': 3.0,
        'grade3plus': 5.0,
        'ULN': 40.0
    },
    'AST': {
        'grade2plus': 3.0,
        'grade3plus': 5.0,
        'ULN': 34.0
    },
    'bilirubin': {
        'grade2plus': 1.5,
        'grade3plus': 3.0,
        'ULN': 22.0
    }
}

def monitor_gpu_memory(interval_sec=5):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes you're using GPU 0
    while True:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / 1024**2
        total_mb = mem_info.total / 1024**2
        logger.info(f"[GPU Monitor] Used: {used_mb:.2f} MB | Total: {total_mb:.2f} MB")
        time.sleep(interval_sec)

def monitor_cpu_memory(interval_sec=5):
    pid = os.getpid()
    process = psutil.Process(pid)
    while True:
        mem_mb = process.memory_info().rss / 1024**2
        logger.info(f"[CPU Monitor] RAM Used: {mem_mb:.2f} MB")
        time.sleep(interval_sec)

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs.copy()
        logs["step"] = state.global_step
        self.logs.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.csv_path, index=False)

def formatting_prompts_func(df_, prompt_template, string_to_add):
    texts = []
    targets = df_["label"].tolist()
    for date_, text_, lbl in zip(df_["treatment_date_str"], df_["note"], targets):
        # Insert the prompt: we fill in the text_ and the target name and the correct label (0/1)
        full = prompt_template.format(date_, text_, string_to_add, lbl)
        texts.append(full)
    return texts

def generate_target_description(target_string, simplify):

    time_period = "within the next 30 days"
    additional_info = ""
    if target_string == "target_ED_visit":
        target_prompt = f"visit the emergency department {time_period}"
    elif target_string == "target_death_in_365d":
        target_prompt = "die within the next year"
    elif target_string == "target_death_in_30d":
        target_prompt = "die within the next 30 days"
    elif "esas" in target_string:
        if "well_being" in target_string:
            target_string = target_string.replace("well_being", "well-being")

        if "shortness_of_breath" in target_string:
            target_string = target_string.replace(
                "shortness_of_breath", "shortness of breath"
            )

        # extract the target
        esas_target = target_string.split("_")[2]
        esas_change_value = target_string.split("_")[3][0]

        if simplify == 0:
            target_prompt = (f"experience a {esas_change_value} point change in the ESAS score for {esas_target}" +
                             ' ' + time_period)

            # extract the point change

            additional_info = (
                "The ESAS score refers to the Edmonton Symptom Assessment System. "
                + "It's a clinical tool used to assess the severity of common symptoms "
                + "experienced by patients with cancer and other advanced illnesses. Patients rate the "
                + "severity of each symptom on a scale from 0 to 10, with 0 indicating no symptom "
                + "and 10 indicating the worst possible severity. This assessment helps healthcare "
                + "providers manage symptoms and improve quality of life for patients. "
            )
        else:
            target_prompt = f"experience worsening {esas_target} {time_period}"

    elif re.search(r'grade\d+plus', target_string) is not None:

        match = re.search(r'target_(.*?)_grade([1-5])plus', target_string)
        # extract the grade
        grade = match.group(2)
        # extract the quantity
        quantity = match.group(1)

        CTCAE_meaning = "defined by the CTCAE (Common Terminology Criteria for Adverse Events)"

        if "hemoglobin" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above anemia {time_period}, {CTCAE_meaning} "
                    f" as a hemoglobin level under {CTCAE_constants[quantity][f'grade{grade}plus']} g/L"
                )
            else:
                target_prompt = f"experience worsening anemia {time_period}"

        elif "neutrophil" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above neutrophil count decrease {time_period}, {CTCAE_meaning} "
                    f"as a neutrophil count under {CTCAE_constants[quantity][f'grade{grade}plus']} x 10e9/L"
                )
            else:
                target_prompt = f"experience worsening neutrophil count {time_period}"

        elif "platelet" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above platelet count decrease {time_period}, {CTCAE_meaning} "
                    f"as a platelet count under {CTCAE_constants[quantity][f'grade{grade}plus']} x 10e9/L"
                )
            else:
                target_prompt = f"experience worsening platelet count {time_period}"

        elif "AKI" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above creatinine increase {time_period}, {CTCAE_meaning} "
                    f"as creatinine increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"baseline or {CTCAE_constants[quantity][f'grade{grade}plus']} times above " 
                    f" the upper limit of normal ({CTCAE_constants[quantity]['ULN']} umol/L)"
                )
            else:
                target_prompt = f"experience acute kidney injury {time_period}"

        elif "ALT" in target_string or "AST" in target_string:
            if 'ALT' in target_string:
                quantity_full = 'alanine aminotransferase'

            elif 'AST' in target_string:
                quantity_full = 'aspartate aminotransferase'

            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above {quantity_full} increase {time_period}, {CTCAE_meaning} "
                    f"as {quantity_full} increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"the upper limit of normal ({CTCAE_constants[quantity]['ULN']} U/L) or baseline if the baseline was abnormal"
                )
            else:
                target_prompt = f"experience increasing {quantity_full} level {time_period}"

        elif "bilirubin" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above blood bilirubin increase {time_period}, {CTCAE_meaning} "
                    f"as blood bilirubin increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"the upper limit of normal ({CTCAE_constants[quantity]['ULN']} umol/L) or baseline if the baseline was abnormal"
                )
            else:
                target_prompt = f"experience increasing blood bilirubin level {time_period}"

    target_prompt = target_prompt + '? ' + additional_info

    return target_prompt


# ====================================================================================
# Build a custom DataCollator that only trains on the final “0/1” token  
# ====================================================================================
class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        reverse_map: Dict[int, int],     # ⇨ new required argument
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.reverse_map = reverse_map  # store locally

    def torch_call(
        self,
        examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            # find last non-ignored token
            last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
            # mask everything except that last token
            batch["labels"][i, :last_token_idx] = self.ignore_index

            # re-map old_id → new lm_head index via self.reverse_map
            old_id = batch["labels"][i, last_token_idx].item()
            batch["labels"][i, last_token_idx] = self.reverse_map[old_id]

        return batch

def run_inference(
    inference_prompt_template,
    model,
    tokenizer,
    df,
    string_to_add,
    number_token_ids,
    device,
    max_seq_length,
    batch_size
):
    """
    Run inference on a DataFrame of clinical notes and return predictions and probabilities.

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        df: DataFrame with columns ['note', 'treatment_date_str', 'label']
        string_to_add: Adverse event string to complete the inference prompt
        number_token_ids: List or array with the token IDs corresponding to "0" and "1"
        device: Device to run the model on
        max_seq_length: Max token length (default 2048)
        batch_size: Batch size for inference (default 4)
        description: TQDM progress bar description

    Returns:
        A new DataFrame with ['pred', 'prob'] columns (aligned with the input DataFrame)
    """

    df = df.copy()
    df["prompt_text"] = df.apply(
        lambda row: inference_prompt_template.format(
            row["treatment_date_str"], row["note"], string_to_add
        ),
        axis=1,
    )

    df["token_length"] = df["prompt_text"].apply(
        lambda txt: len(tokenizer.encode(txt, add_special_tokens=False))
    )

    df_sorted = df.sort_values(by="token_length").reset_index(drop=True)

    results = []

    logger.info(
        f"[Evaluating] Starting inference: Allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    # with torch.inference_mode():
    with torch.inference_mode():
        for i in tqdm(range(0, len(df_sorted), batch_size), desc="Evaluating"):
            batch = df_sorted.iloc[i : i + batch_size]
            prompts = batch["prompt_text"].tolist()

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )# .to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            # last_idxs = inputs.attention_mask.sum(1) - 1
            last_idxs = inputs["attention_mask"].sum(1) - 1
            last_logits = logits[torch.arange(len(batch)), last_idxs, :]

            probs_all = F.softmax(last_logits, dim=-1)
            probs = probs_all[:, number_token_ids]  # Shape: (batch_size, 2)

            preds = torch.argmax(probs, dim=-1).cpu().numpy()

            for j in range(len(batch)):
                pred_label = int(preds[j])
                pred_prob = float(probs[j, 1].cpu().numpy())

                results.append({
                    "pred": pred_label,
                    "prob": pred_prob,
                })

            del inputs, logits, last_logits, probs_all, probs, preds
            torch.cuda.empty_cache()
            gc.collect()

    df_results = pd.DataFrame(results)
    df_sorted = df_sorted.reset_index(drop=True)
    df_merged = df_sorted.drop(columns=["prompt_text", "token_length"]).copy()
    df_merged["pred"] = df_results["pred"]
    df_merged["prob"] = df_results["prob"]

    # compute AUC
    auc_val = roc_auc_score(df_merged["label"], df_merged["prob"])
    # compute cross-entropy loss
    cross_entropy_loss = log_loss(df_merged['label'], df_merged['prob'])

    return df_merged, auc_val, cross_entropy_loss

class LLMFineTuner:
    def __init__(self, LLM_path, max_seq_length=8192, num_classes=2):
        self.LLM_path = LLM_path
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.load_in_4bit = '4bit' in LLM_path
        self.dtype = torch.bfloat16
        self.old_size = None
        self.model = None
        self.tokenizer = None
        self.number_token_ids = None
        self.reverse_map = None
        self.collator = None

    def load_model(self):
        # Load model & tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.LLM_path,
            load_in_4bit=self.load_in_4bit,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
        )

        # Find token IDs for "0" and "1"
        self.number_token_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(self.num_classes)
        ]
        self.reverse_map = {value: idx for idx, value in enumerate(self.number_token_ids)}

        # Trim lm_head to only keep weights for "0" and "1"
        par = torch.nn.Parameter(self.model.lm_head.weight[self.number_token_ids, :])
        self.old_size = self.model.lm_head.weight.shape[0]
        self.model.lm_head.weight = par

        # Wrap with PEFT
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
            modules_to_save=["lm_head"]
        )

        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def create_collator(self):
        self.collator =  DataCollatorForLastTokenLM(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100,
            reverse_map=self.reverse_map,  # ⇨ pass it here
        )
        
    def train_model(self, train_dataset, eval_dataset, training_args, csv_logger):

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
            data_collator=self.collator,
            dataset_text_field="text",
            callbacks=[csv_logger]
        )

        assert self.model.config.num_labels == 2, "Model should have exactly 2 output labels"

        # (Optional) Print GPU memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_gpu = round(gpu_stats.total_memory / 1024**3, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_gpu} GB. Start reserved = {start_gpu} GB.")

        Thread(target=monitor_gpu_memory, daemon=True).start()
        Thread(target=monitor_cpu_memory, daemon=True).start()

        trainer_stats = trainer.train()

        # (Optional) Print final memory/time stats
        used_gpu = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_for_lora = round(used_gpu - start_gpu, 3)
        percent_used = round(used_gpu / max_gpu * 100, 3)
        percent_lora = round(used_for_lora / max_gpu * 100, 3)
        logger.info(f"Training runtime: {trainer_stats.metrics['train_runtime']} seconds")
        logger.info(f"Peak GPU memory reserved: {used_gpu} GB ({percent_used}%).  LoRA‐only: {percent_lora}%.")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def fix_lm_head_for_inference(self):
        # Reconstruct full lm_head so it can produce logits for "0" and "1"
        trimmed_lm_head = self.model.lm_head.weight.data.clone()
        trimmed_lm_head_bias = (
            self.model.lm_head.bias.data.clone()
            if hasattr(self.model.lm_head, "bias") and self.model.lm_head.bias is not None
            else torch.zeros(len(self.number_token_ids), device=trimmed_lm_head.device)
        )

        hidden_dim = trimmed_lm_head.shape[1]
        new_lm_head = torch.full((self.old_size, hidden_dim), 0.0, dtype=trimmed_lm_head.dtype, device=trimmed_lm_head.device)
        new_lm_head_bias = torch.full((self.old_size,), -1000.0, dtype=trimmed_lm_head_bias.dtype, device=trimmed_lm_head_bias.device)

        for new_idx, orig_token_id in enumerate(self.number_token_ids):
            new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
            new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]

        with torch.no_grad():
            new_lm_head_module = torch.nn.Linear(hidden_dim, self.old_size, bias=True, device=self.model.device)
            new_lm_head_module.weight.data.copy_(new_lm_head)
            new_lm_head_module.bias.data.copy_(new_lm_head_bias)
            self.model.lm_head.modules_to_save["default"] = new_lm_head_module
        
        logger.info(f"Remade lm_head: shape = {self.model.lm_head.weight.shape}. Allowed tokens: {self.number_token_ids}")

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

# need to figure out the train/validation/test split
def main(
        LLM_path,
        notes_path,
        target_name,
        development_set_date,
        results_dir,
        learning_rate,
        n_epochs,
        batch_size_train,
        batch_size_test,
        gradient_accumulation_steps
    ):
    """
    
    """

    set_seed(3407)

    # extract LLM name from LLMpath
    LLM_name = LLM_path.split("/")[-1]

    # add subdirectory to results_dir based on target_name
    results_dir = os.path.join(results_dir, target_name)
    os.makedirs(results_dir, exist_ok=True)

    param_string = (
        f"LLM-{LLM_name}_"
        f"lr-{learning_rate}_"
        f"epochs-{n_epochs}_"
        f"batchsizetrain-{batch_size_train}_"
        f"gradientsteps-{gradient_accumulation_steps}"
    )    

    # print parameter string
    logger.info(f"Configuration: {target_name}_{param_string}")
    
    # ====================================================================================
    # Load CSV file and partition into train and test sets
    # ====================================================================================
    notes_df = pd.read_csv(notes_path, index_col=0)
    notes_df = notes_df[['mrn', 'note', 'treatment_date', target_name]].copy()
    # only keep notes where target value is not -1
    notes_df = notes_df[notes_df[target_name] != -1].copy()
    # assert that notes_df[target_name] has length 2
    assert len(notes_df[target_name].unique()) == 2

    # may need to add treatment date to the prompt
    notes_df = notes_df.rename(columns={target_name: "label"})
    # create a date string column from treatment_date
    notes_df['treatment_date_str'] = notes_df['treatment_date'].apply(lambda x: 
                                                                      datetime.strptime(x[:10], 
                                                                        "%Y-%m-%d").strftime("%b %d, %Y"))

    # split into development and test set
    # development set is the notes that are on or before the development_set_date
    train_set_df = notes_df[notes_df['treatment_date'] <= development_set_date].copy()
    test_set_df = notes_df[notes_df['treatment_date'] > development_set_date].copy()

    # ====================================================================================
    # Define the binary‐classification prompt template
    # Here we ask the LLM to output “0” or “1”
    # ====================================================================================

    string_to_add = generate_target_description(target_name, simplify=0)

    prompt_template = """
    Here is a de-identified clinical note from the past 30 days for a patient 
    receiving systemic cancer therapy on {}:
    {}

    Based on this note, will the patient {}
    class 0: No
    class 1: Yes

    SOLUTION
    The correct answer is: class {}"""

    train_set_df['text'] = formatting_prompts_func(train_set_df, prompt_template, string_to_add)
    train_set_df, valid_set_df = train_test_split(
        train_set_df,
        test_size=0.2,      # 20% for validation
        random_state=42,    # for reproducibility
        shuffle=True
    )

    train_set_df, eval_set_df = train_test_split(
        train_set_df,
        test_size=0.15,      # 15% for evaluation
        random_state=42,    # for reproducibility
        shuffle=True
    )

    # # save the train/valid/test sets
    # train_set_df.to_csv(os.path.join(results_dir, f"train_set_{param_string}.csv"))
    # valid_set_df.to_csv(os.path.join(results_dir, f"valid_set_{param_string}.csv"))
    # eval_set_df.to_csv(os.path.join(results_dir, f"eval_set_{param_string}.csv"))

    train_dataset = datasets.Dataset.from_pandas(train_set_df, preserve_index=False)
    eval_dataset = datasets.Dataset.from_pandas(eval_set_df, preserve_index=False)
    valid_dataset = datasets.Dataset.from_pandas(valid_set_df, preserve_index=False)

    NUM_CLASSES = 2  # ⇨ CHANGE: 2 classes (0 or 1)
    # assert NUM_CLASSES == 2
    assert NUM_CLASSES == 2, "NUM_CLASSES must be 2"

    max_seq_length = 8000 #8192

    trainer = LLMFineTuner(LLM_path, max_seq_length=max_seq_length, num_classes=NUM_CLASSES)
    trainer.load_model()
    trainer.create_collator()

    # ====================================================================================
    # Prepare TrainingArguments
    # ====================================================================================

    num_batches = len(train_dataset) // batch_size_train
    steps_per_epoch = num_batches // gradient_accumulation_steps
    total_steps = steps_per_epoch * n_epochs
    num_evals = 10

    eval_steps = max(1, total_steps // num_evals)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size_train,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=os.path.join(results_dir, f'{param_string}_checkpoints'),
        num_train_epochs=n_epochs,
        report_to="none",
        group_by_length=True,
        eval_strategy="steps",
        eval_steps = eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    csv_logger = CSVLoggerCallback(csv_path=os.path.join(results_dir, f"loss_log_{param_string}.csv"))
    trainer.train_model(train_dataset, eval_dataset, training_args, csv_logger)

    # ====================================================================================
    # Switch to inference mode
    # ====================================================================================

    trainer.fix_lm_head_for_inference()
    # save model
    save_path = os.path.join(results_dir, param_string)
    trainer.save_model(save_path)

    model, tokenizer = trainer.get_model_and_tokenizer()
    FastLanguageModel.for_inference(model)
    logger.info("Model is ready for inference.")

    # ================================================
    # Evaluate on validation set (build DataFrame)
    # ================================================

    # Build the inference prompt template
    # inference_prompt_template = (
    #     "Here is a de-identified clinical note from the past 30 days for a patient \n"
    #     "receiving systemic cancer therapy on {}:\n"
    #     "{}\n\n"
    #     "Based on this note, will the patient {}\n"
    #     "class "
    # )

    inference_prompt_template = """
    Here is a de-identified clinical note from the past 30 days for a patient 
    receiving systemic cancer therapy on {}:
    {}

    Based on this note, will the patient {}
    class 0: No
    class 1: Yes

    SOLUTION
    The correct answer is: class """

    set_seed(3407)

    device = model.device
    df_merged_train, train_auc, train_cross_entropy_loss = run_inference(
        inference_prompt_template,
        model,
        tokenizer,
        train_set_df,
        string_to_add,
        trainer.number_token_ids,
        device,
        max_seq_length,
        batch_size_test
    )

    df_merged_valid, valid_auc, valid_cross_entropy_loss = run_inference(
        inference_prompt_template,
        model,
        tokenizer,
        valid_set_df,
        string_to_add,
        trainer.number_token_ids,
        device,
        max_seq_length,
        batch_size_test
    )

    df_merged_test, test_auc, test_cross_entropy_loss = run_inference(
        inference_prompt_template,
        model,
        tokenizer,
        test_set_df,
        string_to_add,
        trainer.number_token_ids,
        device,
        max_seq_length,
        batch_size_test
    )
    
    logger.info(f"Train AUC: {train_auc:.4f}")
    logger.info(f"Train Cross-Entropy Loss: {train_cross_entropy_loss:.4f}")

    logger.info(f"Eval AUC: {valid_auc:.4f}")
    logger.info(f"Eval Cross-Entropy Loss: {valid_cross_entropy_loss:.4f}")

    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Cross-Entropy Loss: {test_cross_entropy_loss:.4f}")

    # save predictions to output directory
    csv_path = os.path.join(results_dir, f"{target_name}_train_predictions_{param_string}.csv")
    df_merged_train.to_csv(csv_path, index=False)
    csv_path = os.path.join(results_dir, f"{target_name}_valid_predictions_{param_string}.csv")
    df_merged_valid.to_csv(csv_path, index=False)
    csv_path = os.path.join(results_dir, f"{target_name}_test_predictions_{param_string}.csv")
    df_merged_test.to_csv(csv_path, index=False)

    # save train loss, test loss, train AUC, test AUC into csv file
    results = {
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "test_auc": test_auc,
        "train_loss": train_cross_entropy_loss,
        "valid_loss": valid_cross_entropy_loss,
        "test_loss": test_cross_entropy_loss,
    }
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(os.path.join(results_dir, f"{target_name}_metrics_{param_string}.csv"), index=False)

    # write code to load the finetuned model and perform inference
    # what happens when the number of evaluation steps is smaller than the total number of steps?

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("LLM_path", help="path of LLM", type=str)  # path of LLM
    parser.add_argument("notes_path", help="path of notes", type=str)  # path of notes
    parser.add_argument("target_name", help="name of target", type=str)  # name of target
    parser.add_argument("development_set_date", help="date of development set", type=str)  # date of development set
    parser.add_argument("results_dir", help="results directory", type=str)  # directory to save results
    parser.add_argument("learning_rate", help="learning rate", type=float) # learning rate 1e-4 (0.0001) to 5e-5 (0.00005)
    parser.add_argument("n_epochs", help="number of epochs", type=int) # number of epochs 1-3
    parser.add_argument("batch_size_train", help="batch size for training", type=int) # train batch size
    parser.add_argument("batch_size_test", help="batch size for testing", type=int) # test batch size
    parser.add_argument("gradient_accumulation_steps", help="gradient accumulation steps", type=int) # gradient accumulation steps
    args = parser.parse_args()

    main(
        args.LLM_path,
        args.notes_path,
        args.target_name,
        args.development_set_date,
        args.results_dir,
        args.learning_rate,
        args.n_epochs,
        args.batch_size_train,
        args.batch_size_test,
        args.gradient_accumulation_steps
    )