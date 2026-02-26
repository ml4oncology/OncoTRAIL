import os
os.environ['UNSLOTH_DISABLE_RL_PATCHING'] = '1'

from unsloth import FastLanguageModel
import math
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datasets
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, AutoTokenizer, DataCollatorForLanguageModeling
from sklearn.metrics import roc_auc_score, log_loss
from threading import Thread
import pynvml
import psutil
import time
import gc
from typing import Dict, List, Union, Any
from oncotrail.prompt.generate_prompts import generate_target_description

# Suppress unsloth warnings
from unsloth import tokenizer_utils
def do_nothing(*args, **kwargs):
    pass
tokenizer_utils.fix_untrained_tokens = do_nothing

# Environment setup
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["UNSLOTH_CACHE_DISABLE"] = "1"
os.environ["FLASH_ATTENTION_FORCE"] = "True"
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

logger = logging.getLogger(__name__)

def monitor_gpu_memory(interval_sec=5):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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

class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        reverse_map: Dict[int, int],
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.reverse_map = reverse_map

    def torch_call(
        self,
        examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            # Find last non-ignored token
            last_token_idx = (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
            # Mask everything except that last token
            batch["labels"][i, :last_token_idx] = self.ignore_index

            # Re-map old_id → new lm_head index via self.reverse_map
            old_id = batch["labels"][i, last_token_idx].item()
            batch["labels"][i, last_token_idx] = self.reverse_map[old_id]

        return batch

def formatting_prompts_func(df_, prompt_template, string_to_add):
    texts = []
    targets = df_["label"].tolist()
    for date_, text_, lbl in zip(df_["treatment_date_str"], df_["note"], targets):
        full = prompt_template.format(date_, text_, string_to_add, lbl)
        texts.append(full)
    return texts

def tokenize_function_factory(tokenizer, max_length):
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
        )
    return tokenize_function

class DecoderFineTuner:
    def __init__(self, LLM_path, target_name, results_dir, param_string, max_seq_length=8192, num_classes=2):
        self.LLM_path = LLM_path
        self.target_name = target_name
        self.results_dir = results_dir
        self.param_string = param_string
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
        self.string_to_add = None
        self.prompt_template = None
        self.inference_prompt_template = None

    def load_model(self):
        """Load model and tokenizer, prepare for fine-tuning."""
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

        # Setup prompt templates
        self.string_to_add = generate_target_description(self.target_name, 0, False, False, True)
        self.prompt_template = """
            Here is a de-identified clinical note from the past 30 days for a patient 
            receiving systemic cancer therapy on {}:
            {}

            Based on this note, will the patient {}
            class 0: No
            class 1: Yes

            SOLUTION
            The correct answer is: class {}"""
        
        self.inference_prompt_template = self.prompt_template.split("class {}")[0] + "class "

        # Create data collator
        self.collator = DataCollatorForLastTokenLM(
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100,
            reverse_map=self.reverse_map,
        )

    def calculate_max_seq_length(self, notes_df):
        """Calculate optimal max_seq_length based on data."""
        tokenizer_temp = AutoTokenizer.from_pretrained(self.LLM_path) 
        notes_df['token_length'] = notes_df['note'].apply(lambda x: len(tokenizer_temp.tokenize(x)))
        max_seq_length_temp = notes_df['token_length'].max()
        
        # Add buffer for prompt template and string_to_add
        max_seq_length_temp += len(tokenizer_temp.tokenize(self.string_to_add))
        max_seq_length_temp += len(tokenizer_temp.tokenize(self.prompt_template))
        max_seq_length = min(8192, math.ceil((max_seq_length_temp + 100) / 8) * 8)
        
        logger.info(f"Calculated max_seq_length: {max_seq_length}")
        self.max_seq_length = max_seq_length

    def prepare_datasets(self, train_set_df, eval_set_df, valid_set_df, test_set_df):
        """Prepare datasets for training."""

        # Format training data with prompts
        train_set_df = train_set_df.copy()
        train_set_df['text'] = formatting_prompts_func(train_set_df, self.prompt_template, self.string_to_add)
        
        eval_set_df = eval_set_df.copy()
        eval_set_df['text'] = formatting_prompts_func(eval_set_df, self.prompt_template, self.string_to_add)

        return train_set_df, eval_set_df, valid_set_df, test_set_df 

    def _run_inference(self, df, batch_size):
        """Run inference on a DataFrame."""
        device = self.model.device
        df = df.copy()
        df["prompt_text"] = df.apply(
            lambda row: self.inference_prompt_template.format(
                row["treatment_date_str"], row["note"], self.string_to_add
            ),
            axis=1,
        )

        df["token_length"] = df["prompt_text"].apply(
            lambda txt: len(self.tokenizer.encode(txt, add_special_tokens=False))
        )

        df_sorted = df.sort_values(by="token_length").reset_index(drop=True)
        results = []

        with torch.inference_mode():
            for i in tqdm(range(0, len(df_sorted), batch_size), desc="Running inference"):
                batch = df_sorted.iloc[i : i + batch_size]
                prompts = batch["prompt_text"].tolist()

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                logits = self.model(**inputs).logits
                last_idxs = inputs["attention_mask"].sum(1) - 1
                last_logits = logits[torch.arange(len(batch)), last_idxs, :]

                probs_all = F.softmax(last_logits, dim=-1)
                probs = probs_all[:, self.number_token_ids]
                probs = probs / probs.sum(dim=1, keepdim=True)

                preds = torch.argmax(probs, dim=-1).cpu().numpy()

                for j in range(len(batch)):
                    pred_label = int(preds[j])
                    pred_prob = probs[j, :].float().cpu().numpy()

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

        probs_np = np.stack(df_merged["prob"].values)
        auc_val = roc_auc_score(df_merged["label"], probs_np[:, 1])
        cross_entropy_loss = log_loss(df_merged['label'], probs_np, labels=[0, 1])

        return df_merged, auc_val, cross_entropy_loss

    def _perform_inference_on_sets(self, train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size, str_descriptor):
        """Perform inference on all data sets."""
        data_set_dict = {
            'train': train_set_df,
            'eval': eval_set_df,
            'valid': valid_set_df,
            'test': test_set_df
        }
        
        results = {}
        
        for key in data_set_dict:
            df_merged, auc, loss = self._run_inference(data_set_dict[key], batch_size)
            
            logger.info(f"{str_descriptor} {key} AUC: {auc:.4f}")
            logger.info(f"{str_descriptor} {key} Cross-Entropy Loss: {loss:.4f}")

            results[f"{key}_auc"] = auc
            results[f"{key}_loss"] = loss

            csv_path = os.path.join(self.results_dir, f"{str_descriptor}_{self.target_name}_{key}_predictions_{self.param_string}.csv")
            df_merged.to_csv(csv_path, index=False)

        results_df = pd.DataFrame(results, index=[0])
        results_df.to_csv(os.path.join(self.results_dir, f"{str_descriptor}_{self.target_name}_metrics_{self.param_string}.csv"), index=False)

    def perform_pre_training_inference(self, train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size):
        """Perform inference before training."""
        self.fix_lm_head_for_inference()
        FastLanguageModel.for_inference(self.model)
        self._perform_inference_on_sets(train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size, "pre_finetune")
        
        # Reset model for training
        self.load_model()

    def perform_post_training_inference(self, train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size):
        """Perform inference after training."""
        FastLanguageModel.for_inference(self.model)
        self._perform_inference_on_sets(train_set_df, eval_set_df,valid_set_df, test_set_df, batch_size, "post_finetune")

    def train_model(self, train_dataset, eval_dataset, learning_rate, n_epochs, batch_size_train, gradient_accumulation_steps):
        """Train the model."""

        # Create datasets
        train_dataset = datasets.Dataset.from_pandas(train_dataset, preserve_index=False)
        eval_dataset = datasets.Dataset.from_pandas(eval_dataset, preserve_index=False)

        num_batches = len(train_dataset) // batch_size_train
        steps_per_epoch = num_batches // gradient_accumulation_steps
        total_steps = steps_per_epoch * n_epochs
        num_evals = 12 * n_epochs
        eval_steps = max(1, total_steps // num_evals)

        training_args = SFTConfig(
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
            output_dir=os.path.join(self.results_dir, f'{self.param_string}_checkpoints'),
            num_train_epochs=n_epochs,
            report_to="none",
            group_by_length=True,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        csv_logger = CSVLoggerCallback(csv_path=os.path.join(self.results_dir, f"loss_log_{self.param_string}.csv"))

        # Tokenize datasets
        tokenize_function = tokenize_function_factory(self.tokenizer, max_length=self.max_seq_length)
        train_dataset = train_dataset.map(tokenize_function, batched=True)

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

        # Print GPU memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_gpu = round(gpu_stats.total_memory / 1024**3, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_gpu} GB. Start reserved = {start_gpu} GB.")

        Thread(target=monitor_gpu_memory, daemon=True).start()
        Thread(target=monitor_cpu_memory, daemon=True).start()

        trainer_stats = trainer.train()

        # Print final memory/time stats
        used_gpu = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_for_lora = round(used_gpu - start_gpu, 3)
        percent_used = round(used_gpu / max_gpu * 100, 3)
        percent_lora = round(used_for_lora / max_gpu * 100, 3)
        logger.info(f"Training runtime: {trainer_stats.metrics['train_runtime']} seconds")
        logger.info(f"Peak GPU memory reserved: {used_gpu} GB ({percent_used}%).  LoRA‐only: {percent_lora}%.")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        self.fix_lm_head_for_inference()

    def fix_lm_head_for_inference(self):
        """Reconstruct full lm_head for inference."""
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
        """Save the fine-tuned model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")