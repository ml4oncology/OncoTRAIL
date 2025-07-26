import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import datasets
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding
)
from sklearn.metrics import roc_auc_score, log_loss
from copy import deepcopy
import gc

logger = logging.getLogger(__name__)

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

class EncoderFineTuner:
    def __init__(self, LLM_path, target_name, results_dir, param_string):
        self.LLM_path = LLM_path
        self.target_name = target_name
        self.results_dir = results_dir
        self.param_string = param_string
        self.model = None
        self.tokenizer = None
        self.max_seq_length = None
        self.tokenized_datasets = None
        self.trainer = None

    def load_model(self):
        """Load model and tokenizer for encoder-based fine-tuning."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_path)
        
        # Define label mappings
        label2id = {"negative": 0, "positive": 1}
        id2label = {0: "negative", 1: "positive"}
        num_labels = 2

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.LLM_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        logger.info(f"Loaded encoder model: {self.LLM_path}")

    def calculate_max_seq_length(self, notes_df):
        """Calculate maximum sequence length based on the data."""
        notes_df['token_length'] = notes_df['text'].apply(lambda x: len(self.tokenizer.tokenize(x)))
        self.max_seq_length = notes_df['token_length'].max()
        logger.info(f"Maximum sequence length: {self.max_seq_length}")

    def prepare_datasets(self, train_set_df, eval_set_df, valid_set_df, test_set_df):
        """Prepare datasets for training."""
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_set_df),
            "evaluation": Dataset.from_pandas(eval_set_df),
            "valid": Dataset.from_pandas(valid_set_df),
            "test": Dataset.from_pandas(test_set_df),
        })

        # Tokenization function
        def tokenize(batch):
            return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=self.max_seq_length)

        # Tokenize datasets
        self.tokenized_dataset = dataset.map(tokenize, batched=True)

        return self.tokenized_dataset["train"], self.tokenized_dataset["evaluation"]

    def _run_inference(self, dataset, batch_size):
        """Run inference on a dataset."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        raw_dataset = dataset
        tensor_dataset = dataset.remove_columns(["mrn", "text"])  # exclude problematic string columns
        tensor_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, collate_fn=data_collator)

        all_probs = []
        all_labels = []
        all_mrns = []
        all_texts = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Running inference")):
                batch_metadata = [raw_dataset[j] for j in range(i * batch_size, min((i + 1) * batch_size, len(dataset)))]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = softmax(logits, dim=-1)

                all_probs.extend(probs.float().cpu().numpy())
                all_labels.extend(labels)
                all_mrns.extend([x["mrn"] for x in batch_metadata])
                all_texts.extend([x["text"] for x in batch_metadata])
        
        prob_df = pd.DataFrame({
            "prob": list(all_probs),  # list of numpy arrays
            "label": all_labels,
            "mrn": all_mrns,
            "text": all_texts
        })

        probs_np = np.stack(prob_df["prob"].values)
        true_labels_np = np.array(prob_df["label"].values)
        loss = log_loss(true_labels_np, probs_np)
        auc = roc_auc_score(true_labels_np, probs_np[:, 1])

        return prob_df, auc, loss

    def _perform_inference_on_sets(self, batch_size, str_descriptor):
        """Perform inference on all datasets."""
        
        data_set_types = ['train', 'valid', 'test']
        results = {}

        for data_type in data_set_types:
            prob_df, auc, loss = self._run_inference(
                self.tokenized_datasets[data_type],
                batch_size
            )
            
            # Log results
            logger.info(f"{data_type} AUC: {auc:.4f}")
            logger.info(f"{data_type} Cross-Entropy Loss: {loss:.4f}")

            results[f"{data_type}_auc"] = auc
            results[f"{data_type}_loss"] = loss

            # Save results
            csv_path = os.path.join(self.results_dir, f"{str_descriptor}_{self.target_name}_{data_type}_predictions_{self.param_string}.csv")
            prob_df.to_csv(csv_path, index=False)

        results_df = pd.DataFrame(results, index=[0])
        # drop the column "note" if it exists
        if "note" in results_df.columns:
            results_df = results_df.drop(columns=["note"])
        results_df.to_csv(os.path.join(self.results_dir, f"{str_descriptor}_{self.target_name}_metrics_{self.param_string}.csv"), index=False)

    def perform_pre_training_inference(self, batch_size):
        """Perform inference before training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_for_inference = deepcopy(self.model)
        model_for_inference.to(device)
        
        # Temporarily replace model for inference
        original_model = self.model
        self.model = model_for_inference
        
        self._perform_inference_on_sets(batch_size, "pre_finetune")
        
        # Restore original model
        self.model = original_model
        del model_for_inference
        gc.collect()
        torch.cuda.empty_cache()

    def perform_post_training_inference(self, batch_size):
        """Perform inference after training."""
        self._perform_inference_on_sets(batch_size, "post_finetune")

    def train_model(self, learning_rate, n_epochs, batch_size_train, gradient_accumulation_steps):
        """Train the encoder model."""
        num_batches = len(self.tokenized_datasets["train"]) // batch_size_train
        steps_per_epoch = num_batches // gradient_accumulation_steps
        total_steps = steps_per_epoch * n_epochs
        num_evals = 12 * n_epochs

        eval_steps = max(1, total_steps // num_evals)

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size_train,
            per_device_eval_batch_size=batch_size_train,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=10,
            logging_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=os.path.join(self.results_dir, f'{self.param_string}_checkpoints'),
            num_train_epochs=n_epochs,
            report_to="none",
            group_by_length=True,
            bf16=True,  # bfloat16 training 
            optim="adamw_torch_fused",  # improved optimizer 
            eval_strategy="steps",
            save_strategy="steps", 
            save_total_limit=1,  
            eval_steps=eval_steps,
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        csv_logger = CSVLoggerCallback(csv_path=os.path.join(self.results_dir, f"loss_log_{self.param_string}.csv"))

        self.trainer = Trainer(
            args=training_args,
            model=self.model,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["evaluation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            callbacks=[csv_logger]
        )
        
        self.trainer.train()
        
        torch.cuda.empty_cache()

    def save_model(self, output_dir):
        """Save the fine-tuned model."""
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


    # set_seed(3407)
    # # Extract model name from path
    # LLM_name = LLM_path.split("/")[-1]
    
    # # Create results directory
    # results_dir = os.path.join(results_dir, target_name)
    # os.makedirs(results_dir, exist_ok=True)
    
    # # Create parameter string for naming
    # param_string = (
    #     f"LLM-{LLM_name}_"
    #     f"lr-{learning_rate}_"
    #     f"epochs-{n_epochs}_"
    #     f"batchsizetrain-{batch_size_train}_"
    #     f"gradientsteps-{gradient_accumulation_steps}"
    # )
    
    # logger.info(f"Configuration: {target_name}_{param_string}")
    
    # # Prepare data
    # train_set_df, eval_set_df, valid_set_df, test_set_df = prepare_data(
    #     notes_path, target_name, development_set_date
    # )
    
    # finetuner = EncoderFineTuner(
    #         LLM_path=LLM_path,
    #         target_name=target_name,
    #         results_dir=results_dir,
    #         param_string=param_string
    #     )
    
    # # Load model and tokenizer
    # finetuner.load_model()

    # # Calculate maximum sequence length
    # finetuner.calculate_max_seq_length(pd.concat([train_set_df, eval_set_df, valid_set_df, test_set_df]))
    
    # train_dataset, eval_dataset = finetuner.prepare_datasets(
    #         train_set_df, eval_set_df, valid_set_df, test_set_df
    #     )
    
    # # Perform pre-training inference
    # set_seed(3407)
    # logger.info("Running inference before fine-tuning...")
    # finetuner.perform_pre_training_inference(
    #     train_set_df, valid_set_df, test_set_df, batch_size_test
    # )
    
    # # Train the model
    # logger.info("Starting fine-tuning...")
    # finetuner.train_model(
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     learning_rate=learning_rate,
    #     n_epochs=n_epochs,
    #     batch_size_train=batch_size_train,
    #     gradient_accumulation_steps=gradient_accumulation_steps
    # )
    
    # # Save the model
    # save_path = os.path.join(results_dir, param_string)
    
    # # Perform post-training inference
    # set_seed(3407)
    # logger.info("Running inference after fine-tuning...")
    # finetuner.perform_post_training_inference(
    #     train_set_df, valid_set_df, test_set_df, batch_size_test
    # )
    
    # logger.info("Fine-tuning complete!")