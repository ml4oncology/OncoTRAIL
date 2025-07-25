import argparse
import logging
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)
logger = logging.getLogger(__name__)
from datetime import datetime
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import random
from tqdm import tqdm
import gc
from copy import deepcopy

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

def run_inference(
    model,
    tokenizer,
    dataset,
    device,
    batch_size
):
    """
    """
    model.to(device)
    model.eval()

    raw_dataset = dataset
    tensor_dataset = dataset.remove_columns(["mrn", "text"])  # exclude problematic string columns
    tensor_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

def perform_inference(model,
                    tokenizer,
                    dataset,
                    device,
                    batch_size,
                    str_descriptor,
                    results_dir,
                    target_name,
                    param_string):    
    
    data_set_type = ['train', 'valid', 'test']

    results = {}

    for data_type in data_set_type:
        
        prob_df, auc, loss = run_inference(
                                model,
                                tokenizer,
                                dataset[data_type],
                                device,
                                batch_size
                            )
        
        # log results
        logger.info(f"{data_type} AUC: {auc:.4f}")
        logger.info(f"{data_type} Cross-Entropy Loss: {loss:.4f}")

        results[f"{data_type}_auc"] = auc
        results[f"{data_type}_loss"] = loss

        # save results
        csv_path = os.path.join(results_dir, f"{str_descriptor}_{target_name}_{data_type}_predictions_{param_string}.csv")
        prob_df.to_csv(csv_path, index=False)

    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(os.path.join(results_dir, f"{str_descriptor}_{target_name}_metrics_{param_string}.csv"), index=False)

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
    notes_df = notes_df.rename(columns={target_name: "label", "note": "text"})

    # create a date string column from treatment_date
    notes_df['treatment_date_str'] = notes_df['treatment_date'].apply(lambda x: 
                                                                      datetime.strptime(x[:10], 
                                                                        "%Y-%m-%d").strftime("%b %d, %Y"))

    # split into development and test set
    # development set is the notes that are on or before the development_set_date
    train_set_df = notes_df[notes_df['treatment_date'] <= development_set_date].copy()
    test_set_df = notes_df[notes_df['treatment_date'] > development_set_date].copy()
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

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_set_df),
        "evaluation": Dataset.from_pandas(eval_set_df),
        "valid": Dataset.from_pandas(valid_set_df),
        "test": Dataset.from_pandas(test_set_df),
    })

    # find the maximum sequence length in the dataset
    tokenizer = AutoTokenizer.from_pretrained(LLM_path) 
    notes_df['token_length'] = notes_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
    max_seq_length = notes_df['token_length'].max()

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=max_seq_length)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}
    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        LLM_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    num_batches = len(dataset["train"]) // batch_size_train
    steps_per_epoch = num_batches // gradient_accumulation_steps
    total_steps = steps_per_epoch * n_epochs
    num_evals = 12 * n_epochs

    eval_steps = max(1, total_steps // num_evals)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_train,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate= learning_rate,
        warmup_steps=10,
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=os.path.join(results_dir, f'{param_string}_checkpoints'),
        num_train_epochs=n_epochs,
        report_to="none",
        group_by_length=True,
        bf16=True, # bfloat16 training 
        optim="adamw_torch_fused", # improved optimizer 
        eval_strategy="steps",
        save_strategy="steps", 
        save_total_limit=1,  
        eval_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    csv_logger = CSVLoggerCallback(csv_path=os.path.join(results_dir, f"loss_log_{param_string}.csv"))

    # Perform inference before finetuning
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_inference = deepcopy(model)
    model_for_inference.to(device)
    perform_inference(model_for_inference,
                    tokenizer,
                    tokenized_dataset,
                    device,
                    batch_size_test,
                    "pre_finetune",
                    results_dir,
                    target_name,
                    param_string)

    trainer = Trainer(
            args=training_args,
            model=model,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["evaluation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[csv_logger]
        )
    
    trainer.train()
    
    # save model
    save_path = os.path.join(results_dir, param_string)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # perform inference
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perform_inference(model,
                    tokenizer,
                    tokenized_dataset,
                    device,
                    batch_size_test,
                    "post_finetune",
                    results_dir,
                    target_name,
                    param_string)

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