import argparse
import logging
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import random
from llm_notes_classification.config import date_lower_limit, date_upper_limit

# Import our custom modules
# from ml_common.prep import Splitter
from make_clinical_dataset.epr.prep import Splitter
# from llm_notes_classification.finetune.decoder_finetuner import DecoderFineTuner
from llm_notes_classification.finetune.encoder_finetuner import EncoderFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def prepare_data(notes_path, target_name, development_set_date):
    """
    Load and prepare the dataset for training.
    
    Returns:
        tuple: (train_set_df, eval_set_df, valid_set_df, test_set_df)
    """
    # Load CSV file and partition into train and test sets
    notes_df = pd.read_csv(notes_path, index_col=0)
    notes_df = notes_df[['mrn', 'note', 'treatment_date', target_name]].copy()
    
    # Only keep notes where target value is not -1
    notes_df = notes_df[notes_df[target_name] != -1].copy()
    
    # restrict date range
    notes_df = notes_df[(notes_df['treatment_date'] >= date_lower_limit) & 
                        (notes_df['treatment_date'] <= date_upper_limit)].copy()

    # Assert that notes_df[target_name] has length 2
    assert len(notes_df[target_name].unique()) == 2, f"Target {target_name} must have exactly 2 unique values"

    # Rename columns and create date string
    notes_df = notes_df.rename(columns={target_name: "label"})
    notes_df['text'] = notes_df['note']
    notes_df['treatment_date_str'] = notes_df['treatment_date'].apply(
        lambda x: datetime.strptime(x[:10], "%Y-%m-%d").strftime("%b %d, %Y")
    )

    # # Split into development and test set
    # train_set_df = notes_df[notes_df['treatment_date'] <= development_set_date].copy()
    # test_set_df = notes_df[notes_df['treatment_date'] > development_set_date].copy()
    
    # # Further split training set
    # train_set_df, valid_set_df = train_test_split(
    #     train_set_df,
    #     test_size=0.2,
    #     random_state=42,
    #     shuffle=True
    # )

    # train_set_df, eval_set_df = train_test_split(
    #     train_set_df,
    #     test_size=0.15,
    #     random_state=42,
    #     shuffle=True
    # )

    splitter = Splitter()
    train_eval_data, valid_set_df, test_set_df = splitter.split_data(
            notes_df, development_set_date
        )
    train_set_df, eval_set_df = splitter.random_split(train_eval_data, test_size=0.15)

    return train_set_df, eval_set_df, valid_set_df, test_set_df

def main(
    model_type,
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
    Main function that handles both encoder and decoder fine-tuning.
    
    Args:
        model_type (str): Either 'encoder' or 'decoder'
        LLM_path (str): Path to the model
        notes_path (str): Path to the notes CSV file
        target_name (str): Name of the target column
        development_set_date (str): Date cutoff for development set
        results_dir (str): Directory to save results
        learning_rate (float): Learning rate for training
        n_epochs (int): Number of training epochs
        batch_size_train (int): Training batch size
        batch_size_test (int): Testing batch size
        gradient_accumulation_steps (int): Gradient accumulation steps
    """
    
    set_seed(3407)
    
    # Extract model name from path
    LLM_name = LLM_path.split("/")[-1]
    
    # Create results directory
    results_dir = os.path.join(results_dir, target_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create parameter string for naming
    param_string = (
        f"LLM-{LLM_name}_"
        f"lr-{learning_rate}_"
        f"epochs-{n_epochs}_"
        f"batchsizetrain-{batch_size_train}_"
        f"gradientsteps-{gradient_accumulation_steps}"
    )
    
    logger.info(f"Configuration: {target_name}_{param_string}")
    
    # Prepare data
    train_set_df, eval_set_df, valid_set_df, test_set_df = prepare_data(
        notes_path, target_name, development_set_date
    )
    
    # Initialize the appropriate fine-tuner
    if model_type.lower() == 'decoder':
        # finetuner = DecoderFineTuner(
        #     LLM_path=LLM_path,
        #     target_name=target_name,
        #     results_dir=results_dir,
        #     param_string=param_string,
        #     max_seq_length=8192,
        #     num_classes=2
        # )
        logger.error("Decoder inference is not implemented yet.")
        raise NotImplementedError("Decoder training is not implemented yet.")
    elif model_type.lower() == 'encoder':
        finetuner = EncoderFineTuner(
            LLM_path=LLM_path,
            target_name=target_name,
            results_dir=results_dir,
            param_string=param_string
        )
    else:
        raise ValueError(f"model_type must be either 'encoder' or 'decoder', got {model_type}")
    
    # Load model and tokenizer
    finetuner.load_model()

    # Calculate maximum sequence length
    finetuner.calculate_max_seq_length(pd.concat([train_set_df, eval_set_df, valid_set_df, test_set_df]))
    
    # Prepare datasets for training/pre-training inference
    train_set_df, eval_set_df, valid_set_df, test_set_df = finetuner.prepare_datasets(
            train_set_df, eval_set_df, valid_set_df, test_set_df
        )
        
    # Perform pre-training inference
    set_seed(3407)
    logger.info("Running inference before fine-tuning...")
    finetuner.perform_pre_training_inference(
        train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size_test
    )
    
    # Train the model
    logger.info("Starting fine-tuning...")
    finetuner.train_model(
        train_dataset=train_set_df,
        eval_dataset=eval_set_df,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size_train=batch_size_train,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Save the model
    save_path = os.path.join(results_dir, param_string)
    finetuner.save_model(save_path)
    
    # Perform post-training inference
    set_seed(3407)
    logger.info("Running inference after fine-tuning...")
    finetuner.perform_post_training_inference(
        train_set_df, eval_set_df, valid_set_df, test_set_df, batch_size_test
    )
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune encoder or decoder models for clinical note classification")
    
    parser.add_argument("model_type", choices=['encoder', 'decoder'], 
                       help="Type of model to fine-tune: 'encoder' or 'decoder'")
    parser.add_argument("LLM_path", help="Path to the model", type=str)
    parser.add_argument("notes_path", help="Path to notes CSV file", type=str)
    parser.add_argument("target_name", help="Name of target column", type=str)
    parser.add_argument("development_set_date", help="Date cutoff for development set", type=str)
    parser.add_argument("results_dir", help="Results directory", type=str)
    parser.add_argument("learning_rate", help="Learning rate", type=float)
    parser.add_argument("n_epochs", help="Number of epochs", type=int)
    parser.add_argument("batch_size_train", help="Training batch size", type=int)
    parser.add_argument("batch_size_test", help="Testing batch size", type=int)
    parser.add_argument("gradient_accumulation_steps", help="Gradient accumulation steps", type=int)
    
    args = parser.parse_args()
    
    main(
        args.model_type,
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