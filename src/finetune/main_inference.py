import argparse
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import random

# Import our custom modules
from oncotrail.finetune.main_train import set_seed
from oncotrail.finetune.encoder_finetuner import EncoderFineTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_inference_data(notes_path, target_name):
    """
    Load and prepare the dataset for inference.
    
    Returns:
        pd.DataFrame: Prepared dataset
    """
    # Load CSV file
    notes_df = pd.read_csv(notes_path, header=0)
    if isinstance(notes_df.columns, pd.MultiIndex):
        notes_df.columns = ["_".join(map(str, col)).strip() for col in notes_df.columns.values]

    notes_df = notes_df[['mrn', 'note', 'treatment_date', target_name]].copy()
    
    # Only keep notes where target value is not -1
    notes_df = notes_df[notes_df[target_name] != -1].copy()

    # Assert that notes_df[target_name] has length 2
    assert len(notes_df[target_name].unique()) == 2, f"Target {target_name} must have exactly 2 unique values"

    # Rename columns and create date string
    notes_df = notes_df.rename(columns={target_name: "label"})
    notes_df['text'] = notes_df['note']
    notes_df['treatment_date_str'] = notes_df['treatment_date'].apply(
        lambda x: datetime.strptime(x[:10], "%Y-%m-%d").strftime("%b %d, %Y")
    )

    return notes_df

def main(
    model_type,
    saved_model_path,
    notes_path,
    target_name,
    results_dir,
    batch_size_test
):
    """
    Main function that handles inference with saved models.
    
    Args:
        model_type (str): Either 'encoder' or 'decoder'
        saved_model_path (str): Path to the saved fine-tuned model
        notes_path (str): Path to the notes CSV file
        target_name (str): Name of the target column
        results_dir (str): Directory to save inference results
        batch_size_test (int): Testing batch size
    """
    
    # Check if decoder type and print error
    if model_type.lower() == 'decoder':
        logger.error("Decoder inference is not implemented yet.")
        raise NotImplementedError("Decoder inference is not implemented yet.")
    
    if model_type.lower() != 'encoder':
        raise ValueError(f"model_type must be either 'encoder' or 'decoder', got {model_type}")
    
    set_seed(3407)
    
    # Extract model name from path for naming
    model_name = os.path.basename(saved_model_path.rstrip('/'))
    
    # Create results directory
    results_dir = os.path.join(results_dir, target_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create parameter string for naming (simplified for inference)
    param_string = f"inference_{model_name}"
    
    logger.info(f"Running inference with: {target_name}_{param_string}")
    logger.info(f"Saved model path: {saved_model_path}")
    
    # Prepare data
    inference_df = prepare_inference_data(notes_path, target_name)
    logger.info(f"Loaded {len(inference_df)} samples for inference")
    
    # Initialize the encoder fine-tuner with the saved model path
    finetuner = EncoderFineTuner(
        LLM_path=saved_model_path,  # Use saved model path instead of original model
        target_name=target_name,
        results_dir=results_dir,
        param_string=param_string
    )
    
    # Load the saved model and tokenizer
    finetuner.load_model()
    logger.info(f"Loaded saved model from: {saved_model_path}")

    # Calculate maximum sequence length
    finetuner.calculate_max_seq_length(inference_df)
    
    # Prepare dataset for inference (we'll use the inference data as "test" data)
    # Create dummy datasets for the other splits since the existing function expects all splits
    # Need at least one sample from each class to avoid log_loss error
    unique_labels = inference_df['label'].unique()
    dummy_rows = []
    for label in unique_labels:
        dummy_rows.append(inference_df[inference_df['label'] == label].iloc[0])
    dummy_df = pd.DataFrame(dummy_rows).reset_index(drop=True)
    
    train_set_df, eval_set_df, valid_set_df, test_set_df = finetuner.prepare_datasets(
        dummy_df, dummy_df, dummy_df, inference_df
    )
    
    # Perform inference on the dataset
    set_seed(3407)
    logger.info("Running inference on provided dataset...")
    
    # We'll use a modified version of the post-training inference that only runs on test set
    finetuner._perform_inference_on_sets(
        train_set_df, eval_set_df, valid_set_df, test_set_df, 
        batch_size_test, "inference"
    )
    
    logger.info("Inference complete!")
    logger.info(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with saved fine-tuned models on new datasets")
    
    parser.add_argument("model_type", choices=['encoder', 'decoder'], 
                       help="Type of model: 'encoder' or 'decoder'")
    parser.add_argument("saved_model_path", help="Path to the saved fine-tuned model directory", type=str)
    parser.add_argument("notes_path", help="Path to notes CSV file for inference", type=str)
    parser.add_argument("target_name", help="Name of target column", type=str)
    parser.add_argument("results_dir", help="Directory to save inference results", type=str)
    parser.add_argument("batch_size_test", help="Testing batch size", type=int)
    
    args = parser.parse_args()
    
    main(
        args.model_type,
        args.saved_model_path,
        args.notes_path,
        args.target_name,
        args.results_dir,
        args.batch_size_test
    )