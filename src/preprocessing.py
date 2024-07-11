import numpy as np
import pandas as pd
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
import torch
from pathlib import Path

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


def preprocessing(data_path, LLM_path, LLM_name, save_dir):
    # extract file name
    file_name = os.path.basename(data_path)
    file_name = Path(file_name).stem

    # load the clinical notes file
    clinical_notes = pd.read_csv(data_path, index_col=False)

    # find the target columns
    cols = clinical_notes.columns
    target_cols = cols[cols.str.contains("target")].tolist()
    clinical_notes[target_cols] = clinical_notes[target_cols].astype(int)
    embedding_target_dict = {}
    for elem in target_cols:
        embedding_target_dict[elem] = clinical_notes[elem].to_numpy()

    # convert note data to list
    notes_list = clinical_notes["note"].tolist()

    # define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    # set up LLM -- generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        LLM_path,
        num_labels=2,
        quantization_config=get_quant_config(),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_path)
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    embeddings_list = []

    def decoder_text_to_embedding(text):
        tokenized_texts = tokenizer(text, truncation=True, return_tensors="pt")

        with torch.no_grad():
            transformer_outputs = model.model(
                **tokenized_texts, output_hidden_states=True
            )

        hidden_states = transformer_outputs[0]

        input_ids = tokenized_texts["input_ids"]
        sequence_lengths = (
            torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
        )
        sequence_lengths = sequence_lengths % input_ids.shape[-1]

        return hidden_states[:, sequence_lengths].cpu().numpy()[0]

    def longformer_text_to_embedding(text):
        tokenized_texts = tokenizer(text, truncation=True, return_tensors="pt")

        with torch.no_grad():
            transformer_outputs = model.longformer(
                **tokenized_texts, output_hidden_states=True
            )

        hidden_states = transformer_outputs[0]

        return hidden_states[:, 0, :].cpu().numpy()[0]

    if LLM_name in ["Mistral", "BioMistral"]:
        text_to_embedding = decoder_text_to_embedding
    elif LLM_name in ["ClinicalLongformer"]:
        text_to_embedding = longformer_text_to_embedding
    else:
        raise Exception("Not implemented yet.")

    for ctr, note in enumerate(notes_list):
        embeddings_list.append(text_to_embedding(note))
        print(ctr)

        if ctr % 500 == 0:
            torch.cuda.empty_cache()

    embeddings = np.array(embeddings_list)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embedding_target_dict["embeddings"] = embeddings

    np.savez(f"{save_dir}/embedding_{LLM_name}_{file_name}.npz", **embedding_target_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data file path", type=str)  # data file path
    parser.add_argument("LLM_path", help="path to LLM", type=str)  # path to LLM
    parser.add_argument("LLM_name", help="name of LLM", type=str)  # name of LLM
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    args = parser.parse_args()

    preprocessing(args.data_path, args.LLM_path, args.LLM_name, args.save_dir)
