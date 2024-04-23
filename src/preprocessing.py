import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
import math
import re
import sys
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig)
import torch
import numpy as np
import pandas as pd

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

def preprocessing(dataPath, LLMpath, saveDir, targetCol):

    # extract file name
    file_name = os.path.basename(dataPath)
    file_name = file_name[:-4]

    # load the clinical notes file
    clinical_notes = pd.read_csv(f'{dataPath}', index_col=False)
    clinical_notes[targetCol] = clinical_notes[targetCol].astype(int)
    target = clinical_notes[targetCol].to_numpy()

    # convert note data to list
    notesList = clinical_notes['note'].tolist()

    # define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative":0, "Positive":1}

    # set up LLM -- generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(LLMpath, num_labels=2,\
                                                            quantization_config=get_quant_config(),\
                                                            id2label=id2label, label2id=label2id, device_map='auto')
    model.config.pad_token_id = model.config.eos_token_id

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLMpath)
    if tokenizer.padding_side is None: tokenizer.padding_side = "right"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    embeddings_list = []

    def text_to_embedding(text):

        tokenized_texts = tokenizer( text, truncation=True, return_tensors = "pt" )

        with torch.no_grad():
            transformer_outputs = model.model(**tokenized_texts, output_hidden_states=True)

        hidden_states = transformer_outputs[0]

        input_ids = tokenized_texts['input_ids']
        sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]

        return hidden_states[:, sequence_lengths].cpu().numpy()

    ctr = 0
    for note in notesList:

        embeddings_list.append( text_to_embedding(note)[0] )
        print( ctr )
        ctr = ctr + 1

        if ctr % 500 == 0:
            torch.cuda.empty_cache()

    embeddings = np.array( embeddings_list )
    embeddings = embeddings.reshape( embeddings.shape[0], -1)

    np.savez( f'{saveDir}/embedding_{file_name}.npz', embeddings = embeddings, target = target )

    # to do: add name of LLM in file name
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataPath", help = "data file path", type = str) # data file path
    parser.add_argument("LLMpath", help = "path to LLM", type = str) # path to LLM
    parser.add_argument("saveDir", help = "save directory", type = str) # save directory
    parser.add_argument("targetCol", help = "name of target column", type = str) # name of target column
    args = parser.parse_args()

    preprocessing( args.dataPath, args.LLMpath, args.saveDir, args.targetCol )