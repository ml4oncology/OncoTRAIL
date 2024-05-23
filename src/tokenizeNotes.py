import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer

def tokenizeNotes( dataDir, modelName, modelPath, notesFileName ):
    notes = pd.read_csv(f'{dataDir}/{notesFileName}')
    tokenizer = AutoTokenizer.from_pretrained(f'{modelPath}/{modelName}')
    # Batched tokenization
    if tokenizer.padding_side is None: tokenizer.padding_side = "right"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # convert column to list of strings
    strToTokenize = notes['note'].tolist()

    nnzList = []

    # do in batches of 1000
    sliceSize = 1000
    for i in range(0, len(strToTokenize), sliceSize):
        strToTokenizeSlice = strToTokenize[i:i+sliceSize]
        encodedBatchSlice = tokenizer.batch_encode_plus(strToTokenizeSlice, padding=True, truncation=False, return_tensors="np")
        inputIdsSlice = encodedBatchSlice["input_ids"]
        nnzSlice = np.sum(inputIdsSlice != tokenizer.pad_token_id, axis = 1)
        nnzSlice = nnzSlice.ravel()
        nnzList.append(nnzSlice)

    nnz = np.concatenate(nnzList)

    notes[f'token_length_{modelName}'] = nnz
    notes.to_csv(f'{dataDir}/{notesFileName}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataDir", help = "data directory", type = str) # data directory
    parser.add_argument("modelName", help = "model name", type = str) # model name
    parser.add_argument("modelPath", help = "path to model", type = str) # path to model
    parser.add_argument("notesFileName", help = "notes file name", type = str) # path to notes file
    args = parser.parse_args()

    tokenizeNotes( args.dataDir, args.modelName, args.modelPath, args.notesFileName )