import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
import math
import re
import sys
import numpy as np
import pandas as pd

def mergeEmbedding(dataDir, dataFileName, saveDir, numFiles):
    
    embedding_list = []
    target_list = []

    for idx in range(numFiles):
        with np.load(f'{dataDir}/{dataFileName}_part{idx}.npz') as data:
            embedding_list.append( data['embeddings'] )
            target_list.append( data['target'] )
    
    embedding = np.concatenate( embedding_list )
    target = np.concatenate( target_list )

    np.savez( f'{saveDir}/{dataFileName}.npz', embedding = embedding, target = target )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataDir", help = "data directory", type = str) # data directory
    parser.add_argument("dataFileName", help = "data file name", type = str) # data file name
    parser.add_argument("saveDir", help = "save directory", type = str) # save directory
    parser.add_argument("numFiles", help = "number of files", type = int) # number of files
    args = parser.parse_args()

    mergeEmbedding( args.dataDir, args.dataFileName, args.saveDir, args.numFiles )