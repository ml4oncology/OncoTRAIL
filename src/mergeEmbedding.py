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

    # initialize the dictionary here
    target_embedding_list = {}

    for idx in range(numFiles):
        with np.load(f'{dataDir}/{dataFileName}_part{idx}.npz') as data:

            # initialize all keys in the dictionary
            dict_keys = list(data.keys())
            if idx == 0:
                for elem in dict_keys:
                    target_embedding_list[elem] = []

            for elem in dict_keys:
                target_embedding_list[elem].append( data[elem] )

    target_embedding_concat = {}
    for key in target_embedding_list:
        target_embedding_concat[key] = np.concatenate( target_embedding_list[key] )

    np.savez( f'{saveDir}/{dataFileName}.npz', **target_embedding_concat )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataDir", help = "data directory", type = str) # data directory
    parser.add_argument("dataFileName", help = "data file name", type = str) # data file name
    parser.add_argument("saveDir", help = "save directory", type = str) # save directory
    parser.add_argument("numFiles", help = "number of files", type = int) # number of files
    args = parser.parse_args()

    mergeEmbedding( args.dataDir, args.dataFileName, args.saveDir, args.numFiles )