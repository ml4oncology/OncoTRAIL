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

def splitDataFrame(dataPath, saveDir, numRowsPerPart):

    # extract file name
    file_name = os.path.basename(dataPath)
    file_name = file_name[:-4]

    # load the clinical notes file
    clinical_notes = pd.read_csv(f'{dataPath}', index_col=False)

    # split into parts for parallel embedding processing
    ctr=0
    for idx in range(0, len(clinical_notes), numRowsPerPart):
        clinical_notes.loc[idx:idx+numRowsPerPart-1,:].to_csv(f'{saveDir}/{file_name}_part{ctr}.csv')
        ctr=ctr+1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataPath", help = "data file path", type = str) # data file path
    parser.add_argument("saveDir", help = "save directory", type = str) # save directory
    parser.add_argument("numRowsPerPart", help = "number of rows per part", type = int) # number of rows per dataframe
    args = parser.parse_args()

    splitDataFrame( args.dataPath, args.saveDir, args.numRowsPerPart )