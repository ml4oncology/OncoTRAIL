import numpy as np
import pandas as pd
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    log_loss)
from sklearn.preprocessing import StandardScaler
from .config import startTestDate
from .split import genDataSplit
from .train import Trainer

def main( notesPath, embeddingPath, splitConfig, hyperParamEval, modelName, setupStr, targetName, modelDir, resultsDir ):

    # save string for file
    file_save_str = f'{modelName}_{setupStr}_{splitConfig}_{hyperParamEval}_{targetName}'
    print(file_save_str)

    # load data frame
    df = pd.read_csv(f'{notesPath}', index_col=0)
    df.reset_index(drop=True,inplace=True)

    # get indices of target != -1
    mask = (df[targetName] != -1).to_numpy()

    # load embedding
    with np.load(f'{embeddingPath}') as data:
        embedding = data['embeddings']
        target = data[targetName]
    
    # only extract embedding and target where index != -1
    embedding = embedding[mask,:]
    target = target[mask]
    df = df.loc[mask, ['mrn', 'treatment_date', 'note', targetName]]
    df.reset_index(drop=True,inplace=True)

    # generate train-validation-test split
    X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test = genDataSplit(df, startTestDate, splitConfig, embedding, target)

    # preprocess the data by scaling and centering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # if Logistic Regression, merge train and evaluation data sets
    if modelName == 'LR':
        X_train = np.concatenate([X_train, X_eval])
        Y_train = np.concatenate([Y_train, Y_eval])
        X_eval = None
        Y_eval = None

    # call trainer on predictions
    trainer = Trainer(X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, hyperParamEval, modelDir, modelName, file_save_str)
    train_pred, val_pred, test_pred = trainer.run()

    # save data
    np.savez(  f'{resultsDir}/predData_{file_save_str}.npz', train_pred = train_pred, val_pred = val_pred, test_pred = test_pred, Y_train = Y_train, Y_valid = Y_valid, Y_test = Y_test  )

    # evaluate errors
    def evaluate(Y, pred, split):
        auprc = average_precision_score(Y, pred)
        auroc = roc_auc_score(Y, pred)
        LogLoss = log_loss(Y, pred)
        result = {'AUPRC': auprc, 'AUROC': auroc, 'LogLoss': LogLoss}

        return pd.DataFrame(result, index = [split])

    train_results = evaluate(Y_train, train_pred, split='train')
    valid_results = evaluate(Y_valid, val_pred, split='valid')
    test_results = evaluate(Y_test, test_pred, split='test')

    results = pd.concat([train_results, valid_results, test_results])
    results.to_csv(f'{resultsDir}/{file_save_str}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notesPath", help = "path of notes", type = str ) # path of notes
    parser.add_argument("embeddingPath", help = "path of embedding", type = str) # path of embedding
    parser.add_argument("splitConfig", help = "configuration of train-valid-test split", type = str) # configuration of train-valid-test split
    parser.add_argument("hyperParamEval", help = "function for hyperparameter evaluation", type = str) # hyperparameter evaluation function
    parser.add_argument("modelName", help = "model name", type = str) # name of machine learning model
    parser.add_argument("setupStr", help = "set up string", type = str) # name of set up string
    parser.add_argument("targetName", help = "name of target", type = str) # name of target
    parser.add_argument("modelDir", help = "model directory", type = str) # directory to save model
    parser.add_argument("resultsDir", help = "results directory", type = str) # directory to save results

    args = parser.parse_args()

    main( args.notesPath, args.embeddingPath, args.splitConfig, args.hyperParamEval, args.modelName, args.setupStr, args.targetName, args.modelDir, args.resultsDir )