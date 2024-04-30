from functools import partial
import copy

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch
from util import save_pickle
from config import bayesopt_param, model_static_param, model_tuning_param, algs
torch.manual_seed(0)
np.random.seed(0)

###############################################################################
# Tune Models
###############################################################################
class Tuner:
    def __init__(self, X_train, Y_train, X_valid, Y_valid, score_func, output_path, alg):
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid

        if score_func == 'AUROC':
            self.score_func = roc_auc_score
        elif score_func == 'logloss':
            self.score_func = -1.0*log_loss
        self.output_path = output_path
        
        self.model_static_param = copy.deepcopy(model_static_param)[alg]
        self.model_tuning_param = copy.deepcopy(model_tuning_param)[alg]
        self.bayesopt_param = copy.deepcopy(bayesopt_param)[alg]
        self.alg = copy.deepcopy(algs)[alg]
    
    def bayesopt(
        self, 
        filename, 
        random_state=42, 
        bopt_kwargs=None
    ):
        """Conduct bayesian optimization, a sequential search framework 
        for finding optimal hyperparameters using bayes theorem
        """
        if bopt_kwargs is None: bopt_kwargs = {}

        # set up
        hyperparam_config = self.model_tuning_param
        optim_config = self.bayesopt_param
        eval_func = self._eval_func
        bo = BayesianOptimization(
            f=eval_func, 
            pbounds=hyperparam_config, 
            verbose=2,
            random_state=random_state, 
            **bopt_kwargs
        )
        
        # find the best hyperparameters
        bo.maximize(**optim_config)
        best_param = bo.max['params']
        best_param = self.convert_hyperparams(best_param)

        # save the best hyperparameters
        save_pickle(best_param, f'{self.output_path}', filename)

        return best_param
    
    def _eval_func(self, *args, **kwargs):
        """Evaluation function for bayesian optimization"""
        raise NotImplementedError
    
    def convert_hyperparams(self, best_param):
        """You can overwrite this to convert the hyperparmeters as desired
        """
        return best_param
    
###############################################################################
# Train Models
###############################################################################
class Trainer(Tuner):
    def __init__(self, X_train, Y_train, X_valid, Y_valid, score_func, output_path, alg, strIdentifier, **kwargs):
        """
        Args:
            **kwargs (dict): the parameters of MLModels
        """
        super().__init__(X_train, Y_train, X_valid, Y_valid, score_func, output_path, alg)
        self.strIdentifier = strIdentifier
        
    def run(
        self, 
        bayes_kwargs=None,
        **kwargs
    ):
        """
        Args:
            train_kwargs: keyword arguments fed into Trainer.train_model
            bayes_kwargs: keyword arguments fed into BayesianOptimization
        """
        if bayes_kwargs is None: bayes_kwargs = {}

        # Hyperparameter Tuning

        best_param = self.bayesopt(filename=self.strIdentifier, bopt_kwargs=bayes_kwargs)

        train_kwargs = {}
        # NOTE: train_kwargs takes precedence if there are duplicate keys
        for name, param in best_param.items():
            train_kwargs[name] = train_kwargs.get(name, param)

        # Model Training
        model = self.train_model(**train_kwargs)

        # Apply model to training data
        train_pred = self.predict(model, self.X_train)

        # Apply model to validation data
        val_pred = self.predict(model, self.X_valid)

        # Prediction
        
        return train_pred, val_pred
            
    def train_model(self, **kwargs):
            
        model = self.train_ml_model(**kwargs, **self.model_static_param)
    
        return model
        
    def train_ml_model(self, **kwargs):
        """Train machine learning models"""
        model = self.alg(**kwargs)
        
        model.fit(self.X_train, self.Y_train)        
        return model
    
    def _eval_func(self, **kwargs):
        """Evaluation function for bayesian optimization
        """
        kwargs = self.convert_hyperparams(kwargs)
        model = self.train_model(**kwargs)

        pred = self.predict(model, self.X_valid)

        return self.score_func(self.Y_valid, pred)
    
    def predict(self, model, data):

        pred = model.predict(data)
            
        return pred
    
    def convert_hyperparams(self, params):
        cat_param_choices = np.geomspace(start=16, stop=4096, num=9)
        for param, value in params.items():
            if param == 'max_depth': params[param] = int(value)
            if param == 'num_leaves': params[param] = int(value)
            if param == 'min_data_in_leaf': params[param] = int(value)
            if param == 'bagging_freq': params[param] = int(value)
            if param == 'n_estimators': params[param] = int(value)
            if param == 'min_child_weight': params[param] = int(value)
            if param == 'hidden_layers': params[param] = int(value)
            if param == 'kernel_size': params[param] = int(value)
            if param == 'model': params[param] = 'LSTM' if value > 0.5 else 'GRU'
            if param == 'optimizer': params[param] = 'adam' if value > 0.5 else 'sgd'
            if (param == 'batch_size' or 
                param.startswith('hidden_size') or 
                param.startswith('num_channel')):
                idx = abs(cat_param_choices - value).argmin()
                params[param] = round(cat_param_choices[idx])
                
        return params
    
    # def train_dl_model(
    #     self, 
    #     alg, 
    #     epochs=200, 
    #     batch_size=128, 
    #     early_stop_count=10, 
    #     early_stop_tol=1e-4,
    #     clip_gradients=False,
    #     save=False,
    #     save_checkpoints=False,
    #     **kwargs
    # ):
    #     """Train deep learning models"""
    #     model = self.models.get_model(
    #         alg, self.n_features, self.n_targets, **kwargs
    #     )
        
    #     X_train, Y_train = self.datasets['Train'], self.labels['Train']
    #     X_valid, Y_valid = self.datasets['Valid'], self.labels['Valid']
    #     if alg == 'NN':
    #         train_dataset = self.transform_to_tensor_dataset(X_train, Y_train)
    #         valid_dataset = self.transform_to_tensor_dataset(X_valid, Y_valid)
    #         def collate_fn(batch):
    #             feats, targets = zip(*batch)
    #             feats, targets = torch.stack(feats), torch.stack(targets)
    #             if model.use_gpu: targets = targets.cuda()
    #             return feats, targets
            
    #     loader_params = dict(batch_size=batch_size, collate_fn=collate_fn)
    #     train_loader = DataLoader(dataset=train_dataset, **loader_params)
    #     valid_loader = DataLoader(dataset=valid_dataset, **loader_params)
        
    #     best_val_loss = prev_val_loss = np.inf
    #     best_model_weights = None
    #     early_stop_counter = 0 
    #     perf = {} # performance scores
        
    #     # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(model.optimizer, T_0=10)
    #     for epoch in range(epochs):
    #         model.train() # activate dropout
    #         train_loss = 0
    #         for i, batch in enumerate(train_loader):
    #             model.optimizer.zero_grad() # clear gradients
    #             if alg in ['RNN', 'TCN']:
    #                 preds, targets, _ = model.predict(
    #                     batch, grad=True, bound_pred=False
    #                 )
    #             elif alg == 'NN':
    #                 feats, targets = batch
    #                 preds = model.predict(feats, grad=True, bound_pred=False)

    #             loss = model.criterion(preds, targets)
    #             loss = loss.mean(dim=0)
    #             train_loss += loss
                
    #             if self.task_type == 'C':
    #                 preds = torch.sigmoid(preds) # bound the model prediction
                
    #             loss = loss.mean()
    #             loss.backward() # back propagation, compute gradients
    #             if clip_gradients: model.clip_gradients()
    #             model.optimizer.step() # apply gradients
    #         # lr_scheduler.step()
            
    #         model.eval() # deactivates dropout
    #         valid_loss = self._validate_dl_model(model, alg, valid_loader)
    #         if model.use_gpu: train_loss = train_loss.cpu().detach()
    #         perf[epoch] = {
    #             'Train Loss': train_loss / (i + 1),
    #             'Valid Loss': valid_loss
    #         }
    #         msg = [f'{k}: {v.mean():.4f}' for k, v in perf[epoch].items()]
    #         logger.info(f"Epoch {epoch}, {(', ').join(msg)}")
            
    #         # save best model so far
    #         cur_val_loss = valid_loss.mean()
    #         if cur_val_loss < best_val_loss:
    #             best_val_loss = cur_val_loss
    #             best_model_weights = model.state_dict()
    #             early_stop_counter = 0
                
    #             if save_checkpoints:
    #                 save_path = f"{self.output_path}/train_perf/{alg}-checkpoint"
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model_state_dict': best_model_weights,
    #                     'optimizer_state_dict': model.optimizer.state_dict()
    #                 }, save_path)

    #         # early stopping
    #         if ((early_stop_counter > early_stop_count) or 
    #             (prev_val_loss - cur_val_loss < early_stop_tol)): 
    #             break
    #         early_stop_counter += 1
    #         prev_val_loss = cur_val_loss
            
    #     if save:
    #         if self.n_targets == 1: alg += f'_{self.target_events[0]}'
    #         save_path = f"{self.output_path}/train_perf"
    #         save_pickle(perf, save_path, f"{alg}_perf")
    #         save_path = f'{self.output_path}/{alg}'
    #         torch.save(best_model_weights, save_path)

    #     model.load_weights(best_model_weights)
    #     return model
    
    # def predict(self, model, split, alg, calibrated=True):
    #     if self.task_type == 'R': calibrated = False

    #     X, Y = self.datasets[split], self.labels[split]
    #     if alg == 'NN': 
    #         pred = self._nn_predict(model, X)
    #     else: 
    #         pred = model.predict(X)
            
    #     # make your life easier by ensuring pred and Y have same data format
    #     pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
            
    #     return pred
    
    # def _nn_predict(self, model, X):
    #     model.eval() # deactivates dropout
    #     pred = model.predict(X, grad=False, bound_pred=True)
    #     if model.use_gpu: pred = pred.cpu()
    #     return pred
    
    # def _validate_dl_model(self, model, alg, loader):
    #     total_loss = 0
    #     for i, batch in enumerate(loader):
    #         if alg in ['RNN', 'TCN']:
    #             preds, targets, _ = model.predict(
    #                 batch, grad=True, bound_pred=False
    #             )
    #         elif alg == 'NN':
    #             feats, targets = batch
    #             preds = model.predict(feats, grad=True, bound_pred=False)
    #         loss = model.criterion(preds, targets)
    #         loss = loss.mean(dim=0)
    #         total_loss += loss
    #     if model.use_gpu: total_loss = total_loss.cpu().detach()
    #     return total_loss / (i + 1)
    
    # def _eval_func(self, alg, split='Valid', **kwargs):
    #     """Evaluation function for bayesian optimization
        
    #     Returns:
    #         Either the mean (macro-mean) of 
    #             1. auroc scores
    #             2. root mean squared error
    #         of all target types
    #     """
    #     kwargs = self.convert_hyperparams(kwargs)
    #     try:
    #         model = self.train_model(alg, calibrate=False, save=False, **kwargs)
    #     except Exception as e:
    #         raise e
    #         logger.warning(e)
    #         return -1e9
    #     pred = self.predict(model, split, alg, calibrated=False)
    #     if pred.isnull().any().any():
    #         # TODO: figure out how to prevent this
    #         logger.warning('Invalid prediction - contains NaNs')
    #         return -1e9
    #     return self.score_func(self.labels[split], pred)
    
    # def convert_hyperparams(self, params):
    #     cat_param_choices = np.geomspace(start=16, stop=4096, num=9)
    #     for param, value in params.items():
    #         if param == 'max_depth': params[param] = int(value)
    #         if param == 'n_estimators': params[param] = int(value)
    #         if param == 'min_child_weight': params[param] = int(value)
    #         if param == 'hidden_layers': params[param] = int(value)
    #         if param == 'kernel_size': params[param] = int(value)
    #         if param == 'model': params[param] = 'LSTM' if value > 0.5 else 'GRU'
    #         if param == 'optimizer': params[param] = 'adam' if value > 0.5 else 'sgd'
    #         if (param == 'batch_size' or 
    #             param.startswith('hidden_size') or 
    #             params.startswith('num_channel')):
    #             idx = abs(cat_param_choices - value).argmin()
    #             params[param] = round(cat_param_choices[idx])
                
    #     return params
    
    # def transform_to_tensor_dataset(self, X, Y):
    #     try:
    #         X = X.astype(float)
    #     except Exception as e:
    #         logger.warning(f'Could not convert to float. Please check your input X')
    #         return None
    #     X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    #     Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)
    #     return TensorDataset(X, Y)
    
