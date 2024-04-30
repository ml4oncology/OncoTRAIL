from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

###############################################################################
# Machine Learning Models
###############################################################################
class MLModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
    
    def fit(self, X, Y):
        self.model.fit(X, Y)
    
class LR(MLModel):
    def __init__(self, 
                 penalty='l1', 
                 class_weight='balanced',
                 max_iter=2000,
                 solver='liblinear',
                 C=0.0001,
                 **kwargs):
        super().__init__(**kwargs)
        params = {
            'penalty': penalty,
            'class_weight': class_weight,
            'max_iter': max_iter,
            'solver': solver,
            'C': C,
            'random_state': self.random_state
        }
        model = LogisticRegression(**params)
        self.model = model
    
class XGB(MLModel):
    def __init__(
        self, 
        n_estimators=50, 
        max_depth=7, 
        learning_rate=0.3, 
        min_split_loss=0, 
        min_child_weight=6,
        reg_lambda=1,
        reg_alpha=0,
        verbosity=0,
        **kwargs):
        super().__init__(**kwargs)
        params = {
            'n_estimators': n_estimators, 
            'max_depth': max_depth,
            'learning_rate': learning_rate, 
            'min_split_loss': min_split_loss, 
            'min_child_weight': min_child_weight,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'verbosity': verbosity,
            'random_state': self.random_state
        }
        model = XGBClassifier(**params)
        self.model = model

class LGBM(MLModel):
    def __init__(
        self, 
        n_estimators=20, 
        max_depth=7, 
        learning_rate=0.3, 
        num_leaves=5,
        min_data_in_leaf=6,
        feature_fraction=0.5,
        bagging_fraction=0.5,
        bagging_freq=0,
        reg_lambda=1,
        reg_alpha=0,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **kwargs):
        super().__init__(**kwargs)
        params = {
            'n_estimators': n_estimators, 
            'max_depth': max_depth,
            'learning_rate': learning_rate, 
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'verbosity': verbosity,
            'deterministic': deterministic,
            'force_col_wise': force_col_wise,
            'random_state': self.random_state
        }
        model = LGBMClassifier(**params)
        self.model = model

# methods needed: fit, model.classes_, predict