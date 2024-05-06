from models import LR, XGB, LGBM, MLP
# hyperparameter tuning

startTestDate = '2015-01-01'

algs = {
    'LR': LR,
    'XGB': XGB,
    'LGBM': LGBM,
    'MLP': MLP
    }

# bayesopt_param = {
#     'LR': {'init_points': 2, 'n_iter': 5}, 
#     'XGB': {'init_points': 2, 'n_iter': 5},
#     'LGBM': {'init_points': 2, 'n_iter': 5},
#     'MLP': {'init_points': 2, 'n_iter': 5}
# }

bayesopt_param = {
    'LR': {'init_points': 10, 'n_iter': 25}, 
    'XGB': {'init_points': 100, 'n_iter': 100},
    'LGBM': {'init_points': 250, 'n_iter': 250},
    'MLP': {'init_points': 500, 'n_iter': 250}
}

model_static_param = {
    'LR': {
        'penalty': 'l1', 
        'class_weight': 'balanced', 
        'max_iter': 2000,
        'random_state': 42,
        'solver': 'liblinear'
    },
    'XGB': {
        'random_state': 42
    },
    'LGBM': {
        'random_state': 42,
        'verbosity': -1,
        'deterministic': True,
        'force_col_wise': True
    },
    'MLP': {}
}
model_tuning_param = {
    'LR': {
        'C': (0.00001, 1)
    },
    'XGB': {
        'n_estimators': (50, 200),
        'max_depth': (1, 7),
        'learning_rate': (0.01, 0.3),
        'min_split_loss': (0, 0.5),
        'min_child_weight': (6, 100),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1000)
    },
    'LGBM': {
        'n_estimators': (20, 200),
        'max_depth': (1, 7),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (5, 40),
        'min_data_in_leaf': (6, 30),
        'feature_fraction': (0.5, 1),
        'bagging_fraction': (0.5, 1),
        'bagging_freq': (0, 10),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1000)
    },
    'MLP': {
        'batch_size': (64, 4096),
        'hidden_size1': (16, 4000),
        'hidden_size2': (16, 4000),
        'dropout': (0, 0.5),
        'optimizer': (0, 1),
        'learning_rate': (0.0001, 0.1),
        'weight_decay': (0.0001, 1),
        'momentum': (0, 0.9),
    }
}