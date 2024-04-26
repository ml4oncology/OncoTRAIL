# hyperparameter tuning

startTestDate = '2015-01-01'

bayesopt_param = {
    'LR': {'init_points': 10, 'n_iter': 25}, 
    'XGB': {'init_points': 100, 'n_iter': 100},
    'LGBM': {'init_points': 250, 'n_iter': 250},
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
    }
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
    }
}