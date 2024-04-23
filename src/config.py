# hyperparameter tuning

bayesopt_param = {
    'LR': {'init_points': 2, 'n_iter': 10}, 
    'XGB': {'init_points': 15, 'n_iter': 100},
    'LGBM': {'init_points': 20, 'n_iter': 200},
}
model_static_param = {
    'LR': {
        'penalty': 'l2', 
        'class_weight': 'balanced', 
        'max_iter': 2000,
        'random_state': 42
    },
    'XGB': {
        'random_state': 42
    },
    'LGBM': {
        'random_state': 42,
        'verbosity': -1
    }
}
model_tuning_param = {
    'LR': {
        'C': (0.0001, 1)
    },
    'XGB': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_split_loss': (0, 0.5),
        'min_child_weight': (6, 100),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1000)
    },
    'LGBM': {
        'n_estimators': (50, 200),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (20, 40),
        'min_data_in_leaf': (6, 30),
        'feature_fraction': (0.5, 1),
        'bagging_fraction': (0.5, 1),
        'bagging_freq': (0, 10),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1000)
    }
}