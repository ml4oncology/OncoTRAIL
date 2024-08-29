# hyperparameter tuning

start_test_date = "2015-01-01"
# bayesopt_param = {
#     'LR': {'init_points': 2, 'n_iter': 5},
#     'XGB': {'init_points': 2, 'n_iter': 5},
#     'LGBM': {'init_points': 2, 'n_iter': 5},
#     'MLP': {'init_points': 2, 'n_iter': 5}
# }

LLM_embedding_dim = {
    "Mistral": 4096,
    "BioMistral": 4096,
    "ClinicalLongformer": 768,
    "Llama3-8B": 4096,
    "Llama3-8B-Instruct": 4096
}

bayesopt_param = {
    "LR": {"init_points": 10, "n_iter": 25},
    "XGB": {"init_points": 100, "n_iter": 100},
    "LGBM": {"init_points": 250, "n_iter": 250},
    "MLP": {"init_points": 500, "n_iter": 250},
    "Midfusion": {"init_points": 750, "n_iter": 250}
}

model_static_param = {
    "LR": {
        "penalty": "l1",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 42,
        "solver": "liblinear",
    },
    "XGB": {"random_state": 42, "early_stopping_rounds": 25},
    "LGBM": {
        "random_state": 42,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "early_stopping_rounds": 25,
    },
    "MLP": {},
    "Midfusion": {}
}

model_tuning_param = {
    "LR": {"C": (0.00001, 1)},
    "XGB": {
        "n_estimators": (50, 200),
        "max_depth": (1, 7),
        "learning_rate": (0.01, 0.3),
        "min_split_loss": (0, 0.5),
        "min_child_weight": (6, 100),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1000),
    },
    "LGBM": {
        "n_estimators": (20, 200),
        "max_depth": (1, 7),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (5, 40),
        "min_data_in_leaf": (6, 30),
        "feature_fraction": (0.5, 1),
        "bagging_fraction": (0.5, 1),
        "bagging_freq": (0, 10),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1000),
    },
    "MLP": {
        "batch_size": (64, 4096),
        "hidden_size1": (16, 4000),
        "hidden_size2": (16, 4000),
        "hidden_size3": (16, 4000),
        "three_layers": (0, 1),
        "dropout": (0, 0.5),
        "optimizer": (0, 1),
        "learning_rate": (0.0001, 0.1),
        "weight_decay": (0.0001, 1),
        "momentum": (0, 0.9),
    },
    "Midfusion": {
        "batch_size": (64, 4096),
        "fusion_size": (16, 2000),
        "hidden_size1": (16, 4000),
        "hidden_size2": (16, 4000),
        "hidden_size3": (16, 4000),
        "three_layers": (0, 1),
        "dropout": (0, 0.5),
        "optimizer": (0, 1),
        "learning_rate": (0.0001, 0.1),
        "weight_decay": (0.0001, 1),
        "momentum": (0, 0.9),
        "batchnorm": (0,1)
    },
}