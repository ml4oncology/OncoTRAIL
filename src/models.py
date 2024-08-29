from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
        return self.model.predict_proba(X)[:, 1]

    def fit(self, X, Y, eval_set=None):
        if eval_set is None:
            self.model.fit(X, Y)
        else:
            self.model.fit(X, Y, eval_set=eval_set)


class LR(MLModel):
    def __init__(
        self,
        penalty="l1",
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        C=0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        params = {
            "penalty": penalty,
            "class_weight": class_weight,
            "max_iter": max_iter,
            "solver": solver,
            "C": C,
            "random_state": self.random_state,
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
        early_stopping_rounds=25,
        eval_set=[(None, None)],
        **kwargs,
    ):
        super().__init__(**kwargs)
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_split_loss": min_split_loss,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "verbosity": verbosity,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_set": eval_set,
            "random_state": self.random_state,
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
        early_stopping_rounds=25,
        eval_set=[(None, None)],
        **kwargs,
    ):
        super().__init__(**kwargs)
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "verbosity": verbosity,
            "deterministic": deterministic,
            "force_col_wise": force_col_wise,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_set": eval_set,
            "random_state": self.random_state,
        }
        model = LGBMClassifier(**params)
        self.model = model


###############################################################################
# Deep Learning Models
###############################################################################
class DLModel:
    def __init__(self):
        # NOTE: BCEWithLogitLoss COMBINES Sigmoid and BCELoss. This is more
        # numerically stable than using Sigmoid followed by BCELoss. As a
        # result, the model output during training should NOT use a Simgoid
        # layer at the end
        self.loss_type = nn.BCEWithLogitsLoss
        self.use_gpu = torch.cuda.is_available()
        self.model = None

    def eval(self):
        # enters evaluation mode - deactivates dropout
        self.model.eval()

    def train(self):
        # enters training mode - activates dropout
        self.model.train()

    def state_dict(self):
        # return the model weights
        return self.model.state_dict()

    def load_weights(self, state_dict):
        if isinstance(state_dict, str):
            map_location = None if self.use_gpu else torch.device("cpu")
            state_dict = torch.load(state_dict, map_location=map_location)
        self.model.load_state_dict(state_dict)

    def clip_gradients(self, clip_value=1):
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

    def predict(self, X, grad=False, bound_pred=True):
        """
        Args:
            grad (bool): If True, enable gradients for backward pass
            bound_pred (bool): If True, bound the predictions by using Sigmoid
                over the model output
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X.astype(float))

        if self.use_gpu:
            X = X.cuda()

        with torch.set_grad_enabled(grad):
            # NOTE: need to call .detach() is True
            pred = self.model(X)

        if bound_pred:
            pred = torch.sigmoid(pred)

        return pred


class MLP(DLModel):
    def __init__(
        self,
        n_features,
        n_targets,
        hidden_size1=128,
        hidden_size2=64,
        hidden_size3=32,
        three_layers=0,
        dropout=0,
        optimizer="adam",
        learning_rate=1e-2,
        weight_decay=0,
        momentum=0,
        beta1=0.9,
        beta2=0.999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        optimizer_types = {"adam": optim.Adam, "sgd": optim.SGD}
        params = {
            "input_size": n_features,
            "output_size": n_targets,
            "hidden_size1": hidden_size1,
            "hidden_size2": hidden_size2,
            "hidden_size3": hidden_size3,
            "three_layers": three_layers,
            "dropout": dropout,
        }
        self.model = NN(**params)
        self.criterion = self.loss_type(reduction="none")
        if optimizer == "sgd":
            params = {"momentum": momentum}
        elif optimizer == "adam":
            params = {"betas": (beta1, beta2)}
        self.optimizer = optimizer_types[optimizer](
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **params,
        )
        if self.use_gpu:
            self.model.cuda()


class NN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        three_layers,
        dropout=0,
    ):
        super(NN, self).__init__()
        self.three_layers = three_layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        if self.three_layers:
            self.layer3 = nn.Linear(hidden_size2, hidden_size3)
            self.output = nn.Linear(hidden_size3, output_size)
        else:
            self.output = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.dropout(self.relu(self.layer1(X)))
        X = self.dropout(self.relu(self.layer2(X)))
        if self.three_layers:
            X = self.dropout(self.relu(self.layer3(X)))
        return self.output(X)


class MidfusionMLP(DLModel):
    def __init__(
        self,
        n_features,
        embedding_size,
        n_targets,
        fusion_size=128,
        hidden_size1=128,
        hidden_size2=64,
        hidden_size3=32,
        three_layers=0,
        dropout=0,
        batchnorm=0,
        optimizer="adam",
        learning_rate=1e-2,
        weight_decay=0,
        momentum=0,
        beta1=0.9,
        beta2=0.999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        optimizer_types = {"adam": optim.Adam, "sgd": optim.SGD}
        params = {
            "input_size": n_features,
            "embedding_size": embedding_size,
            "fusion_size": fusion_size,
            "output_size": n_targets,
            "hidden_size1": hidden_size1,
            "hidden_size2": hidden_size2,
            "hidden_size3": hidden_size3,
            "three_layers": three_layers,
            "dropout": dropout,
            "batchnorm": batchnorm,
        }
        self.model = NNFusion(**params)
        self.criterion = self.loss_type(reduction="none")
        if optimizer == "sgd":
            params = {"momentum": momentum}
        elif optimizer == "adam":
            params = {"betas": (beta1, beta2)}
        self.optimizer = optimizer_types[optimizer](
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **params,
        )
        if self.use_gpu:
            self.model.cuda()


class NNFusion(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        fusion_size,
        output_size,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        three_layers,
        dropout=0,
        batchnorm=0,
    ):
        super(NNFusion, self).__init__()
        self.three_layers = three_layers
        self.batchnorm = batchnorm
        self.embedding_size = embedding_size
        self.fusion_layer1 = nn.Linear(embedding_size, fusion_size)
        self.fusion_layer2 = nn.Linear(input_size - embedding_size, fusion_size)
        self.layer1 = nn.Linear(2 * fusion_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        if self.batchnorm == 1:
            self.bn1 = nn.BatchNorm1d(hidden_size1)
            self.bn2 = nn.BatchNorm1d(hidden_size2)
        if self.three_layers:
            self.layer3 = nn.Linear(hidden_size2, hidden_size3)
            self.output = nn.Linear(hidden_size3, output_size)
            if self.batchnorm == 1:
                self.bn3 = nn.BatchNorm1d(hidden_size3)
        else:
            self.output = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # split the input into two parts
        notes = X[:, : self.embedding_size]
        tabular = X[:, self.embedding_size :]
        notes_embedding = self.dropout(self.relu(self.fusion_layer1(notes)))
        tabular_embedding = self.dropout(self.relu(self.fusion_layer2(tabular)))
        X = torch.cat((notes_embedding, tabular_embedding), dim=1)
        if self.batchnorm == 0:
            X = self.dropout(self.relu(self.layer1(X)))
            X = self.dropout(self.relu(self.layer2(X)))
        else:
            X = self.dropout(self.relu(self.bn1(self.layer1(X))))
            X = self.dropout(self.relu(self.bn2(self.layer2(X))))
        if self.three_layers:
            if self.batchnorm == 0:
                X = self.dropout(self.relu(self.layer3(X)))
            else:
                X = self.dropout(self.relu(self.bn3(self.layer3(X))))
        return self.output(X)
