import logging
import sys
import re
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
# from ml_common.prep import Splitter, PrepData
from make_clinical_dataset.epr.prep import Splitter, PrepData
from oncotrail.ML.nlp import (process_df, 
                                             extract_top_ngrams,
                                             build_tfidf_matrix,
                                             build_count_matrix)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing, splitting, and saves preprocessing artifacts for later inference"""
    
    def __init__(self):
        self.embedding_scaler = None
        self.tabular_prep = None
        self.nlp_vectorizer = None
        self.nlp_vocab = None
        self.nlp_scaler = None
        self.unique_physician_names = None
        self.splits = {}
        self.target_splits = {}
        self.mrn_splits = {}
        self.var_names = None
        self.train_dataframe = None
        
    def prepare_data(self, df, embedding, target, end_devt_date, split_config, 
                    data_type, model_name):
        """
        Main method to prepare and split data
        Returns: Dictionary with all necessary data and saves preprocessing artifacts
        """
        
        # Preprocess notes if needed
        if 'nlp' in data_type:
            df = process_df(df, "note")
        
        # Generate splits
        self._generate_splits(df, end_devt_date, split_config)
        
        # Extract target values
        self._extract_targets(target, model_name)
        
        # Extract MRN values
        self._extract_mrns(model_name)
        
        # Prepare features based on data type
        if data_type in ["notes", "notes-tabular"]:
            self._prepare_embedding_features(embedding, model_name)
        elif data_type == "tabular":
            self._prepare_empty_features(model_name)
        elif 'nlp' in data_type:
            self._prepare_nlp_features(data_type, model_name)
        
        # Scale embedding/NLP features BEFORE adding tabular features
        self._scale_features(data_type, model_name)

        # Add tabular features if needed
        if data_type in ["notes-tabular", "tabular"]:
            self._prepare_tabular_features(model_name)
        
        self.var_names = self._get_variable_names(data_type)

        return {
            'X_train': self.splits['train'],
            'Y_train': self.target_splits['train'],
            'X_eval': self.splits['eval'],
            'Y_eval': self.target_splits['eval'],
            'X_valid': self.splits['valid'],
            'Y_valid': self.target_splits['valid'],
            'X_test': self.splits['test'],
            'Y_test': self.target_splits['test'],
            'var_names': self.var_names,
            'mrn_train': self.mrn_splits['train'],
            'mrn_eval': self.mrn_splits['eval'],
            'mrn_valid': self.mrn_splits['valid'],
            'mrn_test': self.mrn_splits['test']
        }
    
    def save_preprocessing_artifacts(self, save_dir, target_name_nospace, split_config, data_type, setup_str, model_name):
        """Save all preprocessing artifacts needed for inference"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_type = "LR" if model_name == "LR" else "NonLR"
        # Create preprocessing key based only on parameters that affect preprocessing
        preprocessing_key = f"{setup_str}_{target_name_nospace}_{split_config}_{data_type}_{model_type}"
        artifact_file = f"{save_dir}/{preprocessing_key}_preprocessing_artifacts.pkl"
        
        # Only save if this preprocessing configuration hasn't been saved yet
        if not os.path.exists(artifact_file):
            artifacts = {
                'embedding_scaler': self.embedding_scaler,
                'tabular_prep': self.tabular_prep,
                'nlp_vectorizer': self.nlp_vectorizer,
                'nlp_vocab': self.nlp_vocab,
                'nlp_scaler': self.nlp_scaler,
                'unique_physician_names': self.unique_physician_names,
                'var_names': self.var_names
            }
            
            with open(artifact_file, 'wb') as f:
                pickle.dump(artifacts, f)
            logger.info(f"Saved preprocessing artifacts: {artifact_file}")
        else:
            logger.info(f"Preprocessing artifacts already exist: {artifact_file}")
    
    def load_preprocessing_artifacts(self, preprocessing_file_path):
        """Load preprocessing artifacts for inference"""
        # preprocessing_file_path is the full path to the artifact file with pkl extension
        with open(preprocessing_file_path, 'rb') as f:
            artifacts = pickle.load(f)
            
        self.embedding_scaler = artifacts['embedding_scaler']
        self.tabular_prep = artifacts['tabular_prep']
        self.nlp_vectorizer = artifacts['nlp_vectorizer']
        self.nlp_vocab = artifacts['nlp_vocab']
        self.nlp_scaler = artifacts['nlp_scaler']
        self.unique_physician_names = artifacts['unique_physician_names']
        self.var_names = artifacts['var_names']
    
    def _generate_splits(self, df, end_devt_date, split_config):
        """Generate train/eval/valid/test splits"""
        splitter = Splitter()
        
        if split_config == "Temporal":
            train_eval_data, valid_data, test_data = splitter.split_data(df, end_devt_date)
        elif split_config == "Random":
            devt_cohort, test_data = splitter.random_split(df, test_size=0.35)
            train_eval_data, valid_data = splitter.random_split(devt_cohort, test_size=0.2)
        
        train_data, eval_data = splitter.random_split(train_eval_data, test_size=0.15)
        
        self.split_dfs = {
            "train": train_data,
            "eval": eval_data,
            "valid": valid_data,
            "test": test_data
        }
        
        self.index_sets = {
            "train": train_data.index.to_list(),
            "eval": eval_data.index.to_list(),
            "valid": valid_data.index.to_list(),
            "test": test_data.index.to_list(),
        }
    
    def _extract_targets(self, target, model_name):
        """Extract target values for each split"""
        self.target_splits = {k: target[idx] for k, idx in self.index_sets.items()}
        
        # Merge train and eval for LR
        if model_name == "LR":
            self.target_splits["train"] = np.concatenate([
                self.target_splits["train"], 
                self.target_splits["eval"]
            ])
            self.target_splits["eval"] = None
    
    def _extract_mrns(self, model_name):
        """Extract MRN values for each split"""
        if model_name == "LR":
            self.mrn_splits = {
                "train": np.concatenate([
                    self.split_dfs["train"]["mrn"].to_numpy(), 
                    self.split_dfs["eval"]["mrn"].to_numpy()
                ]).astype(int),
                "eval": None,
                "valid": self.split_dfs["valid"]["mrn"].to_numpy().astype(int),
                "test": self.split_dfs["test"]["mrn"].to_numpy().astype(int),
            }
        else:
            self.mrn_splits = {
                "train": self.split_dfs["train"]["mrn"].to_numpy().astype(int),
                "eval": self.split_dfs["eval"]["mrn"].to_numpy().astype(int),
                "valid": self.split_dfs["valid"]["mrn"].to_numpy().astype(int),
                "test": self.split_dfs["test"]["mrn"].to_numpy().astype(int),
            }
    
    def _prepare_embedding_features(self, embedding, model_name):
        """Prepare embedding features"""
        self.splits = {k: embedding[idx, :] for k, idx in self.index_sets.items()}
        
        # Merge train and eval for LR
        if model_name == "LR":
            self.splits["train"] = np.concatenate([
                self.splits["train"], 
                self.splits["eval"]
            ])
            self.splits["eval"] = None
    
    def _prepare_empty_features(self, model_name):
        """Prepare empty features for tabular-only data"""
        self.splits = {k: np.zeros((len(idx), 0)) for k, idx in self.index_sets.items()}
    
        # Merge train and eval for LR
        if model_name == "LR":
            self.splits["train"] = np.concatenate([
                self.splits["train"], 
                self.splits["eval"]
            ])
            self.splits["eval"] = None

    def _prepare_nlp_features(self, data_type, model_name):
        """Prepare NLP features with TF-IDF or count vectorization"""
        # Generate vocabulary from training data
        self.nlp_vocab = extract_top_ngrams(
            self.split_dfs["train"], 
            text_col="note_lemmatized_note", 
            top_k=250
        )
        
        train_data = self.split_dfs["train"]
        if model_name == "LR":
            train_data = pd.concat([self.split_dfs["train"], self.split_dfs["eval"]])
        
        # Build feature matrix
        if 'tfidf' in data_type:
            X_train, self.nlp_vectorizer = build_tfidf_matrix(
                train_data, 
                text_col="note_lemmatized_note", 
                vocabulary=self.nlp_vocab
            )
        else:
            X_train, self.nlp_vectorizer = build_count_matrix(
                train_data, 
                text_col="note_lemmatized_note", 
                vocabulary=self.nlp_vocab
            )
        
        self.splits = {"train": X_train.toarray().astype(float)}
        
        # Transform other splits
        for split in ["eval", "valid", "test"]:
            if split != "eval" or model_name != "LR":
                X_split = self.nlp_vectorizer.transform(
                    self.split_dfs[split]["note_lemmatized_note"]
                ).toarray().astype(float)
                self.splits[split] = X_split
            else:
                self.splits[split] = None
    
    def _prepare_tabular_features(self, model_name):
        """Prepare tabular features"""
        # Initialize tabular preprocessor
        self.tabular_prep = PrepData()
        
        # Handle physician names for notes-tabular
        if hasattr(self, 'splits') and self.splits["train"].shape[1] > 0:  # notes-tabular case
            self.unique_physician_names = self._find_unique_phys(self.split_dfs["train"])
            
            # Add physician features to each split
            for split_name in ["train", "eval", "valid", "test"]:
                if model_name == "LR" and split_name == "eval":
                    continue
                    
                if model_name == "LR" and split_name == "train":
                    df_concat = pd.concat([self.split_dfs["train"], self.split_dfs["eval"]])
                    phys = self._convert_physician_name_tabular(df_concat, self.unique_physician_names)
                else:
                    phys = self._convert_physician_name_tabular(
                        self.split_dfs[split_name], 
                        self.unique_physician_names
                    )
                
                self.splits[split_name] = np.concatenate([self.splits[split_name], phys], axis=1)
        
        # Process tabular data
        data_frames = {}
        # Remove unnecessary columns
        drop_cols = ["mrn", "treatment_date", "stats_physician"]
        data_frames = {
                "train": self.tabular_prep.transform_data(
                    pd.concat([self.split_dfs["train"], self.split_dfs["eval"]]) if model_name == "LR" else self.split_dfs["train"],
                    data_name="training"
                ),
                "eval": None if model_name == "LR" else self.tabular_prep.transform_data(self.split_dfs["eval"], data_name="evaluation"),
                "valid": self.tabular_prep.transform_data(self.split_dfs["valid"], data_name="validation"),
                "test": self.tabular_prep.transform_data(self.split_dfs["test"], data_name="test"),
            }
        
        for k in data_frames:
            if data_frames[k] is not None:
                data_frames[k].drop(columns=drop_cols, inplace=True)

        self.train_dataframe = data_frames["train"]

        for key in ["train", "eval", "valid", "test"]:
            if data_frames[key] is None:
                self.splits[key] = None
            else:
                self.splits[key] = np.concatenate([self.splits[key], data_frames[key].to_numpy()], axis=1)
    
    def _scale_features(self, data_type, model_name):
        """Scale features appropriately"""
        if 'nlp' in data_type or data_type in ["notes", "notes-tabular"]:
            self.embedding_scaler = StandardScaler()
            self.splits["train"] = self.embedding_scaler.fit_transform(self.splits["train"])
            
            if self.splits["eval"] is not None:
                self.splits["eval"] = self.embedding_scaler.transform(self.splits["eval"])
            
            self.splits["valid"] = self.embedding_scaler.transform(self.splits["valid"])
            self.splits["test"] = self.embedding_scaler.transform(self.splits["test"])
    
    def _get_variable_names(self, data_type):
        """Get variable names based on data type"""
        if 'nlp' in data_type:
            return self.nlp_vocab
        else:
            return self.train_dataframe.columns.to_list()
    
    def _find_unique_phys(self, df):
        """Find unique physician names"""
        physician_names = df["stats_physician"].unique()
        unique_physician_names = []
        for elem in physician_names:
            unique_physician_names = unique_physician_names + self._convert_str_list(elem)
        unique_physician_names = list(set(unique_physician_names))
        return np.array(unique_physician_names)
    
    def _convert_str_list(self, y):
        """Convert string representation of list to actual list"""
        match = re.search(r'\[.*?\]', y)
        y = match.group(0)
        words = y.strip("[]").split("'")
        result = [word for word in words if word.strip()]
        return result
    
    def _convert_physician_name_tabular(self, df, unique_phys):
        """Convert physician names to tabular format"""
        physician_names_tabular = []
        physician_names_values = df["stats_physician"].values
        
        for elem in physician_names_values:
            phys_list = np.array(self._convert_str_list(elem))
            temp = [int(phys in phys_list) for phys in unique_phys]
            physician_names_tabular.append(temp)
        
        return np.array(physician_names_tabular)
    
    def prepare_inference_data(self, df, embedding=None, data_type="notes"):
        """
        Prepare new data for inference using saved preprocessing artifacts
        """
        if not self.embedding_scaler and not self.tabular_prep and not self.nlp_vectorizer:
            raise ValueError("No preprocessing artifacts loaded. Call load_preprocessing_artifacts first.")
        
        # Preprocess notes if needed
        if 'nlp' in data_type:
            df = process_df(df, "note")
        
        # Prepare features based on data type
        if data_type in ["notes", "notes-tabular"]:
            if embedding is None:
                raise ValueError("Embedding is required for notes data")
            X = embedding
        elif data_type == "tabular":
            X = np.zeros((len(df), 0))
        elif 'nlp' in data_type:
            if self.nlp_vectorizer is None:
                raise ValueError("NLP vectorizer not found in artifacts")
            X = self.nlp_vectorizer.transform(df["note_lemmatized_note"]).toarray().astype(float)
        
        # Scale features
        if 'nlp' in data_type or data_type in ["notes", "notes-tabular"]:
            if self.embedding_scaler is None:
                raise ValueError("Embedding scaler not found in artifacts")
            X = self.embedding_scaler.transform(X)

        # Add tabular features if needed
        if data_type in ["notes-tabular", "tabular"]:
            if self.tabular_prep is None:
                raise ValueError("Tabular preprocessor not found in artifacts")
            
            # Handle physician names for notes-tabular
            if data_type == "notes-tabular":
                if self.unique_physician_names is None:
                    raise ValueError("Unique physician names not found in artifacts")
                phys = self._convert_physician_name_tabular(df, self.unique_physician_names)
                X = np.concatenate([X, phys], axis=1)
            
            # Process tabular data
            valid_cols = [col for col in df.columns if col in self.var_names]
            df = df[valid_cols]
            df_processed = self.tabular_prep.transform_data(df, data_name="test", one_hot_encode=True)
            # if a column in self.var_names is not in df, add a column of zeros to df
            for col in self.var_names:
                if col not in df_processed.columns:
                    df_processed[col] = np.zeros(len(df_processed))
            # arrange the columns in df to be the same as training data
            df_processed = df_processed[self.var_names]
            drop_cols = ["mrn", "treatment_date", "stats_physician"]
            df_processed.drop(columns=drop_cols, inplace=True, errors='ignore')
            X = np.concatenate([X, df_processed.to_numpy()], axis=1)
        
        return X