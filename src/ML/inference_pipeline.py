import pickle
import numpy as np
import pandas as pd
import torch
import logging
import os
from ml_common.util import load_pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelInference:
    """Handles model inference on new data"""
    
    def __init__(self, model_file, preprocessing_file):
        """
        Initialize inference pipeline
        
        Args:
            model_file: File path of the saved model
            preprocessing_file: File path of the saved preprocessing artifacts
        """
        self.preprocessing_file = preprocessing_file
        self.model_file = model_file
        self.model = None
        self.preprocessor = None
        
    def load_model_and_preprocessor(self):
        """Load trained model and preprocessing artifacts"""
        # Load model
        try:
            directory, filename = os.path.split(self.model_file)
            self.model = load_pickle(directory, filename)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise ValueError(f"Could not load model: {e}")
        
        # Load preprocessing artifacts
        try:
            from oncotrail.ML.data_pipeline import DataPreprocessor  # Import here to avoid circular imports
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessing_artifacts(self.preprocessing_file)
            logger.info("Preprocessing artifacts loaded successfully")
        except Exception as e:
            raise ValueError(f"Could not load preprocessing artifacts: {e}")
    
    def predict(self, df, embedding=None, data_type="notes"):
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with new data (same format as training data)
            embedding: Embedding matrix for notes (if data_type includes notes)
            data_type: Type of data ("notes", "tabular", "notes-tabular", "nlp", "nlp-tfidf")
        
        Returns:
            predictions: Array of predictions
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first. Call load_model_and_preprocessor().")
        
        # Prepare data for inference
        X = self.preprocessor.prepare_inference_data(df, embedding, data_type)
        
        # Make predictions based on model type
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
        else:
            raise ValueError("Unknown model type")
        
        return predictions
    
    
