"""
Model training pipeline
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost is optional
HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    pass

from src.utils import compute_metrics


class ModelTrainer:
    """
    Trains and evaluates classification models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        **kwargs,
    ) -> LogisticRegression:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained model
        """
        model = LogisticRegression(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        **kwargs,
    ) -> RandomForestClassifier:
        """
        Train random forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained model
        """
        model = RandomForestClassifier(random_state=self.random_state, **kwargs)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained model
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        model = xgb.XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            **kwargs
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        
        self.models['xgboost'] = model
        return model
    
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        models: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            models: List of model types to train (None = all)
            
        Returns:
            Dictionary mapping model names to trained models
        """
        if models is None:
            models = ['logistic_regression', 'random_forest']
            if HAS_XGBOOST:
                models.append('xgboost')
        
        trained_models = {}
        
        if 'logistic_regression' in models:
            print("Training Logistic Regression...")
            trained_models['logistic_regression'] = self.train_logistic_regression(
                X_train, y_train,
                max_iter=1000,
            )
        
        if 'random_forest' in models:
            print("Training Random Forest...")
            trained_models['random_forest'] = self.train_random_forest(
                X_train, y_train,
                n_estimators=100,
                n_jobs=-1,
            )
        
        if 'xgboost' in models:
            if HAS_XGBOOST:
                print("Training XGBoost...")
                trained_models['xgboost'] = self.train_xgboost(
                    X_train, y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_estimators=100,
                )
        
        return trained_models
    
    def evaluate(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model_name: Name of model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred.astype(float)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        self.metrics[model_name] = metrics
        
        return metrics
    
    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary mapping model names to metrics
        """
        results = {}
        for model_name in self.models:
            results[model_name] = self.evaluate(model_name, X_test, y_test)
        return results
    
    def get_best_model(self) -> tuple:
        """
        Get model with best validation AUC.
        
        Returns:
            Tuple of (model_name, model, metrics)
        """
        if not self.metrics:
            raise ValueError("No models evaluated yet")
        
        best_model_name = max(
            self.metrics,
            key=lambda x: self.metrics[x].get('auc', self.metrics[x].get('f1', 0)),
        )
        
        return (
            best_model_name,
            self.models[best_model_name],
            self.metrics[best_model_name],
        )
