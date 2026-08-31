"""
Utility functions for XAI research project
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_model(model: Any, path: str) -> None:
    """
    Save model to disk using pickle.
    
    Args:
        model: Model object to save
        path: Path to save model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """
    Load model from disk.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model object
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model


def save_json(data: Dict, path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {path}")


def load_json(path: str) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics["auc"] = auc(fpr, tpr)
    
    return metrics


def normalize_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize values to [0, 1] range.
    
    Args:
        values: Array of values
        
    Returns:
        Normalized array
    """
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.ones_like(values)
    return (values - min_val) / (max_val - min_val)


def create_explanation_report(
    instance: Dict,
    prediction: float,
    explanations: Dict[str, Any],
    selected_explanation: str,
    utilities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Create a comprehensive explanation report.
    
    Args:
        instance: Input instance
        prediction: Model prediction
        explanations: Dictionary of explanations
        selected_explanation: Name of selected explanation
        utilities: Utility scores for each explanation
        
    Returns:
        Report dictionary
    """
    return {
        "instance": instance,
        "prediction": float(prediction),
        "explanations": explanations,
        "selected_explanation": selected_explanation,
        "utilities": utilities,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
