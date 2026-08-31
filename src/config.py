"""
Configuration and constants for XAI research project
"""

import os
from typing import Dict, List

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset configuration
DATASET_SOURCE = "heart_disease"  # Can be: heart_disease, sepsis, diabetes
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Model configurations
MODELS_TO_TRAIN = ["logistic_regression", "random_forest", "xgboost"]
MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    },
}

# Explanation methods
EXPLANATION_METHODS = ["shap", "lime", "counterfactual", "rule_based"]

# Stakeholders
STAKEHOLDERS = ["doctor", "patient", "regulator"]

# Utility weights (as defined in the research paper)
UTILITY_WEIGHTS = {
    "doctor": {
        "actionability": 0.4,
        "clinical_relevance": 0.3,
        "fidelity": 0.2,
        "cognitive_load": 0.1,
    },
    "patient": {
        "interpretability": 0.4,
        "actionability": 0.3,
        "trust": 0.2,
        "jargon": 0.1,
    },
    "regulator": {
        "fairness": 0.4,
        "auditability": 0.3,
        "transparency": 0.3,
    },
}

# Evaluation metrics
EVALUATION_METRICS = ["accuracy", "f1", "auc", "precision", "recall"]

# Number of instances to generate explanations for
NUM_INSTANCES_FOR_EXPLANATION = 10

print(f"Configuration loaded from {os.path.abspath(__file__)}")
