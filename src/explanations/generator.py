"""
Explanation generation implementations
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .base import (
    CounterfactualExplanation,
    LIMEExplanation,
    RuleBasedExplanation,
    SHAPExplanation,
)


class ExplanationGenerator:
    """
    Base class for generating explanations.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize explanation generator.
        
        Args:
            model: Trained model
            X_train: Training data for reference
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Prediction probabilities
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)


class SHAPExplanationGenerator(ExplanationGenerator):
    """
    SHAP explanation generator using Kernel SHAP approximation.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        num_samples: int = 100,
    ):
        """
        Initialize SHAP generator.
        
        Args:
            model: Trained model
            X_train: Training data
            num_samples: Number of samples for SHAP calculation
        """
        super().__init__(model, X_train)
        self.num_samples = num_samples
    
    def generate(self, x: np.ndarray) -> SHAPExplanation:
        """
        Generate SHAP explanation for instance.
        
        Args:
            x: Instance to explain
            
        Returns:
            SHAPExplanation object
        """
        # Simplified SHAP calculation: use permutation-based approach
        base_value = self.predict_proba(self.X_train).mean()
        
        shap_values = {}
        x_df = pd.DataFrame([x], columns=self.feature_names)
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            # Create perturbed samples
            contributions = []
            for _ in range(self.num_samples):
                # Sample from training data
                sample_idx = np.random.randint(0, len(self.X_train))
                
                # Create two versions: with and without feature
                x_with = x_df.copy()
                x_without = x_df.copy()
                x_without.iloc[0, feature_idx] = self.X_train.iloc[
                    sample_idx, feature_idx
                ]
                
                pred_with = self.predict_proba(x_with)[0]
                pred_without = self.predict_proba(x_without)[0]
                
                contributions.append(pred_with - pred_without)
            
            shap_values[feature_name] = np.mean(contributions)
        
        return SHAPExplanation(shap_values, base_value)


class LIMEExplanationGenerator(ExplanationGenerator):
    """
    LIME explanation generator using local linear approximation.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        num_samples: int = 100,
        kernel_width: float = 0.1,
    ):
        """
        Initialize LIME generator.
        
        Args:
            model: Trained model
            X_train: Training data
            num_samples: Number of samples for local approximation
            kernel_width: Kernel width for local weights
        """
        super().__init__(model, X_train)
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def generate(self, x: np.ndarray) -> LIMEExplanation:
        """
        Generate LIME explanation for instance.
        
        Args:
            x: Instance to explain
            
        Returns:
            LIMEExplanation object
        """
        x_df = pd.DataFrame([x], columns=self.feature_names)
        
        # Generate perturbed samples
        perturbed = np.random.normal(0, 1, (self.num_samples, len(self.feature_names)))
        
        # Scale to original feature ranges
        X_array = self.X_train.values
        feature_mins = X_array.min(axis=0)
        feature_maxs = X_array.max(axis=0)
        feature_ranges = feature_maxs - feature_mins
        
        perturbed = perturbed * feature_ranges + x
        
        # Get predictions for perturbed samples
        perturbed_df = pd.DataFrame(perturbed, columns=self.feature_names)
        perturbed_preds = self.predict_proba(perturbed_df)
        
        # Calculate distances and weights
        distances = np.linalg.norm(perturbed - x, axis=1)
        weights = np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))
        
        # Fit weighted linear model
        # Normalize features for linear model
        perturbed_norm = (perturbed - x) / (feature_ranges + 1e-10)
        
        # Simple weighted linear regression
        weights_matrix = np.diag(weights)
        X_weighted = perturbed_norm.T @ weights_matrix @ perturbed_norm
        y_weighted = perturbed_norm.T @ weights_matrix @ perturbed_preds
        
        try:
            coefficients = np.linalg.solve(X_weighted + 1e-6 * np.eye(len(self.feature_names)), y_weighted)
        except np.linalg.LinAlgError:
            coefficients = np.zeros(len(self.feature_names))
        
        weights_dict = {
            self.feature_names[i]: float(coefficients[i])
            for i in range(len(self.feature_names))
        }
        
        prediction = self.predict_proba(x_df)[0]
        
        return LIMEExplanation(weights_dict, prediction)


class CounterfactualExplanationGenerator(ExplanationGenerator):
    """
    Counterfactual explanation generator (simplified DiCE-like approach).
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize counterfactual generator.
        
        Args:
            model: Trained model
            X_train: Training data
        """
        super().__init__(model, X_train)
        self.feature_stats = self._compute_feature_stats()
    
    def _compute_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute feature statistics"""
        stats = {}
        for col in self.feature_names:
            stats[col] = {
                "mean": self.X_train[col].mean(),
                "std": self.X_train[col].std(),
                "min": self.X_train[col].min(),
                "max": self.X_train[col].max(),
            }
        return stats
    
    def generate(self, x: np.ndarray, target_class: int = 1) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation.
        
        Args:
            x: Instance to explain
            target_class: Target class for counterfactual
            
        Returns:
            CounterfactualExplanation object
        """
        x_df = pd.DataFrame([x], columns=self.feature_names)
        original_pred = self.predict_proba(x_df)[0]
        
        # Generate counterfactuals by iteratively modifying features
        counterfactual = x.copy()
        changes = {}
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            # Try shifting feature towards population mean
            original_val = x[feature_idx]
            feature_mean = self.feature_stats[feature_name]["mean"]
            
            # Create candidate with modified feature
            candidate = counterfactual.copy()
            candidate[feature_idx] = feature_mean
            
            candidate_df = pd.DataFrame([candidate], columns=self.feature_names)
            candidate_pred = self.predict_proba(candidate_df)[0]
            
            # Accept if it improves prediction towards target
            if candidate_pred > original_pred:
                counterfactual = candidate
                changes[feature_name] = (float(original_val), float(feature_mean))
        
        counterfactual_dict = {
            self.feature_names[i]: x[i] for i in range(len(self.feature_names))
        }
        original_dict = {
            self.feature_names[i]: x[i] for i in range(len(self.feature_names))
        }
        
        return CounterfactualExplanation(original_dict, counterfactual_dict, changes)


class RuleBasedExplanationGenerator(ExplanationGenerator):
    """
    Rule-based explanation generator using decision tree rules.
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Initialize rule-based generator.
        
        Args:
            model: Trained model
            X_train: Training data
            y_train: Training labels
        """
        super().__init__(model, X_train)
        self.y_train = y_train
        self._extract_rules()
    
    def _extract_rules(self) -> None:
        """Extract rules from model or create simple decision rules"""
        # Create a simple decision tree for rule extraction
        self.tree_model = RandomForestClassifier(
            n_estimators=1, max_depth=3, random_state=42
        )
        self.tree_model.fit(self.X_train, self.y_train)
    
    def generate(self, x: np.ndarray) -> RuleBasedExplanation:
        """
        Generate rule-based explanation.
        
        Args:
            x: Instance to explain
            
        Returns:
            RuleBasedExplanation object
        """
        # Extract rules that apply to this instance
        rules = []
        rule_weights = {}
        
        # Get decision path through tree
        x_df = pd.DataFrame([x], columns=self.feature_names)
        
        # Simple rule generation based on feature values
        X_array = self.X_train.values
        feature_medians = np.median(X_array, axis=0)
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            median = feature_medians[feature_idx]
            value = x[feature_idx]
            
            if value > median:
                rule = f"{feature_name} > {median:.2f}"
                confidence = np.mean(
                    (X_array[:, feature_idx] > median) == (self.y_train == 1)
                )
            else:
                rule = f"{feature_name} <= {median:.2f}"
                confidence = np.mean(
                    (X_array[:, feature_idx] <= median) == (self.y_train == 1)
                )
            
            rules.append(rule)
            rule_weights[rule] = float(confidence)
        
        pred = self.predict_proba(x_df)[0]
        prediction_class = "Positive" if pred > 0.5 else "Negative"
        
        return RuleBasedExplanation(rules, rule_weights, prediction_class)
