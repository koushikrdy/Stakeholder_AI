"""
Base class and interface for explanations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ExplanationMetadata:
    """Metadata for explanations"""
    
    explanation_type: str
    timestamp: Optional[str] = None
    model_name: Optional[str] = None
    instance_id: Optional[int] = None


class Explanation(ABC):
    """
    Abstract base class for explanations.
    
    All explanation methods must inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        explanation_type: str,
        content: Any,
        metadata: Optional[ExplanationMetadata] = None,
    ):
        """
        Initialize explanation.
        
        Args:
            explanation_type: Type of explanation (SHAP, LIME, etc.)
            content: The actual explanation content
            metadata: Optional metadata about the explanation
        """
        self.explanation_type = explanation_type
        self.content = content
        self.metadata = metadata or ExplanationMetadata(explanation_type)
    
    @abstractmethod
    def get_summary(self) -> str:
        """
        Get a summary of the explanation.
        
        Returns:
            Summary string
        """
        pass
    
    @abstractmethod
    def get_importance_scores(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "type": self.explanation_type,
            "content": self.content,
            "metadata": asdict(self.metadata),
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"{self.explanation_type}(summary={self.get_summary()[:50]}...)"


class SHAPExplanation(Explanation):
    """
    SHAP (SHapley Additive exPlanations) explanation.
    
    Content should be a dictionary with feature names and SHAP values.
    """
    
    def __init__(
        self,
        shap_values: Dict[str, float],
        base_value: float,
        metadata: Optional[ExplanationMetadata] = None,
    ):
        """
        Initialize SHAP explanation.
        
        Args:
            shap_values: Dictionary of feature names to SHAP values
            base_value: Base value for the prediction
            metadata: Optional metadata
        """
        content = {
            "shap_values": shap_values,
            "base_value": base_value,
        }
        super().__init__("SHAP", content, metadata)
        self.shap_values = shap_values
        self.base_value = base_value
    
    def get_summary(self) -> str:
        """Get summary of SHAP explanation"""
        top_features = sorted(
            self.shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        summary = "Top factors: " + ", ".join(
            [f"{name} ({value:.3f})" for name, value in top_features]
        )
        return summary
    
    def get_importance_scores(self) -> Dict[str, float]:
        """Get absolute SHAP values as importance"""
        return {k: abs(v) for k, v in self.shap_values.items()}


class LIMEExplanation(Explanation):
    """
    LIME (Local Interpretable Model-agnostic Explanations) explanation.
    
    Content should be a dictionary with feature names and weights.
    """
    
    def __init__(
        self,
        weights: Dict[str, float],
        prediction: float,
        metadata: Optional[ExplanationMetadata] = None,
    ):
        """
        Initialize LIME explanation.
        
        Args:
            weights: Dictionary of feature names to weights
            prediction: Local prediction value
            metadata: Optional metadata
        """
        content = {
            "weights": weights,
            "prediction": prediction,
        }
        super().__init__("LIME", content, metadata)
        self.weights = weights
        self.prediction = prediction
    
    def get_summary(self) -> str:
        """Get summary of LIME explanation"""
        top_features = sorted(
            self.weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        summary = "Key features: " + ", ".join(
            [f"{name} ({weight:.3f})" for name, weight in top_features]
        )
        return summary
    
    def get_importance_scores(self) -> Dict[str, float]:
        """Get absolute weights as importance"""
        return {k: abs(v) for k, v in self.weights.items()}


class CounterfactualExplanation(Explanation):
    """
    Counterfactual explanation showing how to change input to alter prediction.
    """
    
    def __init__(
        self,
        original_instance: Dict[str, float],
        counterfactual_instance: Dict[str, float],
        changes: Dict[str, tuple],
        metadata: Optional[ExplanationMetadata] = None,
    ):
        """
        Initialize counterfactual explanation.
        
        Args:
            original_instance: Original input values
            counterfactual_instance: Modified input values
            changes: Dictionary mapping feature names to (original, modified) tuples
            metadata: Optional metadata
        """
        content = {
            "original": original_instance,
            "counterfactual": counterfactual_instance,
            "changes": changes,
        }
        super().__init__("Counterfactual", content, metadata)
        self.original = original_instance
        self.counterfactual = counterfactual_instance
        self.changes = changes
    
    def get_summary(self) -> str:
        """Get summary of counterfactual explanation"""
        if not self.changes:
            return "No changes needed"
        changes_str = ", ".join(
            [f"{name}: {old} → {new}" for name, (old, new) in self.changes.items()]
        )
        return f"Change: {changes_str}"
    
    def get_importance_scores(self) -> Dict[str, float]:
        """
        Get importance scores based on magnitude of change.
        """
        scores = {}
        for name, (original, modified) in self.changes.items():
            scores[name] = abs(float(modified) - float(original))
        return scores


class RuleBasedExplanation(Explanation):
    """
    Rule-based explanation using decision rules.
    """
    
    def __init__(
        self,
        rules: List[str],
        rule_weights: Dict[str, float],
        prediction_class: str,
        metadata: Optional[ExplanationMetadata] = None,
    ):
        """
        Initialize rule-based explanation.
        
        Args:
            rules: List of decision rules
            rule_weights: Dictionary mapping rules to weights
            prediction_class: The predicted class
            metadata: Optional metadata
        """
        content = {
            "rules": rules,
            "weights": rule_weights,
            "predicted_class": prediction_class,
        }
        super().__init__("Rule-based", content, metadata)
        self.rules = rules
        self.rule_weights = rule_weights
        self.prediction_class = prediction_class
    
    def get_summary(self) -> str:
        """Get summary of rule-based explanation"""
        top_rules = sorted(
            self.rule_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:2]
        summary = "Decision rules: " + "; ".join(
            [f"{rule} ({weight:.3f})" for rule, weight in top_rules]
        )
        return summary
    
    def get_importance_scores(self) -> Dict[str, float]:
        """Get rule weights as importance"""
        return self.rule_weights.copy()
