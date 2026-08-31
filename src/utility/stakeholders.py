"""
Stakeholder utility framework for explanation selection
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from src.explanations.base import Explanation


class StakeholderUtility(ABC):
    """
    Abstract base class for stakeholder utility functions.
    
    Represents: U_s(E | x, ŷ, a) - Utility of explanation E for stakeholder s
    """
    
    def __init__(self, name: str, weights: Dict[str, float]):
        """
        Initialize stakeholder utility.
        
        Args:
            name: Stakeholder name
            weights: Dictionary of component weights
        """
        self.name = name
        self.weights = weights
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
    
    @abstractmethod
    def _compute_component_scores(
        self, explanation: Explanation, prediction: float
    ) -> Dict[str, float]:
        """
        Compute individual utility component scores.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Dictionary mapping component names to scores [0, 1]
        """
        pass
    
    def compute_utility(
        self, explanation: Explanation, prediction: float
    ) -> float:
        """
        Compute total utility as weighted sum of components.
        
        Formula: U_s(E) = Σ w_i * score_i
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Utility score [0, 1]
        """
        component_scores = self._compute_component_scores(explanation, prediction)
        
        total_utility = sum(
            self.weights.get(component, 0) * score
            for component, score in component_scores.items()
        )
        
        return float(np.clip(total_utility, 0, 1))
    
    def get_component_utilities(
        self, explanation: Explanation, prediction: float
    ) -> Dict[str, float]:
        """
        Get utility scores for each component.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Dictionary mapping components to utility scores
        """
        return self._compute_component_scores(explanation, prediction)


class DoctorUtility(StakeholderUtility):
    """
    Doctor utility function.
    
    Weights:
    - Actionability: 0.4 (can the doctor act on this?)
    - Clinical Relevance: 0.3 (is it medically relevant?)
    - Fidelity: 0.2 (does it match the model?)
    - Cognitive Load: 0.1 (complexity)
    """
    
    def __init__(self):
        """Initialize doctor utility"""
        weights = {
            "actionability": 0.4,
            "clinical_relevance": 0.3,
            "fidelity": 0.2,
            "cognitive_load": 0.1,
        }
        super().__init__("doctor", weights)
    
    def _compute_component_scores(
        self, explanation: Explanation, prediction: float
    ) -> Dict[str, float]:
        """
        Compute doctor utility components.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Dictionary of component scores
        """
        scores = {}
        
        # Actionability: high for interpretable explanations
        # SHAP and Counterfactual are most actionable
        if explanation.explanation_type == "Counterfactual":
            scores["actionability"] = 0.9
        elif explanation.explanation_type == "SHAP":
            scores["actionability"] = 0.8
        elif explanation.explanation_type == "Rule-based":
            scores["actionability"] = 0.7
        else:  # LIME
            scores["actionability"] = 0.6
        
        # Clinical Relevance: based on feature count
        # Fewer features = higher clinical relevance
        num_features = len(explanation.get_importance_scores())
        scores["clinical_relevance"] = max(0.5, 1.0 - (num_features / 20.0))
        
        # Fidelity: higher prediction confidence = higher fidelity
        scores["fidelity"] = abs(prediction - 0.5) * 2  # Closer to 1 or 0
        
        # Cognitive Load: inverse of complexity
        # Simpler explanations have lower cognitive load
        if explanation.explanation_type == "Rule-based":
            scores["cognitive_load"] = 0.9
        elif explanation.explanation_type == "Counterfactual":
            scores["cognitive_load"] = 0.8
        else:
            scores["cognitive_load"] = 0.6
        
        return scores


class PatientUtility(StakeholderUtility):
    """
    Patient utility function.
    
    Weights:
    - Interpretability: 0.4 (understandable?)
    - Actionability: 0.3 (can patient act?)
    - Trust: 0.2 (does it build trust?)
    - Jargon: 0.1 (technical language level)
    """
    
    def __init__(self):
        """Initialize patient utility"""
        weights = {
            "interpretability": 0.4,
            "actionability": 0.3,
            "trust": 0.2,
            "jargon": 0.1,
        }
        super().__init__("patient", weights)
    
    def _compute_component_scores(
        self, explanation: Explanation, prediction: float
    ) -> Dict[str, float]:
        """
        Compute patient utility components.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Dictionary of component scores
        """
        scores = {}
        
        # Interpretability: patients prefer simple rules and counterfactuals
        if explanation.explanation_type == "Rule-based":
            scores["interpretability"] = 0.9
        elif explanation.explanation_type == "Counterfactual":
            scores["interpretability"] = 0.8
        elif explanation.explanation_type == "LIME":
            scores["interpretability"] = 0.6
        else:  # SHAP
            scores["interpretability"] = 0.5
        
        # Actionability: rules and counterfactuals are most actionable
        if explanation.explanation_type in ["Rule-based", "Counterfactual"]:
            scores["actionability"] = 0.8
        elif explanation.explanation_type == "LIME":
            scores["actionability"] = 0.6
        else:
            scores["actionability"] = 0.5
        
        # Trust: moderate-to-high predictions build more trust
        scores["trust"] = min(abs(prediction - 0.5) * 2.2, 1.0)
        
        # Jargon: non-technical explanations score higher
        if explanation.explanation_type == "Rule-based":
            scores["jargon"] = 0.9
        elif explanation.explanation_type == "Counterfactual":
            scores["jargon"] = 0.8
        else:
            scores["jargon"] = 0.5
        
        return scores


class RegulatorUtility(StakeholderUtility):
    """
    Regulator utility function.
    
    Weights:
    - Fairness: 0.4 (is model fair?)
    - Auditability: 0.3 (can we audit it?)
    - Transparency: 0.3 (is it transparent?)
    """
    
    def __init__(self):
        """Initialize regulator utility"""
        weights = {
            "fairness": 0.4,
            "auditability": 0.3,
            "transparency": 0.3,
        }
        super().__init__("regulator", weights)
    
    def _compute_component_scores(
        self, explanation: Explanation, prediction: float
    ) -> Dict[str, float]:
        """
        Compute regulator utility components.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            
        Returns:
            Dictionary of component scores
        """
        scores = {}
        
        # Fairness: based on feature diversity
        # More diverse features suggest less bias
        num_features = len(explanation.get_importance_scores())
        scores["fairness"] = min(num_features / 15.0, 1.0)
        
        # Auditability: SHAP and Rule-based are most auditable
        if explanation.explanation_type in ["SHAP", "Rule-based"]:
            scores["auditability"] = 0.9
        elif explanation.explanation_type == "Counterfactual":
            scores["auditability"] = 0.7
        else:  # LIME
            scores["auditability"] = 0.6
        
        # Transparency: SHAP and Rule-based are most transparent
        if explanation.explanation_type in ["SHAP", "Rule-based"]:
            scores["transparency"] = 0.9
        elif explanation.explanation_type == "Counterfactual":
            scores["transparency"] = 0.7
        else:  # LIME
            scores["transparency"] = 0.6
        
        return scores


class UtilityFramework:
    """
    Framework for computing stakeholder utilities.
    """
    
    def __init__(self):
        """Initialize utility framework with all stakeholders"""
        self.stakeholders = {
            "doctor": DoctorUtility(),
            "patient": PatientUtility(),
            "regulator": RegulatorUtility(),
        }
    
    def compute_all_utilities(
        self,
        explanation: Explanation,
        prediction: float,
        stakeholders: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute utilities for all or specified stakeholders.
        
        Args:
            explanation: Explanation object
            prediction: Model prediction value
            stakeholders: List of stakeholder names (None = all)
            
        Returns:
            Dictionary mapping stakeholder names to utility scores
        """
        if stakeholders is None:
            stakeholders = list(self.stakeholders.keys())
        
        utilities = {}
        for stakeholder in stakeholders:
            if stakeholder in self.stakeholders:
                utilities[stakeholder] = self.stakeholders[stakeholder].compute_utility(
                    explanation, prediction
                )
        
        return utilities
    
    def get_best_explanation(
        self,
        explanations: List[Explanation],
        prediction: float,
        stakeholder: str,
    ) -> tuple:
        """
        Select best explanation for stakeholder.
        
        Args:
            explanations: List of explanation candidates
            prediction: Model prediction value
            stakeholder: Stakeholder name
            
        Returns:
            Tuple of (best_explanation, utility_scores)
        """
        if stakeholder not in self.stakeholders:
            raise ValueError(f"Unknown stakeholder: {stakeholder}")
        
        utility_func = self.stakeholders[stakeholder]
        utilities = [
            utility_func.compute_utility(expl, prediction) for expl in explanations
        ]
        
        best_idx = np.argmax(utilities)
        
        return explanations[best_idx], dict(
            zip([e.explanation_type for e in explanations], utilities)
        )
