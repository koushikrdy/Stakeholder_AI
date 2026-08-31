"""
Explanation selection algorithm based on stakeholder utility

Core research contribution: 
E*_s = argmax_{E ∈ ℰ} U_s(E | x, ŷ, a)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.explanations.base import Explanation
from src.utility.stakeholders import UtilityFramework


class ExplanationSelector:
    """
    Selects optimal explanations based on stakeholder utility.
    
    Implements the core formula:
    E*_s = argmax_{E ∈ ℰ} U_s(E | x, ŷ, a)
    
    where:
    - E*_s: optimal explanation for stakeholder s
    - ℰ: set of available explanations
    - U_s: utility function for stakeholder s
    - x: instance
    - ŷ: prediction
    - a: downstream action
    """
    
    def __init__(self, utility_framework: Optional[UtilityFramework] = None):
        """
        Initialize explanation selector.
        
        Args:
            utility_framework: Utility framework (creates default if None)
        """
        self.utility_framework = utility_framework or UtilityFramework()
        self.selection_history = []
    
    def select(
        self,
        explanations: List[Explanation],
        prediction: float,
        stakeholder: str,
    ) -> Tuple[Explanation, Dict[str, float], Dict[str, float]]:
        """
        Select best explanation for stakeholder.
        
        Implementation of: E*_s = argmax_{E ∈ ℰ} U_s(E | x, ŷ, a)
        
        Args:
            explanations: List of candidate explanations
            prediction: Model prediction value
            stakeholder: Stakeholder name
            
        Returns:
            Tuple of:
            - selected_explanation: Best explanation for stakeholder
            - utility_scores: Dict mapping explanation types to utility scores
            - component_scores: Detailed utility component breakdown for best explanation
        """
        if not explanations:
            raise ValueError("No explanations provided")
        
        if stakeholder not in self.utility_framework.stakeholders:
            raise ValueError(f"Unknown stakeholder: {stakeholder}")
        
        utility_func = self.utility_framework.stakeholders[stakeholder]
        
        # Compute utility for each explanation
        utility_scores = {}
        component_scores_dict = {}
        
        for explanation in explanations:
            utility = utility_func.compute_utility(explanation, prediction)
            utility_scores[explanation.explanation_type] = utility
            component_scores_dict[explanation.explanation_type] = (
                utility_func.get_component_utilities(explanation, prediction)
            )
        
        # Select explanation with highest utility (argmax)
        best_explanation_type = max(
            utility_scores, key=utility_scores.get
        )
        best_idx = next(
            i for i, e in enumerate(explanations)
            if e.explanation_type == best_explanation_type
        )
        selected_explanation = explanations[best_idx]
        
        # Get component scores for selected explanation
        component_scores = component_scores_dict[best_explanation_type]
        
        # Record selection
        self.selection_history.append({
            "stakeholder": stakeholder,
            "selected": best_explanation_type,
            "utilities": utility_scores.copy(),
            "components": component_scores.copy(),
        })
        
        return selected_explanation, utility_scores, component_scores
    
    def select_for_multiple_stakeholders(
        self,
        explanations: List[Explanation],
        prediction: float,
        stakeholders: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Explanation, Dict[str, float]]]:
        """
        Select best explanation for multiple stakeholders.
        
        Args:
            explanations: List of candidate explanations
            prediction: Model prediction value
            stakeholders: List of stakeholder names (None = all)
            
        Returns:
            Dictionary mapping stakeholder to (explanation, utilities)
        """
        if stakeholders is None:
            stakeholders = list(self.utility_framework.stakeholders.keys())
        
        results = {}
        for stakeholder in stakeholders:
            explanation, utilities, _ = self.select(
                explanations, prediction, stakeholder
            )
            results[stakeholder] = (explanation, utilities)
        
        return results
    
    def get_selection_summary(self) -> Dict:
        """
        Get summary of all selections made.
        
        Returns:
            Summary dictionary
        """
        if not self.selection_history:
            return {"total_selections": 0}
        
        summary = {
            "total_selections": len(self.selection_history),
            "by_stakeholder": {},
        }
        
        for entry in self.selection_history:
            stakeholder = entry["stakeholder"]
            if stakeholder not in summary["by_stakeholder"]:
                summary["by_stakeholder"][stakeholder] = {
                    "count": 0,
                    "selections": {},
                }
            
            summary["by_stakeholder"][stakeholder]["count"] += 1
            selected = entry["selected"]
            if selected not in summary["by_stakeholder"][stakeholder]["selections"]:
                summary["by_stakeholder"][stakeholder]["selections"][selected] = 0
            summary["by_stakeholder"][stakeholder]["selections"][selected] += 1
        
        return summary
    
    def clear_history(self) -> None:
        """Clear selection history"""
        self.selection_history = []


class SelectionAnalyzer:
    """
    Analyzes explanation selections and generates reports.
    """
    
    @staticmethod
    def compare_selections(
        selections: Dict[str, Tuple[Explanation, Dict[str, float]]],
    ) -> Dict:
        """
        Compare selections across stakeholders.
        
        Shows that E*_doctor ≠ E*_patient ≠ E*_regulator when utilities differ.
        
        Args:
            selections: Dictionary of stakeholder to (explanation, utilities)
            
        Returns:
            Analysis report
        """
        report = {
            "num_stakeholders": len(selections),
            "selected_explanations": {},
            "utilities": {},
            "divergence": {},
        }
        
        # Get unique explanations selected
        selected_types = set()
        for stakeholder, (explanation, _) in selections.items():
            report["selected_explanations"][stakeholder] = (
                explanation.explanation_type
            )
            selected_types.add(explanation.explanation_type)
        
        # Check if all stakeholders selected same explanation
        report["all_same"] = len(selected_types) == 1
        
        # Get utility scores
        for stakeholder, (_, utilities) in selections.items():
            report["utilities"][stakeholder] = utilities
        
        # Compute divergence metrics
        for explanation_type in selected_types:
            divergence = 0
            count = 0
            for stakeholder, (_, utilities) in selections.items():
                if explanation_type in utilities:
                    divergence += utilities[explanation_type]
                    count += 1
            if count > 0:
                report["divergence"][explanation_type] = divergence / count
        
        return report
    
    @staticmethod
    def generate_explanation_report(
        stakeholder: str,
        explanation: Explanation,
        prediction: float,
        utility_scores: Dict[str, float],
        component_scores: Dict[str, float],
    ) -> str:
        """
        Generate human-readable report for explanation.
        
        Args:
            stakeholder: Stakeholder name
            explanation: Selected explanation
            prediction: Model prediction
            utility_scores: Utility scores for all candidates
            component_scores: Component scores for selected explanation
            
        Returns:
            Report string
        """
        report = f"""
EXPLANATION REPORT
==================

Stakeholder: {stakeholder.upper()}
Selected Explanation Type: {explanation.explanation_type}
Model Prediction: {prediction:.3f}

UTILITY COMPARISON
------------------
"""
        
        # Add utility scores
        sorted_utilities = sorted(
            utility_scores.items(), key=lambda x: x[1], reverse=True
        )
        for exp_type, utility in sorted_utilities:
            marker = "✓" if exp_type == explanation.explanation_type else " "
            report += f"{marker} {exp_type}: {utility:.3f}\n"
        
        # Add component breakdown
        report += f"\nCOMPONENT BREAKDOWN ({explanation.explanation_type})\n"
        report += "-" * 40 + "\n"
        for component, score in sorted(
            component_scores.items(), key=lambda x: x[1], reverse=True
        ):
            report += f"{component}: {score:.3f}\n"
        
        # Add explanation summary
        report += f"\nEXPLANATION SUMMARY\n"
        report += "-" * 40 + "\n"
        report += explanation.get_summary() + "\n"
        
        return report
