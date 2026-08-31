"""
Experimental evaluation pipeline
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.explanations.generator import (
    CounterfactualExplanationGenerator,
    LIMEExplanationGenerator,
    RuleBasedExplanationGenerator,
    SHAPExplanationGenerator,
)
from src.selector import ExplanationSelector
from src.utility import UtilityFramework


class ExperimentalEvaluator:
    """
    Evaluates explanation selection across instances and stakeholders.
    """
    
    def __init__(self, model: any, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            X_train: Training data
            y_train: Training labels
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        # Initialize explanation generators
        self.explanation_generators = {
            'shap': SHAPExplanationGenerator(model, X_train),
            'lime': LIMEExplanationGenerator(model, X_train),
            'counterfactual': CounterfactualExplanationGenerator(model, X_train),
            'rule_based': RuleBasedExplanationGenerator(model, X_train, y_train),
        }
        
        # Initialize selector and utility framework
        self.utility_framework = UtilityFramework()
        self.selector = ExplanationSelector(self.utility_framework)
        
        self.results = []
    
    def evaluate_instance(
        self,
        instance: np.ndarray,
        stakeholders: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate explanation selection for single instance.
        
        Args:
            instance: Instance to explain
            stakeholders: List of stakeholders (None = all)
            
        Returns:
            Evaluation results
        """
        if stakeholders is None:
            stakeholders = list(self.utility_framework.stakeholders.keys())
        
        # Get prediction
        instance_df = pd.DataFrame([instance], columns=self.X_train.columns)
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(instance_df)[0, 1]
        else:
            prediction = self.model.predict(instance_df)[0]
        
        # Generate all explanations
        explanations = []
        for gen_name, generator in self.explanation_generators.items():
            try:
                explanation = generator.generate(instance)
                explanations.append(explanation)
            except Exception as e:
                print(f"Error generating {gen_name}: {e}")
        
        # Select best explanation for each stakeholder
        selections = self.selector.select_for_multiple_stakeholders(
            explanations, prediction, stakeholders
        )
        
        # Compute utility comparison
        result = {
            'instance': instance,
            'prediction': float(prediction),
            'selections': {},
            'utilities': {},
        }
        
        for stakeholder, (explanation, utilities) in selections.items():
            result['selections'][stakeholder] = explanation.explanation_type
            result['utilities'][stakeholder] = utilities
        
        self.results.append(result)
        return result
    
    def evaluate_batch(
        self,
        X: pd.DataFrame,
        stakeholders: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Evaluate multiple instances.
        
        Args:
            X: Feature data
            stakeholders: List of stakeholders
            limit: Maximum number of instances to evaluate
            
        Returns:
            List of evaluation results
        """
        batch_results = []
        n_instances = min(len(X), limit) if limit else len(X)
        
        for i in range(n_instances):
            print(f"Evaluating instance {i+1}/{n_instances}...", end='\r')
            result = self.evaluate_instance(X.iloc[i].values, stakeholders)
            batch_results.append(result)
        
        print(f"Evaluation complete for {n_instances} instances.          ")
        return batch_results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame for analysis.
        
        Returns:
            DataFrame with results
        """
        rows = []
        for result in self.results:
            for stakeholder, selections in result['selections'].items():
                row = {
                    'stakeholder': stakeholder,
                    'prediction': result['prediction'],
                    'selected_explanation': selections,
                }
                # Add utility scores
                for exp_type, utility_score in result['utilities'][stakeholder].items():
                    row[f'utility_{exp_type}'] = utility_score
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_selection_summary(self) -> Dict:
        """
        Get summary of selections.
        
        Returns:
            Summary statistics
        """
        df = self.get_results_dataframe()
        
        summary = {
            'total_instances': len(self.results),
            'by_stakeholder': {},
        }
        
        for stakeholder in df['stakeholder'].unique():
            stakeholder_df = df[df['stakeholder'] == stakeholder]
            selections = stakeholder_df['selected_explanation'].value_counts()
            
            summary['by_stakeholder'][stakeholder] = {
                'total': len(stakeholder_df),
                'selections': selections.to_dict(),
                'selection_percentages': (selections / len(stakeholder_df) * 100).to_dict(),
            }
        
        return summary
    
    def compare_baseline_vs_proposed(self, baseline_explanation_type: str = 'shap') -> Dict:
        """
        Compare baseline (same explanation for all) vs. proposed (stakeholder-specific).
        
        Args:
            baseline_explanation_type: Baseline explanation method
            
        Returns:
            Comparison results
        """
        df = self.get_results_dataframe()
        
        # Baseline: all stakeholders get the same explanation
        baseline_utility = {}
        for stakeholder in df['stakeholder'].unique():
            stakeholder_df = df[df['stakeholder'] == stakeholder]
            utility_col = f'utility_{baseline_explanation_type}'
            if utility_col in stakeholder_df.columns:
                baseline_utility[stakeholder] = stakeholder_df[utility_col].mean()
            else:
                baseline_utility[stakeholder] = 0.0
        
        # Proposed: each stakeholder gets optimal explanation
        proposed_utility = {}
        for stakeholder in df['stakeholder'].unique():
            stakeholder_df = df[df['stakeholder'] == stakeholder]
            # For each row, get the utility of the selected explanation
            utilities = []
            for _, row in stakeholder_df.iterrows():
                selected = row['selected_explanation']
                utility_col = f'utility_{selected}'
                if utility_col in row.index:
                    utilities.append(row[utility_col])
            if utilities:
                proposed_utility[stakeholder] = np.mean(utilities)
            else:
                proposed_utility[stakeholder] = 0.0
        
        return {
            'baseline': baseline_utility,
            'proposed': proposed_utility,
            'improvement': {
                stakeholder: (
                    (proposed_utility.get(stakeholder, 0) -
                     baseline_utility.get(stakeholder, 0)) /
                    (baseline_utility.get(stakeholder, 1e-6))
                ) * 100
                for stakeholder in baseline_utility.keys()
            },
        }
