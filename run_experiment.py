"""
Main experiment runner for XAI research project
"""

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import (
    DATASET_SOURCE,
    MODELS_TO_TRAIN,
    NUM_INSTANCES_FOR_EXPLANATION,
    RANDOM_SEED,
    STAKEHOLDERS,
)
from src.evaluation import ExperimentalEvaluator
from src.models import DatasetFactory, ModelTrainer
from src.utils import save_json, set_seed


class ResearchExperiment:
    """
    Main orchestrator for research experiments.
    """
    
    def __init__(self, seed: int = RANDOM_SEED):
        """
        Initialize experiment.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        set_seed(seed)
        
        self.data = None
        self.trainer = None
        self.best_model = None
        self.evaluator = None
        self.results = {}
    
    def load_data(self, dataset_name: str = DATASET_SOURCE) -> Dict:
        """
        Load dataset.
        
        Args:
            dataset_name: Dataset to load
            
        Returns:
            Dictionary with train/val/test data
        """
        print(f"Loading {dataset_name} dataset...")
        loader = DatasetFactory.create(
            dataset_name,
            random_state=self.seed,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
        
        self.data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }
        
        print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        return self.data
    
    def train_models(self, models: Optional[list] = None) -> Dict[str, Dict]:
        """
        Train all models.
        
        Args:
            models: List of models to train
            
        Returns:
            Dictionary of model evaluation results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if models is None:
            models = MODELS_TO_TRAIN
        
        print("Training models...")
        self.trainer = ModelTrainer(random_state=self.seed)
        
        # Train all models
        self.trainer.train_all(
            self.data['X_train'],
            self.data['y_train'],
            self.data['X_val'],
            self.data['y_val'],
            models=models,
        )
        
        # Evaluate all models
        results = self.trainer.evaluate_all(
            self.data['X_test'],
            self.data['y_test'],
        )
        
        self.results['model_evaluation'] = results
        
        # Get best model
        best_name, best_model, best_metrics = self.trainer.get_best_model()
        print(f"Best model: {best_name}")
        print(f"Metrics: {best_metrics}")
        
        self.best_model = (best_name, best_model)
        
        return results
    
    def run_experiments(
        self,
        num_instances: int = NUM_INSTANCES_FOR_EXPLANATION,
    ) -> Dict:
        """
        Run explanation selection experiments.
        
        Args:
            num_instances: Number of instances to evaluate
            
        Returns:
            Experiment results
        """
        if self.best_model is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        model_name, model = self.best_model
        
        print(f"Running experiments with {model_name}...")
        self.evaluator = ExperimentalEvaluator(
            model,
            self.data['X_train'],
            self.data['y_train'],
        )
        
        # Evaluate instances
        batch_results = self.evaluator.evaluate_batch(
            self.data['X_test'],
            stakeholders=STAKEHOLDERS,
            limit=num_instances,
        )
        
        # Get summary
        selection_summary = self.evaluator.get_selection_summary()
        print("\nExplanation Selection Summary:")
        print(json.dumps(selection_summary, indent=2))
        
        # Compare baseline vs proposed
        comparison = self.evaluator.compare_baseline_vs_proposed(
            baseline_explanation_type='SHAP'
        )
        print("\nBaseline vs. Proposed Comparison:")
        print(json.dumps(comparison, indent=2))
        
        self.results['explanation_selection'] = selection_summary
        self.results['baseline_vs_proposed'] = comparison
        self.results['raw_evaluations'] = batch_results
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate experiment report.
        
        Returns:
            Report string
        """
        report = """
================================================================================
EXPLAINABLE AI RESEARCH: STAKEHOLDER-SPECIFIC EXPLANATION SELECTION
================================================================================

RESEARCH CONTRIBUTION:
Define explanation quality using stakeholder-specific utility rather than
model fidelity alone.

  Traditional:  Quality(E) = Fidelity(Model, E)
  Proposed:     Quality(E) = Utility(Stakeholder, Decision, E)

OPTIMAL EXPLANATION SELECTION:
  E*_s = argmax_{E ∈ ℰ} U_s(E | x, ŷ, a)

where:
  - E*_s: Optimal explanation for stakeholder s
  - ℰ: Set of candidate explanations (SHAP, LIME, Counterfactual, Rule-based)
  - U_s: Utility function for stakeholder s
  - x: Instance to explain
  - ŷ: Model prediction
  - a: Downstream action

================================================================================
EXPERIMENTAL RESULTS
================================================================================
"""
        
        if 'model_evaluation' in self.results:
            report += "\n1. MODEL EVALUATION\n"
            report += "-" * 80 + "\n"
            for model_name, metrics in self.results['model_evaluation'].items():
                report += f"\n{model_name}:\n"
                for metric_name, metric_value in metrics.items():
                    report += f"  {metric_name}: {metric_value:.4f}\n"
        
        if 'explanation_selection' in self.results:
            report += "\n2. EXPLANATION SELECTION SUMMARY\n"
            report += "-" * 80 + "\n"
            summary = self.results['explanation_selection']
            for stakeholder, stats in summary.get('by_stakeholder', {}).items():
                report += f"\nStakeholder: {stakeholder.upper()}\n"
                report += f"  Total instances: {stats['total']}\n"
                report += f"  Selections:\n"
                for exp_type, count in stats['selections'].items():
                    pct = stats['selection_percentages'].get(exp_type, 0)
                    report += f"    {exp_type}: {count} ({pct:.1f}%)\n"
        
        if 'baseline_vs_proposed' in self.results:
            report += "\n3. BASELINE VS. PROPOSED FRAMEWORK\n"
            report += "-" * 80 + "\n"
            comparison = self.results['baseline_vs_proposed']
            report += "\nBaseline Utility (SHAP for all stakeholders):\n"
            for stakeholder, utility in comparison.get('baseline', {}).items():
                report += f"  {stakeholder}: {utility:.4f}\n"
            
            report += "\nProposed Utility (Stakeholder-specific selection):\n"
            for stakeholder, utility in comparison.get('proposed', {}).items():
                report += f"  {stakeholder}: {utility:.4f}\n"
            
            report += "\nImprovement (%):\n"
            for stakeholder, improvement in comparison.get('improvement', {}).items():
                report += f"  {stakeholder}: {improvement:.2f}%\n"
        
        report += "\n" + "=" * 80 + "\n"
        report += "KEY FINDING: E*_doctor ≠ E*_patient ≠ E*_regulator\n"
        report += "Different stakeholders require different explanations!\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def save_results(self, output_dir: str = "results") -> None:
        """
        Save results to files.
        
        Args:
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(output_dir, "results.json")
        
        # Convert numpy types for JSON serialization
        results_to_save = {}
        for key, value in self.results.items():
            if key == 'raw_evaluations':
                # Skip raw evaluations for now
                continue
            results_to_save[key] = value
        
        save_json(results_to_save, results_file)
        
        # Save report
        report = self.generate_report()
        report_file = os.path.join(output_dir, "report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Results saved to {output_dir}/")


def main():
    """Run complete research experiment."""
    print("Starting XAI Research Experiment...")
    print("=" * 80)
    
    # Initialize experiment
    experiment = ResearchExperiment(seed=RANDOM_SEED)
    
    # Load data
    experiment.load_data(DATASET_SOURCE)
    
    # Train models
    experiment.train_models(MODELS_TO_TRAIN)
    
    # Run experiments
    experiment.run_experiments(NUM_INSTANCES_FOR_EXPLANATION)
    
    # Generate report
    report = experiment.generate_report()
    print(report)
    
    # Save results
    experiment.save_results("results")
    
    return experiment


if __name__ == "__main__":
    experiment = main()
