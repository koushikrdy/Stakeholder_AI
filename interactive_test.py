"""
Interactive test: Manually inspect explanations and utilities

Usage:
    python3 interactive_test.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.explanations.generator import (
    CounterfactualExplanationGenerator,
    LIMEExplanationGenerator,
    RuleBasedExplanationGenerator,
    SHAPExplanationGenerator,
)
from src.models import DatasetFactory
from src.selector import ExplanationSelector
from src.utility import UtilityFramework


def main():
    print("\n" + "=" * 80)
    print("  INTERACTIVE: Test Model Explanations by Stakeholder")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading dataset...")
    loader = DatasetFactory.create('synthetic', random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
    print(f"✓ Loaded: Train={X_train.shape}, Test={X_test.shape}")
    
    # Step 2: Train model
    print("\n[STEP 2] Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(f"✓ Model trained. Accuracy: {model.score(X_test, y_test):.4f}")
    
    # Step 3: Initialize generators
    print("\n[STEP 3] Initializing explanation generators...")
    generators = {
        'SHAP': SHAPExplanationGenerator(model, X_train, num_samples=30),
        'LIME': LIMEExplanationGenerator(model, X_train, num_samples=30),
        'Counterfactual': CounterfactualExplanationGenerator(model, X_train),
        'Rule-based': RuleBasedExplanationGenerator(model, X_train, y_train),
    }
    print("✓ Generators ready:")
    for name in generators.keys():
        print(f"  • {name}")
    
    # Interactive loop
    selector = ExplanationSelector()
    
    while True:
        print("\n" + "-" * 80)
        print("Select an instance to explain (or type 'quit' to exit):")
        print(f"Available instances: 0 to {len(X_test)-1}")
        
        user_input = input("\nInstance index (0-{}, 'quit'): ".format(len(X_test)-1)).strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        try:
            instance_idx = int(user_input)
            if not (0 <= instance_idx < len(X_test)):
                print(f"✗ Invalid index. Please enter 0-{len(X_test)-1}")
                continue
        except ValueError:
            print("✗ Invalid input. Please enter a number or 'quit'")
            continue
        
        # Get instance and prediction
        instance = X_test.iloc[instance_idx].values
        prediction = model.predict_proba([instance])[0, 1]
        
        print("\n" + "=" * 80)
        print(f"INSTANCE {instance_idx}")
        print("=" * 80)
        print(f"Prediction (Prob. of positive): {prediction:.4f}")
        print(f"Number of features: {len(instance)}")
        
        # Generate explanations
        print("\n[Generating explanations...]")
        explanations = []
        for name, generator in generators.items():
            exp = generator.generate(instance)
            explanations.append(exp)
            print(f"✓ {name}: {exp.get_summary()}")
        
        # Show stakeholder comparisons
        print("\n" + "-" * 80)
        print("STAKEHOLDER UTILITY COMPARISON")
        print("-" * 80)
        
        utility_framework = UtilityFramework()
        
        # Menu for stakeholder selection
        stakeholder_menu = {
            '1': 'doctor',
            '2': 'patient',
            '3': 'regulator',
            '0': 'all',
        }
        
        print("\nSelect stakeholder to analyze:")
        print("  1) Doctor")
        print("  2) Patient")
        print("  3) Regulator")
        print("  0) Show all stakeholders")
        
        stakeholder_input = input("\nChoice (0-3): ").strip()
        
        if stakeholder_input not in stakeholder_menu:
            print("✗ Invalid choice")
            continue
        
        selected_stakeholders = (
            ['doctor', 'patient', 'regulator'] 
            if stakeholder_input == '0' 
            else [stakeholder_menu[stakeholder_input]]
        )
        
        # Show analysis
        for stakeholder in selected_stakeholders:
            print("\n" + "=" * 80)
            print(f"{stakeholder.upper()}")
            print("=" * 80)
            
            utility_func = utility_framework.stakeholders[stakeholder]
            selected_exp, utilities, components = selector.select(
                explanations, prediction, stakeholder
            )
            
            # Show utility scores
            print("\nUtility Scores (Higher is better):")
            sorted_utilities = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
            for exp_type, utility_score in sorted_utilities:
                marker = "→ SELECTED" if exp_type == selected_exp.explanation_type else " "
                bar = "█" * int(utility_score * 40) + "░" * (40 - int(utility_score * 40))
                print(f"  {exp_type:20s}: {bar} {utility_score:.4f} {marker}")
            
            # Show component breakdown for selected
            print(f"\nComponent Breakdown ({selected_exp.explanation_type}):")
            print(f"Explanation Type: {selected_exp.explanation_type}")
            print("Components:")
            sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
            for component, score in sorted_components:
                weight = utility_func.weights.get(component, 0)
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {component:20s}: {bar} score={score:.3f}, weight={weight:.2f}")
            
            # Show explanation details
            print(f"\nExplanation Summary:")
            print(f"  {selected_exp.get_summary()}")
            
            # Show importance scores
            importance = selected_exp.get_importance_scores()
            print(f"\nTop 5 Important Features:")
            top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for i, (feature, importance_val) in enumerate(top_features, 1):
                bar = "█" * int(abs(importance_val) * 30) + "░" * (30 - int(abs(importance_val) * 30))
                print(f"  {i}. {feature:20s}: {bar} {importance_val:.4f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
