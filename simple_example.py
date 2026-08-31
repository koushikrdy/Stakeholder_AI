"""
Simple Example: Generate and compare explanations for one instance

This is a minimal example showing the core functionality.

Usage:
    python3 simple_example.py
"""

from sklearn.linear_model import LogisticRegression
from src.models import DatasetFactory
from src.explanations.generator import (
    SHAPExplanationGenerator,
    LIMEExplanationGenerator,
    CounterfactualExplanationGenerator,
    RuleBasedExplanationGenerator,
)
from src.selector import ExplanationSelector
from src.utility import UtilityFramework


# ============================================================================
# STEP 1: Load and Train
# ============================================================================
print("\n[1] Loading data and training model...")
loader = DatasetFactory.create('synthetic', random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = loader.load()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"✓ Model accuracy: {model.score(X_test, y_test):.2%}")


# ============================================================================
# STEP 2: Select an instance to explain
# ============================================================================
print("\n[2] Selecting instance to explain...")
instance_idx = 0
instance = X_test.iloc[instance_idx].values
prediction = model.predict_proba([instance])[0, 1]

print(f"Instance index: {instance_idx}")
print(f"Model prediction: {prediction:.4f} (prob. of positive class)")
print(f"Number of features: {len(instance)}")


# ============================================================================
# STEP 3: Generate all explanation types
# ============================================================================
print("\n[3] Generating explanations (all 4 types)...")
print("    (This may take a moment...)")

generators = {
    'SHAP': SHAPExplanationGenerator(model, X_train, num_samples=30),
    'LIME': LIMEExplanationGenerator(model, X_train, num_samples=30),
    'Counterfactual': CounterfactualExplanationGenerator(model, X_train),
    'Rule-based': RuleBasedExplanationGenerator(model, X_train, y_train),
}

explanations = []
for name, gen in generators.items():
    exp = gen.generate(instance)
    explanations.append(exp)
    print(f"    ✓ {name:20s}: {exp.get_summary()}")


# ============================================================================
# STEP 4: Show what each stakeholder prefers
# ============================================================================
print("\n" + "=" * 80)
print("STAKEHOLDER PREFERENCES: Different explanations for different stakeholders!")
print("=" * 80)

selector = ExplanationSelector()
utility_framework = UtilityFramework()

for stakeholder in ['doctor', 'patient', 'regulator']:
    print(f"\n{stakeholder.upper()}:")
    print("-" * 40)
    
    # Select best explanation for this stakeholder
    selected_exp, utilities, components = selector.select(
        explanations, prediction, stakeholder
    )
    
    # Show the ranking
    print("\nExplanation rankings (by utility):")
    for exp_type in sorted(utilities.keys(), key=lambda x: utilities[x], reverse=True):
        utility_val = utilities[exp_type]
        bar = "█" * int(utility_val * 30)
        marker = " ← SELECTED" if exp_type == selected_exp.explanation_type else ""
        print(f"  {exp_type:20s}: {bar} {utility_val:.4f}{marker}")
    
    # Show why this stakeholder prefers this explanation
    print(f"\nWhy {stakeholder} prefers {selected_exp.explanation_type}:")
    utility_func = utility_framework.stakeholders[stakeholder]
    print(f"Stakeholder weights: {utility_func.weights}")
    print("\nComponent scores:")
    for component, score in sorted(components.items(), key=lambda x: x[1], reverse=True):
        weight = utility_func.weights.get(component, 0)
        contribution = weight * score
        print(f"  • {component:20s}: score={score:.3f} × weight={weight:.2f} = {contribution:.4f}")


# ============================================================================
# STEP 5: Show the key finding
# ============================================================================
print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)

selector2 = ExplanationSelector()
doctor_sel = selector2.select(explanations, prediction, 'doctor')[0].explanation_type
patient_sel = selector2.select(explanations, prediction, 'patient')[0].explanation_type
regulator_sel = selector2.select(explanations, prediction, 'regulator')[0].explanation_type

print(f"\nE*_doctor    = {doctor_sel}")
print(f"E*_patient   = {patient_sel}")
print(f"E*_regulator = {regulator_sel}")

if len({doctor_sel, patient_sel, regulator_sel}) > 1:
    print("\n✓✓✓ E*_doctor ≠ E*_patient ≠ E*_regulator")
    print("\nDifferent stakeholders require different explanations!")
    print("\nWhy?")
    print("  • DOCTOR needs actionable changes (Counterfactual)")
    print("  • PATIENT needs simple rules (Rule-based)")
    print("  • REGULATOR needs auditable proofs (SHAP)")

print("\n" + "=" * 80)
