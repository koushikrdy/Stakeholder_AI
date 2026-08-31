# Stakeholder-Aware Explainable AI

This project studies a practical question in explainable AI (XAI):

> Different stakeholders do not need the same explanation for the same model prediction.

A doctor, patient, and regulator often value different explanation properties such as actionability, interpretability, transparency, and auditability. This project implements a stakeholder-aware explanation-selection framework that generates multiple explanation types and selects the most useful one for each audience.

The core objective is:

```text
E*_s = argmax_{E in ℰ} U_s(E | x, ŷ, a)
```

where:
- `E*_s` = best explanation for stakeholder `s`
- `ℰ` = set of candidate explanations
- `U_s` = stakeholder-specific utility function
- `x` = input instance
- `ŷ` = model prediction
- `a` = decision/action context

---

## What this project does

It supports the following explanation methods:
- SHAP
- LIME
- Counterfactual explanations
- Rule-based explanations

It then evaluates them using stakeholder-specific utility functions for:
- doctor
- patient
- regulator

This allows the system to answer questions such as:
- Which explanation is best for a clinician?
- Which explanation is best for a patient?
- Which explanation is best for an auditor/regulator?

---

## Quick start

### 1. Clone and enter the project

```bash
cd /Users/koushikreddytiparthi/Desktop/RP
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the simplest demo

```bash
python3 simple_example.py
```

This is the easiest way to understand the core idea: the same model prediction is explained differently depending on the stakeholder.

---

## Recommended usage paths

Choose the flow that matches what you want to do.

### A. Want to just see the idea working?
Run:

```bash
python3 simple_example.py
```

This prints a simple comparison of how the doctor, patient, and regulator rank the candidate explanations.

### B. Want to run a full experiment?
Run:

```bash
python3 run_experiment.py
```

This runs the main pipeline and produces:
- model training
- explanation generation
- stakeholder selection
- result summaries
- output files in `results/`

### C. Want to evaluate the framework more thoroughly?
Run:

```bash
python3 test_stakeholder_explanations.py
```

This runs a deeper validation suite and reports the selection behavior across several instances.

### D. Want to explore interactively?
Run:

```bash
python3 interactive_test.py
```

This lets you explore explanations and utility scores manually for different instances and stakeholders.

---

## Project structure

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── explanations/
│   │   ├── base.py
│   │   ├── generator.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   └── __init__.py
│   ├── utility/
│   │   ├── stakeholders.py
│   │   └── __init__.py
│   ├── selector/
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── pipeline.py
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   └── test_xai.py
├── paper/
│   ├── paper.tex
│   ├── references.bib
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── results/
├── simple_example.py
├── visual_comparison.py
├── run_experiment.py
├── generate_figures.py
├── interactive_test.py
├── test_stakeholder_explanations.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Training and evaluation workflow

### 1. Data loading
The project loads a dataset and prepares training/validation/test splits.

### 2. Model training
A classifier is trained for the task.

### 3. Explanation generation
Candidates are generated for each instance:
- SHAP
- LIME
- Counterfactual
- Rule-based

### 4. Stakeholder utility scoring
Each explanation is evaluated under a stakeholder-specific utility function.

### 5. Selection
The best explanation is chosen according to:

```python
selected_explanation = selector.select(explanations, prediction, stakeholder)
```

### 6. Reporting and visualization
The project produces summary reports and figures for comparison, utility, and selection statistics.

---

## Running the main experiment

From the project root:

```bash
python3 run_experiment.py
```

This script trains the model and runs the experiment pipeline. Output is written to the `results/` folder.

You can then inspect the report:

```bash
cat results/report.txt
```

---

## Generate figures

To regenerate the paper figures:

```bash
python3 generate_figures.py
```

This saves charts into the `figures/` directory, including:
- stakeholder comparison
- selection statistics
- baseline vs proposed utility
- workflow diagrams

---

## Running the tests

Run the full test suite:

```bash
pytest tests/ -v
```

Or run a focused script:

```bash
python3 test_stakeholder_explanations.py
```

---

## Minimal Python example

```python
from src.selector import ExplanationSelector
from src.utility import UtilityFramework

# assume you already have explanations + prediction
selector = ExplanationSelector()
utility_framework = UtilityFramework()

for stakeholder in ['doctor', 'patient', 'regulator']:
    selected_exp, utilities, components = selector.select(
        explanations,
        prediction=0.78,
        stakeholder=stakeholder,
    )
    print(stakeholder, selected_exp.explanation_type, utilities)
```

---

## Dataset notes

The project supports different datasets depending on the configuration. Check `src/config.py` to modify settings such as:
- dataset source
- split ratios
- model choice
- number of evaluation instances
- explanation settings
- stakeholder utility weights

---

## Interpretation of output

Typical output includes values such as:

```text
Doctor selected: SHAP
Patient selected: Counterfactual
Regulator selected: SHAP
```

This demonstrates that each stakeholder prefers a different explanation style, which is the main finding of the project.

---

## Troubleshooting

### Import errors
If Python cannot find the project modules, run from the repository root:

```bash
cd /Users/koushikreddytiparthi/Desktop/RP
python3 -m pip install -r requirements.txt
```

### Missing dependencies
Install them explicitly:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pytest
```

### Figures not generated
Run:

```bash
python3 generate_figures.py
```

### Results not produced
Run:

```bash
python3 run_experiment.py
```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pytest

---

## Research summary

This project addresses a common limitation in explainable AI: many XAI methods optimize explanations for model fidelity rather than user utility. The framework implemented here makes explanation selection stakeholder-aware and decision-oriented, which is especially relevant in high-stakes domains such as healthcare.

---

## License

This project is distributed under the MIT License unless otherwise specified.

---

## Contact / contribution

If you want to test, extend, or reuse the code, open the project in your local environment and run the scripts above. Contributions and improvements are welcome.
