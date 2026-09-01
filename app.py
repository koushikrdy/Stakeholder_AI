import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from src.explanations.generator import (
    CounterfactualExplanationGenerator,
    LIMEExplanationGenerator,
    RuleBasedExplanationGenerator,
    SHAPExplanationGenerator,
)
from src.models import DatasetFactory
from src.selector import ExplanationSelector


st.set_page_config(page_title="Stakeholder-Aware XAI Explorer", page_icon="🧠", layout="wide")


@st.cache_data
def load_data_and_model():
    loader = DatasetFactory.create("synthetic", random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


@st.cache_data
def get_instance_data(instance_idx: int):
    X_train, X_test, _, _, model = load_data_and_model()
    instance = X_test.iloc[instance_idx].to_numpy()
    prediction = model.predict_proba([instance])[0, 1]
    return instance, prediction, X_test.columns.tolist()


@st.cache_data
def generate_explanations_for_instance(instance, X_train, y_train, model):
    generators = {
        "SHAP": SHAPExplanationGenerator(model, X_train, num_samples=30),
        "LIME": LIMEExplanationGenerator(model, X_train, num_samples=30),
        "Counterfactual": CounterfactualExplanationGenerator(model, X_train),
        "Rule-based": RuleBasedExplanationGenerator(model, X_train, y_train),
    }
    explanations = []
    for name, generator in generators.items():
        explanations.append(generator.generate(instance))
    return explanations


def format_score_dict(scores):
    return {key: round(float(value), 3) for key, value in scores.items()}


X_train, X_test, y_train, _, model = load_data_and_model()

st.title("Stakeholder-Aware Explainable AI Explorer")
st.caption("Understand how different stakeholders prefer different explanations for the same prediction.")

with st.sidebar:
    st.header("Controls")
    stakeholder = st.selectbox("Stakeholder", ["doctor", "patient", "regulator"])
    instance_index = st.slider("Instance index", min_value=0, max_value=max(0, len(X_test) - 1), value=0, step=1)
    st.markdown("---")
    st.markdown("Core rule: `E*_s = argmax_E U_s(E | x, ŷ, a)`")

instance, prediction, feature_names = get_instance_data(instance_index)

col1, col2, col3 = st.columns(3)
col1.metric("Prediction probability", f"{prediction:.3f}")
col2.metric("Positive class probability", f"{prediction:.1%}")
col3.metric("Features", str(len(feature_names)))

st.subheader("Selected instance")
instance_df = pd.DataFrame({"Feature": feature_names, "Value": instance})
st.dataframe(instance_df, use_container_width=True)

explanations = generate_explanations_for_instance(instance, X_train, y_train, model)
selector = ExplanationSelector()
selected_explanation, utility_scores, component_scores = selector.select(explanations, prediction, stakeholder)

st.subheader(f"Selected explanation for {stakeholder}")
st.success(f"Best explanation: {selected_explanation.explanation_type}")

utility_table = pd.DataFrame(
    {
        "Explanation": list(utility_scores.keys()),
        "Utility score": [round(float(v), 3) for v in utility_scores.values()],
    }
).sort_values("Utility score", ascending=False)

st.dataframe(utility_table, use_container_width=True)

st.bar_chart(utility_table.set_index("Explanation")["Utility score"])

st.subheader("Why this explanation was selected")
component_df = pd.DataFrame(
    {
        "Component": list(component_scores.keys()),
        "Score": [round(float(v), 3) for v in component_scores.values()],
    }
).sort_values("Score", ascending=False)
st.dataframe(component_df, use_container_width=True)

st.subheader("Explanation summary")
if selected_explanation.explanation_type == "SHAP":
    top = sorted(selected_explanation.content["shap_values"].items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    st.table(pd.DataFrame({"Feature": [name for name, _ in top], "SHAP value": [round(float(value), 3) for _, value in top]}))
elif selected_explanation.explanation_type == "LIME":
    top = sorted(selected_explanation.content["weights"].items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    st.table(pd.DataFrame({"Feature": [name for name, _ in top], "Weight": [round(float(value), 3) for _, value in top]}))
elif selected_explanation.explanation_type == "Counterfactual":
    original = selected_explanation.content["original"]
    counterfactual = selected_explanation.content["counterfactual"]
    st.write("Original instance vs. counterfactual change:")
    changes_df = pd.DataFrame(
        {
            "Feature": list(selected_explanation.content["changes"].keys()),
            "Original": [round(float(original[k]), 3) for k in selected_explanation.content["changes"].keys()],
            "Counterfactual": [round(float(counterfactual[k]), 3) for k in selected_explanation.content["changes"].keys()],
        }
    )
    st.dataframe(changes_df, use_container_width=True)
else:
    st.write(selected_explanation.get_summary())

st.markdown("---")
st.subheader("Comparison across all stakeholders")
all_results = {}
for role in ["doctor", "patient", "regulator"]:
    _, role_utilities, _ = selector.select(explanations, prediction, role)
    all_results[role] = role_utilities

comparison_df = pd.DataFrame(index=list(utility_scores.keys()))
for role in ["doctor", "patient", "regulator"]:
    comparison_df[role] = [round(float(all_results[role].get(name, 0.0)), 3) for name in comparison_df.index]

st.dataframe(comparison_df, use_container_width=True)

st.caption("This app demonstrates that the same model prediction can be explained differently depending on stakeholder needs.")
