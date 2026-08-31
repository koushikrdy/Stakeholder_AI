"""
Unit tests for XAI research project
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explanations.base import (
    CounterfactualExplanation,
    LIMEExplanation,
    RuleBasedExplanation,
    SHAPExplanation,
)
from src.explanations.generator import (
    CounterfactualExplanationGenerator,
    LIMEExplanationGenerator,
    RuleBasedExplanationGenerator,
    SHAPExplanationGenerator,
)
from src.models import DatasetFactory, ModelTrainer
from src.selector import ExplanationSelector, SelectionAnalyzer
from src.utility import DoctorUtility, PatientUtility, RegulatorUtility, UtilityFramework


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.binomial(1, 0.5, n_samples)
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create trained model for testing"""
    X, y = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


class TestExplanations:
    """Test explanation classes"""
    
    def test_shap_explanation_creation(self):
        """Test SHAP explanation creation"""
        shap_values = {'feature_0': 0.5, 'feature_1': -0.3}
        explanation = SHAPExplanation(shap_values, base_value=0.6)
        
        assert explanation.explanation_type == "SHAP"
        assert explanation.base_value == 0.6
        assert len(explanation.get_importance_scores()) == 2
    
    def test_lime_explanation_creation(self):
        """Test LIME explanation creation"""
        weights = {'feature_0': 0.8, 'feature_1': 0.2}
        explanation = LIMEExplanation(weights, prediction=0.7)
        
        assert explanation.explanation_type == "LIME"
        assert explanation.prediction == 0.7
        assert explanation.get_summary() is not None
    
    def test_counterfactual_explanation_creation(self):
        """Test counterfactual explanation creation"""
        original = {'feature_0': 1.0, 'feature_1': 2.0}
        counterfactual = {'feature_0': 1.5, 'feature_1': 2.0}
        changes = {'feature_0': (1.0, 1.5)}
        
        explanation = CounterfactualExplanation(
            original, counterfactual, changes
        )
        
        assert explanation.explanation_type == "Counterfactual"
        assert len(explanation.get_importance_scores()) == 1
    
    def test_rule_based_explanation_creation(self):
        """Test rule-based explanation creation"""
        rules = ['feature_0 > 0.5', 'feature_1 <= 1.0']
        weights = {rules[0]: 0.7, rules[1]: 0.3}
        
        explanation = RuleBasedExplanation(
            rules, weights, "Positive"
        )
        
        assert explanation.explanation_type == "Rule-based"
        assert explanation.prediction_class == "Positive"


class TestExplanationGenerators:
    """Test explanation generators"""
    
    def test_shap_generator(self, trained_model):
        """Test SHAP explanation generation"""
        model, X, y = trained_model
        generator = SHAPExplanationGenerator(model, X, num_samples=10)
        
        instance = X.iloc[0].values
        explanation = generator.generate(instance)
        
        assert explanation.explanation_type == "SHAP"
        assert len(explanation.get_importance_scores()) == X.shape[1]
    
    def test_lime_generator(self, trained_model):
        """Test LIME explanation generation"""
        model, X, y = trained_model
        generator = LIMEExplanationGenerator(model, X, num_samples=10)
        
        instance = X.iloc[0].values
        explanation = generator.generate(instance)
        
        assert explanation.explanation_type == "LIME"
        assert len(explanation.get_importance_scores()) == X.shape[1]
    
    def test_counterfactual_generator(self, trained_model):
        """Test counterfactual explanation generation"""
        model, X, y = trained_model
        generator = CounterfactualExplanationGenerator(model, X)
        
        instance = X.iloc[0].values
        explanation = generator.generate(instance)
        
        assert explanation.explanation_type == "Counterfactual"
    
    def test_rule_based_generator(self, trained_model):
        """Test rule-based explanation generation"""
        model, X, y = trained_model
        generator = RuleBasedExplanationGenerator(model, X, y)
        
        instance = X.iloc[0].values
        explanation = generator.generate(instance)
        
        assert explanation.explanation_type == "Rule-based"
        assert len(explanation.rules) > 0


class TestUtilityFunctions:
    """Test stakeholder utility functions"""
    
    def test_doctor_utility(self):
        """Test doctor utility computation"""
        utility = DoctorUtility()
        
        shap_values = {'feature_0': 0.5, 'feature_1': 0.3}
        explanation = SHAPExplanation(shap_values, 0.6)
        
        score = utility.compute_utility(explanation, 0.8)
        
        assert 0 <= score <= 1
    
    def test_patient_utility(self):
        """Test patient utility computation"""
        utility = PatientUtility()
        
        rules = ['rule_1', 'rule_2']
        explanation = RuleBasedExplanation(rules, {rules[0]: 0.8}, "Positive")
        
        score = utility.compute_utility(explanation, 0.7)
        
        assert 0 <= score <= 1
    
    def test_regulator_utility(self):
        """Test regulator utility computation"""
        utility = RegulatorUtility()
        
        weights = {'feature_0': 0.5, 'feature_1': 0.3}
        explanation = LIMEExplanation(weights, 0.6)
        
        score = utility.compute_utility(explanation, 0.9)
        
        assert 0 <= score <= 1
    
    def test_utility_framework(self):
        """Test utility framework"""
        framework = UtilityFramework()
        
        weights = {'feature_0': 0.5}
        explanation = LIMEExplanation(weights, 0.6)
        
        utilities = framework.compute_all_utilities(explanation, 0.7)
        
        assert len(utilities) == 3
        assert all(0 <= u <= 1 for u in utilities.values())


class TestExplanationSelector:
    """Test explanation selection"""
    
    def test_selector_basic(self, trained_model):
        """Test basic explanation selection"""
        model, X, y = trained_model
        
        # Generate explanations
        generators = {
            'shap': SHAPExplanationGenerator(model, X, num_samples=5),
            'lime': LIMEExplanationGenerator(model, X, num_samples=5),
        }
        
        instance = X.iloc[0].values
        explanations = [
            generators['shap'].generate(instance),
            generators['lime'].generate(instance),
        ]
        
        # Select
        selector = ExplanationSelector()
        selected, utilities, components = selector.select(
            explanations, 0.7, 'doctor'
        )
        
        assert selected in explanations
        assert len(utilities) == 2
        assert len(components) > 0
    
    def test_multi_stakeholder_selection(self, trained_model):
        """Test selection for multiple stakeholders"""
        model, X, y = trained_model
        
        generator = SHAPExplanationGenerator(model, X, num_samples=5)
        explanations = [generator.generate(X.iloc[i].values) for i in range(3)]
        
        selector = ExplanationSelector()
        results = selector.select_for_multiple_stakeholders(
            explanations, 0.7,
            stakeholders=['doctor', 'patient']
        )
        
        assert len(results) == 2
        assert all(isinstance(r[0], type(explanations[0])) for r in results.values())
    
    def test_selection_analyzer(self):
        """Test selection analysis"""
        # Create dummy explanations
        exp1 = SHAPExplanation({'f1': 0.5}, 0.6)
        exp2 = LIMEExplanation({'f1': 0.5}, 0.6)
        
        selections = {
            'doctor': (exp1, {'SHAP': 0.8}),
            'patient': (exp2, {'LIME': 0.9}),
        }
        
        analysis = SelectionAnalyzer.compare_selections(selections)
        
        assert analysis['num_stakeholders'] == 2
        assert not analysis['all_same']


class TestDataset:
    """Test dataset functionality"""
    
    def test_dataset_factory(self):
        """Test dataset factory"""
        loader = DatasetFactory.create('synthetic', random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
        
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]


class TestModelTrainer:
    """Test model training"""
    
    def test_train_logistic_regression(self):
        """Test logistic regression training"""
        loader = DatasetFactory.create('synthetic', random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
        
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_logistic_regression(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_evaluation(self):
        """Test model evaluation"""
        loader = DatasetFactory.create('synthetic', random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_logistic_regression(X_train, y_train)
        metrics = trainer.evaluate('logistic_regression', X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
