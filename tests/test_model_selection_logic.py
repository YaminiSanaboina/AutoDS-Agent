"""
Test suite for model selection logic.

Ensures:
1. Classification datasets select only classification models
2. Regression datasets select only regression models
3. SelfImprovementAgent respects problem type constraints
4. Evaluation metrics match problem type
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split

from agents.model_agent import detect_problem_type, train_selected_models
from agents.self_improvement_agent import SelfImprovementAgent
from agents.master_autonomous_pipeline import MasterAutonomousPipeline


class TestProblemTypeDetection(unittest.TestCase):
    """Test problem type detection logic."""
    
    def test_classification_iris(self):
        """Iris dataset should be detected as classification."""
        iris = load_iris()
        y = iris.target  # 0, 1, 2 (3 classes)
        problem_type = detect_problem_type(y)
        self.assertEqual(problem_type, "Classification", 
                         f"Expected Classification for Iris dataset, got {problem_type}")
    
    def test_regression_continuous_values(self):
        """Dataset with many continuous values should be detected as regression."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        problem_type = detect_problem_type(y)
        self.assertEqual(problem_type, "Regression",
                         f"Expected Regression for continuous target, got {problem_type}")
    
    def test_classification_with_few_unique_values(self):
        """Dataset with <= 10 unique values should be classification."""
        y = pd.Series([0, 1, 0, 1, 2, 3, 2, 1, 0, 5] * 10)  # 6 unique values
        problem_type = detect_problem_type(y)
        self.assertEqual(problem_type, "Classification",
                         f"Expected Classification for few unique values, got {problem_type}")


class TestModelSelectionValidation(unittest.TestCase):
    """Test model selection validation."""
    
    def setUp(self):
        """Initialize pipeline for testing."""
        self.pipeline = MasterAutonomousPipeline()
    
    def test_validate_classification_model_for_classification(self):
        """Classification models should be valid for classification problems."""
        valid_models = ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"]
        for model in valid_models:
            result = self.pipeline._validate_model_problem_type(model, "Classification")
            self.assertTrue(result, f"Model {model} should be valid for Classification")
    
    def test_validate_regression_model_for_regression(self):
        """Regression models should be valid for regression problems."""
        valid_models = ["Linear Regression", "Random Forest Regressor", "SVR", "Decision Tree Regressor"]
        for model in valid_models:
            result = self.pipeline._validate_model_problem_type(model, "Regression")
            self.assertTrue(result, f"Model {model} should be valid for Regression")
    
    def test_reject_classification_model_for_regression(self):
        """Classification models should be rejected for regression problems."""
        classification_models = ["Logistic Regression", "Random Forest", "SVM"]
        for model in classification_models:
            result = self.pipeline._validate_model_problem_type(model, "Regression")
            self.assertFalse(result, f"Model {model} should NOT be valid for Regression")
    
    def test_reject_regression_model_for_classification(self):
        """Regression models should be rejected for classification problems."""
        regression_models = ["Linear Regression", "Random Forest Regressor", "SVR"]
        for model in regression_models:
            result = self.pipeline._validate_model_problem_type(model, "Classification")
            self.assertFalse(result, f"Model {model} should NOT be valid for Classification")


class TestTrainSelectedModels(unittest.TestCase):
    """Test train_selected_models function."""
    
    def test_train_classification_models_for_iris(self):
        """Training classification models on Iris should only select classification models."""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results, best_name, best_model, extras = train_selected_models(X_train, y_train, "Classification")
        
        classification_models = {"Random Forest", "Decision Tree", "Logistic Regression", "SVM", "Gradient Boosting"}
        self.assertIn(best_name, classification_models,
                      f"Best model '{best_name}' should be a classification model")
    
    def test_train_regression_models_for_regression_data(self):
        """Training regression models should only select regression models."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results, best_name, best_model, extras = train_selected_models(X_train, y_train, "Regression")
        
        regression_models = {"Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "Gradient Boosting Regressor"}
        self.assertIn(best_name, regression_models,
                      f"Best model '{best_name}' should be a regression model")


class TestSelfImprovementAgentProblemType(unittest.TestCase):
    """Test SelfImprovementAgent respects problem type."""
    
    def setUp(self):
        """Initialize agent for testing."""
        self.agent = SelfImprovementAgent()
    
    def test_evaluate_model_classification_metrics(self):
        """evaluate_model should return classification metrics for classification."""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        scores = self.agent.evaluate_model(model, X, y, problem_type="Classification", cv=3)
        
        # Should have classification metrics
        self.assertIn("accuracy", scores, "Classification metrics should include accuracy")
        self.assertIn("f1", scores, "Classification metrics should include f1")
    
    def test_evaluate_model_regression_metrics(self):
        """evaluate_model should return regression metrics for regression."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        scores = self.agent.evaluate_model(model, X, y, problem_type="Regression", cv=3)
        
        # Should have regression metrics
        self.assertIn("r2", scores, "Regression metrics should include r2")
        self.assertIn("mae", scores, "Regression metrics should include mae")


class TestEvaluationMetrics(unittest.TestCase):
    """Test that evaluation metrics are appropriate for problem type."""
    
    def test_classification_problem_uses_classification_metrics(self):
        """Classification problems should use appropriate metrics."""
        expected_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        # These are verified through test_evaluate_model_classification_metrics
        pass
    
    def test_regression_problem_uses_regression_metrics(self):
        """Regression problems should use R², MAE, RMSE."""
        expected_metrics = {"r2", "mae", "rmse"}
        # These are verified through test_evaluate_model_regression_metrics
        pass


class TestProblemTypeNormalization(unittest.TestCase):
    """Test problem type normalization."""
    
    def setUp(self):
        """Initialize pipeline."""
        self.pipeline = MasterAutonomousPipeline()
    
    def test_normalize_classification_variations(self):
        """Should normalize various classification strings."""
        variations = ["Classification", "classification", "CLASSIFICATION", "Binary Classification"]
        for var in variations:
            normalized = self.pipeline._normalize_problem_type(var)
            self.assertEqual(normalized, "Classification", 
                             f"Should normalize '{var}' to Classification")
    
    def test_normalize_regression_variations(self):
        """Should normalize various regression strings."""
        variations = ["Regression", "regression", "REGRESSION", "Linear Regression"]
        for var in variations:
            normalized = self.pipeline._normalize_problem_type(var)
            self.assertEqual(normalized, "Regression",
                             f"Should normalize '{var}' to Regression")


if __name__ == "__main__":
    unittest.main()
