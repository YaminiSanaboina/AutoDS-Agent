from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResearchScientistAgent:
    def __init__(self) -> None:
        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[Dict[str, Any]]:
        return [
            {
                "algorithm": "Linear Regression",
                "category": "Linear Models",
                "advantages": [
                    "Fast training",
                    "Interpretability",
                    "Works well for linear relationships",
                ],
                "limitations": [
                    "Cannot capture non-linear patterns without features",
                    "Sensitive to outliers",
                ],
                "best_use_cases": [
                    "Regression on small-to-medium tabular datasets",
                    "Financial forecasting",
                    "Baseline modeling",
                ],
                "interpretability_score": 9,
                "training_cost": "Low",
                "accuracy_potential": 4,
            },
            {
                "algorithm": "Logistic Regression",
                "category": "Linear Models",
                "advantages": [
                    "Good interpretability",
                    "Efficient for binary classification",
                    "Works well with regularization",
                ],
                "limitations": [
                    "Assumes linear decision boundary",
                    "Poor performance on complex non-linear data",
                ],
                "best_use_cases": [
                    "Binary classification in healthcare",
                    "Risk scoring",
                    "Baseline classification",
                ],
                "interpretability_score": 9,
                "training_cost": "Low",
                "accuracy_potential": 5,
            },
            {
                "algorithm": "Decision Tree",
                "category": "Tree-Based Models",
                "advantages": [
                    "Interpretable decision paths",
                    "Handles mixed feature types",
                    "Requires little preprocessing",
                ],
                "limitations": [
                    "Prone to overfitting",
                    "Can be unstable with small dataset changes",
                ],
                "best_use_cases": [
                    "Small to medium tabular datasets",
                    "Explainable decision rules",
                    "Feature importance analysis",
                ],
                "interpretability_score": 8,
                "training_cost": "Low",
                "accuracy_potential": 6,
            },
            {
                "algorithm": "Random Forest",
                "category": "Ensemble Learning",
                "advantages": [
                    "Robust to overfitting",
                    "Handles high-dimensional data",
                    "Generates feature importance",
                ],
                "limitations": [
                    "Less interpretable than single trees",
                    "Longer training time than simple models",
                ],
                "best_use_cases": [
                    "Strong baseline for classification and regression",
                    "Datasets with mixed feature types",
                    "Problems requiring robustness",
                ],
                "interpretability_score": 5,
                "training_cost": "Medium",
                "accuracy_potential": 8,
            },
            {
                "algorithm": "Gradient Boosting",
                "category": "Ensemble Learning",
                "advantages": [
                    "High predictive performance",
                    "Can model complex relationships",
                    "Often wins tabular ML tasks",
                ],
                "limitations": [
                    "Slower to train",
                    "More sensitive to hyperparameters",
                ],
                "best_use_cases": [
                    "Competitive tabular modeling",
                    "Structured data with moderate size",
                    "Regression and classification tasks",
                ],
                "interpretability_score": 4,
                "training_cost": "High",
                "accuracy_potential": 9,
            },
            {
                "algorithm": "XGBoost",
                "category": "Gradient Boosting",
                "advantages": [
                    "Fast gradient boosting implementation",
                    "Strong regularization support",
                    "Handles missing values well",
                ],
                "limitations": [
                    "Memory intensive for large datasets",
                    "Requires careful tuning",
                ],
                "best_use_cases": [
                    "Winning tabular ML competitions",
                    "Healthcare and finance prediction",
                    "High-performance baselines",
                ],
                "interpretability_score": 4,
                "training_cost": "High",
                "accuracy_potential": 9,
            },
            {
                "algorithm": "LightGBM",
                "category": "Gradient Boosting",
                "advantages": [
                    "Very fast training",
                    "Efficient memory usage",
                    "Handles large datasets",
                ],
                "limitations": [
                    "Can overfit small datasets",
                    "Less interpretable",
                ],
                "best_use_cases": [
                    "Large-scale tabular datasets",
                    "Fast iteration cycles",
                    "High-cardinality categorical data",
                ],
                "interpretability_score": 4,
                "training_cost": "Medium",
                "accuracy_potential": 9,
            },
            {
                "algorithm": "CatBoost",
                "category": "Gradient Boosting",
                "advantages": [
                    "Excellent categorical handling",
                    "Good default performance",
                    "Robust to overfitting",
                ],
                "limitations": [
                    "Training can be slower",
                    "Less ecosystem maturity than XGBoost",
                ],
                "best_use_cases": [
                    "Datasets with categorical features",
                    "Problems needing fast tuning",
                    "Classification with mixed data",
                ],
                "interpretability_score": 4,
                "training_cost": "Medium",
                "accuracy_potential": 9,
            },
            {
                "algorithm": "SVM",
                "category": "Kernel Methods",
                "advantages": [
                    "Strong performance on small datasets",
                    "Effective with high-dimensional data",
                    "Works well with kernel tricks",
                ],
                "limitations": [
                    "Hard to scale to large datasets",
                    "Requires feature scaling",
                ],
                "best_use_cases": [
                    "Small to medium classification tasks",
                    "Text classification",
                    "When margin-based separation is appropriate",
                ],
                "interpretability_score": 3,
                "training_cost": "Medium",
                "accuracy_potential": 8,
            },
            {
                "algorithm": "Neural Networks",
                "category": "Deep Learning",
                "advantages": [
                    "Flexible and expressive models",
                    "Excellent for complex patterns",
                    "Strong performance with large data",
                ],
                "limitations": [
                    "Harder to interpret",
                    "Requires more data and tuning",
                ],
                "best_use_cases": [
                    "Large datasets with complex structure",
                    "Image, text, and time-series tasks",
                    "Problems requiring nonlinear learning",
                ],
                "interpretability_score": 2,
                "training_cost": "High",
                "accuracy_potential": 9,
            },
        ]

    def get_algorithm(self, name: str) -> Optional[Dict[str, Any]]:
        lower = name.strip().lower()
        for item in self.knowledge_base:
            if item["algorithm"].strip().lower() == lower:
                return item
        return None

    def compare_algorithms(self, model_a: str, model_b: str) -> Dict[str, Any]:
        a = self.get_algorithm(model_a)
        b = self.get_algorithm(model_b)
        if a is None or b is None:
            return {
                "winner": "Unknown",
                "comparison": {},
                "recommendation": "One or both algorithms are not in the knowledge base.",
            }

        a_score = a["accuracy_potential"] + (10 - a["training_cost"].count("H") * 2)
        b_score = b["accuracy_potential"] + (10 - b["training_cost"].count("H") * 2)

        winner = a["algorithm"] if a_score >= b_score else b["algorithm"]

        comparison = {
            "accuracy": f"{a['algorithm']} may be {self._compare_metric(a['accuracy_potential'], b['accuracy_potential'])} than {b['algorithm']}",
            "speed": f"{a['algorithm']} is generally {self._compare_training_cost(a['training_cost'], b['training_cost'])} than {b['algorithm']}",
            "interpretability": f"{a['algorithm']} is {self._compare_metric(a['interpretability_score'], b['interpretability_score'])} to interpret than {b['algorithm']}",
            "scalability": f"{a['algorithm']} is {self._compare_scalability(a['training_cost'], b['training_cost'])} than {b['algorithm']}",
        }

        recommendation = (
            f"Use {winner} when you need {('higher accuracy' if winner == a['algorithm'] else 'a stronger baseline')} "
            f"and consider the trade-off between interpretability and training cost."
        )

        return {
            "winner": winner,
            "comparison": comparison,
            "recommendation": recommendation,
        }

    def recommend_research_direction(self, dataset_report: Dict[str, Any]) -> Dict[str, Any]:
        domain = dataset_report.get("domain", "General")
        problem_type = dataset_report.get("problem_type", "Unknown")
        risks = dataset_report.get("risks", [])
        dataset_size = dataset_report.get("dataset_size", 0)
        feature_count = dataset_report.get("feature_count", 0)

        recommendations: List[str] = []
        if "Healthcare" in domain or problem_type.lower().startswith("binary"):
            recommendations.extend(["Logistic Regression", "Random Forest", "XGBoost"])
        else:
            recommendations.extend(["Random Forest", "Gradient Boosting", "LightGBM"])

        feature_ideas = [
            "Create ratio and interaction features",
            "Encode categorical variables carefully",
            "Add polynomial features for numeric predictors",
        ]

        if dataset_size < 500:
            data_suggestions = ["Collect more samples", "Use cross-validation", "Consider simpler models"]
        else:
            data_suggestions = ["Expand feature engineering", "Monitor drift", "Validate with holdout data"]

        metrics = ["ROC-AUC", "Precision", "Recall"] if "classification" in problem_type.lower() else ["RMSE", "MAE", "R²"]

        risks_output = [
            "Data leakage requires careful validation" if "leakage" in str(risks).lower() else "",
            "Class imbalance may require resampling" if "imbalance" in str(risks).lower() else "",
        ]
        risks_output = [item for item in risks_output if item]

        return {
            "recommended_algorithms": recommendations,
            "feature_engineering_ideas": feature_ideas,
            "data_collection_suggestions": data_suggestions,
            "evaluation_metrics": metrics,
            "research_risks": risks_output or ["Verify dataset quality and model assumptions."],
        }

    def suggest_improvements(
        self,
        current_model: str,
        performance_metrics: Dict[str, Any],
        dataset_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        suggestions: Dict[str, List[str]] = {"high_priority": [], "medium_priority": []}
        accuracy = performance_metrics.get("accuracy")
        train_score = performance_metrics.get("train_score")
        test_score = performance_metrics.get("test_score")

        if dataset_report.get("risks"):
            if any("leakage" in str(r).lower() for r in dataset_report["risks"]):
                suggestions["high_priority"].append("Investigate and remove potential data leakage.")
            if any("imbalance" in str(r).lower() for r in dataset_report["risks"]):
                suggestions["high_priority"].append("Address class imbalance with resampling or class weighting.")

        if accuracy is not None and isinstance(accuracy, (int, float)) and accuracy < 0.8:
            suggestions["high_priority"].append("Add feature engineering and collect more samples.")

        if train_score is not None and test_score is not None and isinstance(train_score, (int, float)) and isinstance(test_score, (int, float)):
            if train_score - test_score > 0.15:
                suggestions["high_priority"].append("Reduce overfitting with regularization or simpler models.")
            elif test_score < 0.7:
                suggestions["medium_priority"].append("Try more advanced algorithms like XGBoost or LightGBM.")

        if feature_count := dataset_report.get("feature_count"):
            if feature_count < 10:
                suggestions["high_priority"].append("Create more features from domain knowledge.")

        if not suggestions["high_priority"]:
            suggestions["high_priority"].append("Review current preprocessing and fine-tune hyperparameters.")
        if not suggestions["medium_priority"]:
            suggestions["medium_priority"].append("Test ensemble methods and monitor generalization.")

        return suggestions

    def summarize_research_topic(self, topic: str) -> Dict[str, Any]:
        normalized = topic.strip().lower()
        if "explainable" in normalized:
            return {
                "topic": "Explainable AI",
                "summary": "Explainable AI focuses on building models that humans can understand and trust, often by using interpretable algorithms or post-hoc explanation techniques.",
                "key_techniques": ["SHAP", "LIME", "Feature importance", "Rule extraction"],
                "future_trends": ["Model transparency regulations", "Human-AI collaboration", "Interpretable deep learning"],
                "industry_applications": ["Healthcare", "Finance", "Legal compliance"],
            }
        if "automl" in normalized:
            return {
                "topic": "AutoML",
                "summary": "AutoML automates model selection, feature engineering, and hyperparameter tuning to accelerate machine learning workflows.",
                "key_techniques": ["Bayesian optimization", "Neural architecture search", "Pipeline search"],
                "future_trends": ["AutoML for enterprise", "Responsible AutoML", "Integration with low-code platforms"],
                "industry_applications": ["Retail", "Manufacturing", "Healthcare"],
            }
        if "transfer learning" in normalized:
            return {
                "topic": "Transfer Learning",
                "summary": "Transfer learning reuses knowledge from a pre-trained model on a new but related task, reducing training time and data requirements.",
                "key_techniques": ["Fine-tuning", "Feature extraction", "Domain adaptation"],
                "future_trends": ["Cross-domain transfer", "Few-shot learning", "Model reuse marketplaces"],
                "industry_applications": ["Computer vision", "Natural language processing", "Medical imaging"],
            }
        if "deep learning" in normalized or "neural" in normalized:
            return {
                "topic": "Deep Learning",
                "summary": "Deep learning uses layered neural networks to learn hierarchical representations from data, especially effective for complex tasks.",
                "key_techniques": ["CNNs", "RNNs", "Transformers", "Regularization"],
                "future_trends": ["Foundation models", "Multimodal learning", "Efficient architectures"],
                "industry_applications": ["Vision", "Speech", "Autonomous systems"],
            }
        if "feature engineering" in normalized:
            return {
                "topic": "Feature Engineering",
                "summary": "Feature engineering creates new inputs from raw data to improve model performance, often using domain knowledge and transformations.",
                "key_techniques": ["Encoding", "Normalization", "Interaction features", "Dimensionality reduction"],
                "future_trends": ["Automated feature synthesis", "Graph features", "Feature stores"],
                "industry_applications": ["Finance", "Customer analytics", "Forecasting"],
            }
        if "data drift" in normalized:
            return {
                "topic": "Data Drift",
                "summary": "Data drift refers to changes in data distribution over time, which can degrade model performance if not monitored and managed.",
                "key_techniques": ["Drift detection", "Reference monitoring", "Retraining triggers"],
                "future_trends": ["Continuous monitoring", "Adaptive models", "Drift-aware pipelines"],
                "industry_applications": ["Fraud detection", "Retail demand forecasting", "Healthcare monitoring"],
            }

        return {
            "topic": topic,
            "summary": "This topic is not in the built-in knowledge base, but generally research should focus on foundational principles and practical applications.",
            "key_techniques": [],
            "future_trends": [],
            "industry_applications": [],
        }

    def create_experiment_plan(self, project_goal: str) -> Dict[str, Any]:
        plan = [
            {
                "phase": "Phase 1",
                "focus": "Baseline modeling with interpretable algorithms",
                "steps": ["Train Logistic Regression", "Evaluate baseline metrics", "Review feature quality"],
            },
            {
                "phase": "Phase 2",
                "focus": "Strong tree-based baseline",
                "steps": ["Train Random Forest", "Inspect feature importance", "Compare to baseline"],
            },
            {
                "phase": "Phase 3",
                "focus": "Gradient boosting optimization",
                "steps": ["Train XGBoost or LightGBM", "Tune hyperparameters", "Validate generalization"],
            },
            {
                "phase": "Phase 4",
                "focus": "Explainability and deployment readiness",
                "steps": ["Generate SHAP insights", "Review risks", "Prepare production notes"],
            },
        ]
        return {
            "project_goal": project_goal,
            "experiment_plan": plan,
        }

    def generate_research_report(
        self,
        dataset_report: Dict[str, Any],
        current_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        direction = self.recommend_research_direction(dataset_report)
        current = current_model or "No current model specified"
        risks = dataset_report.get("risks", [])

        return {
            "current_understanding": {
                "dataset_domain": dataset_report.get("domain", "Unknown"),
                "problem_type": dataset_report.get("problem_type", "Unknown"),
                "model": current,
            },
            "recommended_models": direction["recommended_algorithms"],
            "risks": risks,
            "experimental_roadmap": self.create_experiment_plan(dataset_report.get("project_goal", "Improve model performance"))["experiment_plan"],
            "future_improvements": [
                "Monitor data quality and drift",
                "Expand feature engineering based on domain knowledge",
                "Tune model hyperparameters carefully",
            ],
        }

    def _compare_metric(self, score_a: int, score_b: int) -> str:
        if score_a > score_b:
            return "more accurate"
        if score_a < score_b:
            return "less accurate"
        return "similarly accurate"

    def _compare_training_cost(self, cost_a: str, cost_b: str) -> str:
        costs = {"Low": 1, "Medium": 2, "High": 3}
        return "faster" if costs.get(cost_a, 2) < costs.get(cost_b, 2) else "slower" if costs.get(cost_a, 2) > costs.get(cost_b, 2) else "similarly fast"

    def _compare_scalability(self, cost_a: str, cost_b: str) -> str:
        costs = {"Low": 3, "Medium": 2, "High": 1}
        return "more scalable" if costs.get(cost_a, 2) > costs.get(cost_b, 2) else "less scalable" if costs.get(cost_a, 2) < costs.get(cost_b, 2) else "similarly scalable"
