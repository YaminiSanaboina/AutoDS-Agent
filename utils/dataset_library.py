"""Educational dataset catalog for the Dataset Library page."""

from utils.datasets import get_sample_datasets


def get_library_datasets():
    """Return enriched metadata for datasets available in AutoDS."""
    sample_ids = {ds["id"] for ds in get_sample_datasets()}
    catalog = _LIBRARY_CATALOG
    return [entry for entry in catalog if entry["dataset_id"] in sample_ids]


def get_library_dataset(dataset_id):
    for entry in get_library_datasets():
        if entry["dataset_id"] == dataset_id:
            return entry
    raise ValueError(f"Unknown library dataset: {dataset_id}")


_LIBRARY_CATALOG = [
    {
        "dataset_id": "titanic",
        "name": "Titanic Dataset",
        "icon": "🚢",
        "difficulty": "Beginner",
        "story": (
            "In 1912, the Titanic sank on its maiden voyage. This dataset records "
            "who was on board — their age, gender, ticket class, and whether they survived."
        ),
        "learning_outcomes": [
            "Handle missing values in real-world data",
            "Explore how features like gender and class relate to survival",
            "Build and compare binary classification models",
        ],
        "problem_type": "Binary Classification",
        "algorithms": [
            {
                "name": "Random Forest",
                "reason": "Handles mixed data types and missing values well without heavy tuning.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Simple baseline that shows which factors most affect survival odds.",
            },
            {
                "name": "Decision Tree",
                "reason": "Easy to visualize — great for understanding if/else survival rules.",
            },
        ],
        "source_url": "https://www.kaggle.com/competitions/titanic/data",
        "source_label": "Kaggle",
    },
    {
        "dataset_id": "churn",
        "name": "Customer Churn Dataset",
        "icon": "📞",
        "difficulty": "Intermediate",
        "story": (
            "A telecom company wants to know which customers will cancel their service. "
            "Each row is a customer with contract details, services used, and a churn label."
        ),
        "learning_outcomes": [
            "Work with categorical and numerical features together",
            "Understand class imbalance in business problems",
            "Translate model predictions into retention strategies",
        ],
        "problem_type": "Binary Classification",
        "algorithms": [
            {
                "name": "XGBoost",
                "reason": "Strong on tabular data with many mixed feature types.",
            },
            {
                "name": "Random Forest",
                "reason": "Robust to outliers and captures non-linear churn patterns.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Provides interpretable odds for each customer attribute.",
            },
        ],
        "source_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "source_label": "Kaggle",
    },
    {
        "dataset_id": "housing",
        "name": "Housing Price Dataset",
        "icon": "🏠",
        "difficulty": "Intermediate",
        "story": (
            "Home buyers and sellers need fair price estimates. This dataset lists house "
            "features — size, location, quality — and their sale prices."
        ),
        "learning_outcomes": [
            "Predict continuous numeric targets (regression)",
            "Explore correlations between property features and price",
            "Evaluate models with RMSE and R-squared metrics",
        ],
        "problem_type": "Regression",
        "algorithms": [
            {
                "name": "Random Forest Regressor",
                "reason": "Captures complex interactions between location, size, and quality.",
            },
            {
                "name": "XGBoost Regressor",
                "reason": "Often top-performing on structured housing feature data.",
            },
            {
                "name": "Linear Regression",
                "reason": "Simple baseline showing which features drive price linearly.",
            },
        ],
        "source_url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data",
        "source_label": "Kaggle",
    },
    {
        "dataset_id": "iris",
        "name": "Iris Dataset",
        "icon": "🌸",
        "difficulty": "Beginner",
        "story": (
            "Botanists measured iris flowers from three species. Each row has petal and "
            "sepal dimensions — can you tell which species it belongs to?"
        ),
        "learning_outcomes": [
            "Learn the basics of multi-class classification",
            "Visualize low-dimensional data with scatter plots",
            "Compare simple models on a clean, classic dataset",
        ],
        "problem_type": "Multi-class Classification",
        "algorithms": [
            {
                "name": "Decision Tree",
                "reason": "Creates clear rules like 'if petal length > 4.8, then virginica'.",
            },
            {
                "name": "K-Nearest Neighbors",
                "reason": "Works well when similar flowers cluster together in feature space.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Fast baseline for separating three species with linear boundaries.",
            },
        ],
        "source_url": "https://archive.ics.uci.edu/ml/datasets/iris",
        "source_label": "UCI",
    },
    {
        "dataset_id": "heart",
        "name": "Heart Disease Dataset",
        "icon": "❤️",
        "difficulty": "Intermediate",
        "story": (
            "Doctors recorded patient measurements — cholesterol, blood pressure, chest pain "
            "type — to study signs of heart disease. The goal is early risk detection."
        ),
        "learning_outcomes": [
            "Apply classification to healthcare-style tabular data",
            "Understand precision and recall in medical contexts",
            "Explore feature importance for clinical risk factors",
        ],
        "problem_type": "Binary Classification",
        "algorithms": [
            {
                "name": "Random Forest",
                "reason": "Stable predictions and clear feature importance for risk factors.",
            },
            {
                "name": "SVM",
                "reason": "Effective when classes are separated by complex boundaries.",
            },
            {
                "name": "Logistic Regression",
                "reason": "Interpretable coefficients help explain individual risk drivers.",
            },
        ],
        "source_url": "https://archive.ics.uci.edu/ml/datasets/heart+disease",
        "source_label": "UCI",
    },
    {
        "dataset_id": "wine",
        "name": "Wine Quality Dataset",
        "icon": "🍷",
        "difficulty": "Intermediate",
        "story": (
            "Chemists measured acidity, sugar, and alcohol levels in wine samples. "
            "Can these lab readings predict the wine's quality rating or cultivar?"
        ),
        "learning_outcomes": [
            "Explore chemical feature distributions and correlations",
            "Try both classification and clustering on the same data",
            "Learn when ensemble models outperform single trees",
        ],
        "problem_type": "Classification & Clustering",
        "algorithms": [
            {
                "name": "Random Forest",
                "reason": "Handles correlated chemical features without manual engineering.",
            },
            {
                "name": "K-Means Clustering",
                "reason": "Discovers natural wine groups without using labels.",
            },
            {
                "name": "XGBoost",
                "reason": "Strong accuracy when predicting quality from chemical profiles.",
            },
        ],
        "source_url": "https://archive.ics.uci.edu/ml/datasets/wine+quality",
        "source_label": "UCI",
    },
]
