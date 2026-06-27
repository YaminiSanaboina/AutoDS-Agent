import sys
sys.path.insert(0, '.')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from agents.xai_agent import generate_shap_explanation


def try_shap_for_classification(df, target):
    df = df.dropna()
    y = df[target]
    X = df.drop(columns=[target])
    # simple encoding for categorical
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y if len(y.unique())>1 else None)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    X_sample = X_test.sample(n=min(50, len(X_test)), random_state=42)
    explainer, shap_vals = generate_shap_explanation(model, X_sample)
    print('SHAP generated for classification target', target)


def try_shap_for_regression(df, target):
    df = df.dropna()
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    X_sample = X_test.sample(n=min(50, len(X_test)), random_state=42)
    explainer, shap_vals = generate_shap_explanation(model, X_sample)
    print('SHAP generated for regression target', target)


if __name__ == '__main__':
    # Iris
    iris = pd.read_csv('data/Iris.csv')
    # Guess target column (common names)
    for t in ['species', 'Species', 'target']:
        if t in iris.columns:
            try_shap_for_classification(iris, t)
            break

    # Titanic
    titanic = pd.read_csv('data/Titanic-Dataset.csv')
    for t in ['survived', 'Survived', 'target']:
        if t in titanic.columns:
            try_shap_for_classification(titanic, t)
            break

    # Housing
    housing = pd.read_csv('data/Housing.csv')
    for t in ['price', 'SalePrice', 'target']:
        if t in housing.columns:
            try_shap_for_regression(housing, t)
            break

    # Wine
    wine = pd.read_csv('data/Wine-Quality.csv')
    for t in ['quality', 'Quality', 'target']:
        if t in wine.columns:
            # wine quality can be regression/classification, try classification
            try_shap_for_classification(wine, t)
            break
