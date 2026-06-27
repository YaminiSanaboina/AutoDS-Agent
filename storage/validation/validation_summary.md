# AutoDS Validation Summary

Generated: 2026-06-27 08:51 UTC

- Dataset Analysis: PASS
- Problem Type: PASS
- Model Metrics: FAIL
- Best Model Selection: PASS
- Explainability: PASS
- Report Accuracy: PASS
- Overall Validation Score: 83/100

## Section Details

### Dataset Analysis — PASS
- **Row count**: PASS (expected=569, actual=569)
- **Column count**: PASS (expected=31, actual=31)
- **Numeric feature count**: PASS (expected=31, actual=31)
- **Categorical feature count**: PASS (expected=0, actual=0)
  - EDA categorical list unavailable; actual count recorded.
- **Missing values computed**: PASS (expected=>= 0, actual=0)
- **Duplicate rows computed**: PASS (expected=>= 0, actual=0)

### Problem Type — PASS
- **Problem type matches target**: PASS (expected=classification, actual=classification)
  - Target column: target

### Model Metrics — FAIL
- **Primary metric (accuracy)**: PASS (expected=0.9713750435691878, actual=0.9649122807017544)
- **ACCURACY**: FAIL (expected=0.37719298245614036, actual=0.9649122807017544)
- **PRECISION**: FAIL (expected=0.18859649122807018, actual=0.9652053622194477)
- **RECALL**: FAIL (expected=0.5, actual=0.9649122807017544)
- **F1**: FAIL (expected=0.27388535031847133, actual=0.9647382344750765)
- **ROC_AUC**: FAIL (expected=0.5845070422535211, actual=0.9580740255486406)

### Best Model Selection — PASS
- **Best model has highest score**: PASS (expected=AdaBoost, actual=AdaBoost)

### Explainability — PASS
- **Feature importance exists**: PASS (expected=non-empty, actual=30)
- **Importance values numeric**: PASS (expected=numeric, actual=True)
- **Top features exist in dataset**: PASS (expected=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'], actual=['area error', 'compactness error', 'concave points error', 'concavity error', 'fractal dimension error', 'mean area', 'mean compactness', 'mean concave points', 'mean concavity', 'mean fractal dimension', 'mean perimeter', 'mean radius', 'mean smoothness', 'mean symmetry', 'mean texture', 'perimeter error', 'radius error', 'smoothness error', 'symmetry error', 'target', 'texture error', 'worst area', 'worst compactness', 'worst concave points', 'worst concavity', 'worst fractal dimension', 'worst perimeter', 'worst radius', 'worst smoothness', 'worst symmetry', 'worst texture'])

### Report Accuracy — PASS
- **Dataset name**: PASS (expected=Heart-Disease.csv, actual=Heart-Disease.csv)
- **Rows**: PASS (expected=569, actual=569)
- **Columns**: PASS (expected=31, actual=31)
- **Best model**: PASS (expected=AdaBoost, actual=AdaBoost)
- **Primary metric in report**: PASS (expected=0.9713750435691878, actual=0.9714)
