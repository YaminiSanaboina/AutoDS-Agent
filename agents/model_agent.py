import time

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import LabelEncoder



from sklearn.ensemble import (

    RandomForestClassifier,

    RandomForestRegressor,

    ExtraTreesClassifier,

    ExtraTreesRegressor,

    GradientBoostingClassifier,

    GradientBoostingRegressor,

    AdaBoostClassifier,

)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import (

    LogisticRegression,

    LinearRegression,

    Ridge,

    Lasso,

    ElasticNet,

)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    r2_score,

    mean_absolute_error,

    mean_squared_error,

    confusion_matrix,

    roc_curve,

    auc,

)

from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC, SVR



try:

    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True

except ImportError:

    HAS_XGB = False



try:

    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True

except ImportError:

    HAS_LGBM = False



try:

    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True

except ImportError:

    HAS_CATBOOST = False
import pickle as _pickle_module
import json as _json_module
import os as _os_module
import tempfile as _tempfile_module
import multiprocessing as _mp_module


def _model_worker_fit(pickled_model_bytes, X_tr, y_tr, X_te, y_te, prob_type, out_model_path, out_metrics_path):
    """Top-level worker for multiprocessing spawn on Windows. Receives a pickled model bytes."""
    try:
        model_obj = _pickle_module.loads(pickled_model_bytes)
    except Exception:
        try:
            with open(out_metrics_path, "w", encoding="utf-8") as jf:
                _json_module.dump({"error": "unpickle_failed"}, jf)
        except Exception:
            pass
        return

    try:
        model_obj.fit(X_tr, y_tr)
        if prob_type == "Classification":
            metrics, _ = _evaluate_classification(model_obj, X_te, y_te)
        else:
            metrics, _ = _evaluate_regression(model_obj, X_te, y_te)
        try:
            cv_scoring = "accuracy" if prob_type == "Classification" else "r2"
            from sklearn.model_selection import cross_val_score as _cv_score

            cv_scores = _cv_score(model_obj, X_tr, y_tr, cv=3, scoring=cv_scoring)
            metrics["cv_score"] = float(np.mean(cv_scores))
        except Exception:
            pass
        try:
            with open(out_model_path, "wb") as mf:
                _pickle_module.dump(model_obj, mf)
        except Exception:
            pass
        try:
            with open(out_metrics_path, "w", encoding="utf-8") as jf:
                _json_module.dump({"metrics": metrics}, jf)
        except Exception:
            pass
    except Exception:
        try:
            with open(out_metrics_path, "w", encoding="utf-8") as jf:
                _json_module.dump({"error": "fit_failed"}, jf)
        except Exception:
            pass



from utils.model_selection_explainer import build_model_selection_explanation





def preprocess_data(df, target_column):

    """Prepare dataset for machine learning. Returns X, y, encoder."""

    data = df.copy()

    data = data.dropna(subset=[target_column])

    X = data.drop(columns=[target_column])

    y = data[target_column]



    for column in X.columns:

        if pd.api.types.is_numeric_dtype(X[column]):

            X[column] = X[column].fillna(X[column].median())

        else:

            X[column] = X[column].fillna("Unknown")

            encoder = LabelEncoder()

            try:

                X[column] = encoder.fit_transform(X[column].astype(str))

            except Exception:

                X[column] = X[column].astype(str)



    target_encoder = None

    try:

        if not pd.api.types.is_numeric_dtype(y):

            unique_vals = list(map(lambda v: str(v).strip().lower(), pd.Series(y).dropna().unique()))

            binary_like = set(unique_vals) <= set(["yes", "no", "true", "false", "y", "n", "t", "f", "positive", "negative"]) or len(unique_vals) <= 2

            target_encoder = LabelEncoder()

            if binary_like or len(unique_vals) <= 10:

                y = target_encoder.fit_transform(y.astype(str))

            else:

                y = target_encoder.fit_transform(y.astype(str))

        else:

            target_encoder = None

    except Exception:

        try:

            y = pd.to_numeric(y, errors="coerce")

            y = y.fillna(0)

        except Exception:

            pass



    return X, y, target_encoder





def detect_problem_type(y):
    """Detect whether it is a classification or regression problem."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    # If the target is categorical or text-like, treat as Classification.
    if pd is not None and isinstance(y, (pd.Series, pd.Categorical)):
        if not pd.api.types.is_numeric_dtype(y):
            return "Classification"

    try:
        values = list(y)
    except Exception:
        values = []

    if not values:
        return "Classification"

    try:
        unique_values = len(set(values))
    except Exception:
        unique_values = len(values)

    # Numeric targets with low cardinality are likely classification, otherwise regression.
    if unique_values <= 10:
        return "Classification"

    try:
        non_null = [v for v in values if v is not None and (not isinstance(v, float) or not pd.isna(v) if pd is not None else True)]
        unique_ratio = unique_values / max(len(non_null), 1)
    except Exception:
        unique_ratio = 1.0

    if unique_ratio < 0.05:
        return "Classification"

    return "Regression"





def _evaluate_classification(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {

        "accuracy": float(accuracy_score(y_test, predictions)),

    }

    average = "binary" if len(set(y_test)) == 2 else "weighted"

    try:

        metrics["precision"] = float(precision_score(y_test, predictions, average=average, zero_division=0))

        metrics["recall"] = float(recall_score(y_test, predictions, average=average, zero_division=0))

        metrics["f1"] = float(f1_score(y_test, predictions, average=average, zero_division=0))

    except Exception:

        metrics["precision"] = metrics["recall"] = metrics["f1"] = 0.0



    try:

        if hasattr(model, "predict_proba") and len(set(y_test)) == 2:

            proba = model.predict_proba(X_test)[:, 1]

            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

        elif hasattr(model, "predict_proba") and len(set(y_test)) > 2:

            proba = model.predict_proba(X_test)

            metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))

    except Exception:

        pass



    return metrics, predictions





def _evaluate_regression(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    return {

        "r2": float(r2_score(y_test, predictions)),

        "rmse": rmse,

        "mae": float(mean_absolute_error(y_test, predictions)),

    }, predictions





def get_model_registry(problem_type, smart_mode: bool = False):

    """Return candidate models for the detected problem type."""

    fast_n = 30 if smart_mode else 100



    if problem_type == "Classification":

        registry = {

            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),

            "Decision Tree": DecisionTreeClassifier(random_state=42),

            "Random Forest": RandomForestClassifier(n_estimators=fast_n, random_state=42),

            "Extra Trees": ExtraTreesClassifier(n_estimators=fast_n, random_state=42),

            "Gradient Boosting": GradientBoostingClassifier(random_state=42),

            "AdaBoost": AdaBoostClassifier(random_state=42),

            "KNN": KNeighborsClassifier(n_neighbors=5),

            "Naive Bayes": GaussianNB(),

            "SVM": SVC(probability=True, random_state=42),

        }

        # smart_mode uses smaller ensembles but keeps the full candidate registry

        if HAS_XGB:

            registry["XGBoost"] = XGBClassifier(

                n_estimators=fast_n, random_state=42, eval_metric="logloss", verbosity=0

            )

        if HAS_LGBM:

            registry["LightGBM"] = LGBMClassifier(

                n_estimators=fast_n, random_state=42, verbosity=-1

            )

        if HAS_CATBOOST:

            registry["CatBoost"] = CatBoostClassifier(

                iterations=fast_n, random_state=42, verbose=False, allow_writing_files=False

            )

    else:

        registry = {

            "Linear Regression": LinearRegression(),

            "Ridge": Ridge(random_state=42),

            "Lasso": Lasso(random_state=42, max_iter=2000),

            "ElasticNet": ElasticNet(random_state=42, max_iter=2000),

            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),

            "Random Forest Regressor": RandomForestRegressor(n_estimators=fast_n, random_state=42),

            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=fast_n, random_state=42),

            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),

            "SVM": SVR(),

        }

        # smart_mode uses smaller ensembles but keeps the full candidate registry

        if HAS_XGB:

            registry["XGBoost Regressor"] = XGBRegressor(

                n_estimators=fast_n, random_state=42, verbosity=0

            )

        if HAS_LGBM:

            registry["LightGBM Regressor"] = LGBMRegressor(

                n_estimators=fast_n, random_state=42, verbosity=-1

            )

        if HAS_CATBOOST:

            registry["CatBoost Regressor"] = CatBoostRegressor(

                iterations=fast_n, random_state=42, verbose=False, allow_writing_files=False

            )

    return registry





def train_models(X, y, problem_type):

    """Train legacy 3-model subset (backward compatibility)."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    trained_models = {}



    if problem_type == "Classification":

        models = {

            "Random Forest": RandomForestClassifier(),

            "Decision Tree": DecisionTreeClassifier(),

            "Logistic Regression": LogisticRegression(max_iter=1000),

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            trained_models[name] = model

            metrics, _ = _evaluate_classification(model, X_test, y_test)

            results[name] = metrics["accuracy"]

    else:

        models = {

            "Random Forest Regressor": RandomForestRegressor(),

            "Decision Tree Regressor": DecisionTreeRegressor(),

            "Linear Regression": LinearRegression(),

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            trained_models[name] = model

            metrics, _ = _evaluate_regression(model, X_test, y_test)

            results[name] = metrics["r2"]



    best_model_name = max(results, key=results.get)

    best_model_object = trained_models[best_model_name]

    return results, best_model_name, best_model_object





def train_selected_models(X, y, problem_type, selected_models=None, progress_callback=None, max_seconds=None, smart_mode=False):

    """Train candidate models with extended metrics and leaderboard data."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    registry = get_model_registry(problem_type, smart_mode=smart_mode)

    if selected_models:

        models = {k: v for k, v in registry.items() if k in selected_models}

    else:

        models = registry



    results = {}

    trained_models = {}

    detailed_metrics: dict = {}
    training_times: dict = {}

    start_time = time.time()

    # Run each model fit/eval in a separate process with a timeout so a single model
    # cannot hang the entire pipeline.
    import tempfile
    import multiprocessing as mp
    import pickle as _pickle
    import json as _json
    import os as _os

    def _worker_fit(model_obj, X_tr, y_tr, X_te, y_te, prob_type, out_model_path, out_metrics_path):
        try:
            model_obj.fit(X_tr, y_tr)
            if prob_type == "Classification":
                metrics, _ = _evaluate_classification(model_obj, X_te, y_te)
            else:
                metrics, _ = _evaluate_regression(model_obj, X_te, y_te)
            try:
                cv_scoring = "accuracy" if prob_type == "Classification" else "r2"
                cv_scores = cross_val_score(model_obj, X_tr, y_tr, cv=3, scoring=cv_scoring)
                metrics["cv_score"] = float(np.mean(cv_scores))
            except Exception:
                pass
            try:
                with open(out_model_path, "wb") as mf:
                    _pickle.dump(model_obj, mf)
            except Exception:
                pass
            try:
                with open(out_metrics_path, "w", encoding="utf-8") as jf:
                    _json.dump({"metrics": metrics}, jf)
            except Exception:
                pass
        except Exception:
            try:
                with open(out_metrics_path, "w", encoding="utf-8") as jf:
                    _json.dump({"error": "fit_failed"}, jf)
            except Exception:
                pass

    for idx, (name, model) in enumerate(models.items()):
        # enforce global budget
        if max_seconds is not None and time.time() - start_time >= max_seconds - 5:
            break

        if progress_callback:
            progress_callback((idx + 1) / max(total, 1), f"Training {name}...")

        t0 = time.time()

        tmpdir = tempfile.mkdtemp(prefix=f"train_{idx}_")
        model_path = _os.path.join(tmpdir, "model.pkl")
        metrics_path = _os.path.join(tmpdir, "metrics.json")

        per_model_timeout = 60
        if max_seconds is not None:
            try:
                if smart_mode:
                    per_model_timeout = min(30, max(5, int(max_seconds / max(len(models), 1))))
                else:
                    per_model_timeout = min(60, max(5, int(max_seconds)))
            except Exception:
                per_model_timeout = 60

        # Use top-level worker and pass pickled model bytes for Windows spawn compatibility
        try:
            pickled_model = _pickle_module.dumps(model)
        except Exception:
            pickled_model = None
        proc = _mp_module.Process(target=_model_worker_fit, args=(pickled_model, X_train, y_train, X_test, y_test, problem_type, model_path, metrics_path))
        proc.start()
        proc.join(timeout=per_model_timeout)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            training_times[name] = round(time.time() - t0, 3)
            detailed_metrics[name] = {"error": f"timeout after {per_model_timeout}s"}
            continue

        # read model and metrics
        try:
            if _os.path.exists(model_path):
                with open(model_path, "rb") as mf:
                    trained = _pickle.load(mf)
                trained_models[name] = trained
            else:
                continue
        except Exception:
            continue

        try:
            if _os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as jf:
                    mdoc = _json.load(jf)
                metrics = mdoc.get("metrics") or {}
            else:
                metrics = {}
        except Exception:
            metrics = {}

        training_times[name] = round(time.time() - t0, 3)

        cv_score = metrics.get("cv_score")
        if cv_score is not None:
            results[name] = float(cv_score)
        else:
            if problem_type == "Classification":
                results[name] = float(metrics.get("accuracy") or 0.0)
            else:
                results[name] = float(metrics.get("r2") or 0.0)

        detailed_metrics[name] = metrics

    if not results:

        raise RuntimeError("No models completed training within the allotted budget.")



    def _composite_score(name: str) -> float:

        test_score = float(results.get(name, 0))

        cv = detailed_metrics.get(name, {}).get("cv_score")

        if cv is not None:

            return 0.7 * test_score + 0.3 * float(cv)

        return test_score



    best_model_name = max(results, key=_composite_score)

    best_model_object = trained_models[best_model_name]



    extras = _compute_extras(best_model_object, X_test, y_test, problem_type, list(X.columns))

    extras["trained_models"] = trained_models

    extras["X_test"] = X_test

    extras["y_test"] = y_test

    extras["detailed_metrics"] = detailed_metrics

    extras["training_times"] = training_times

    extras["model_comparison"] = [

        {"model": name, "metrics": metrics, "training_time": training_times.get(name)}

        for name, metrics in sorted(

            detailed_metrics.items(),

            key=lambda item: _composite_score(item[0]),

            reverse=True,

        )

    ]

    extras["model_selection_explanation"] = build_model_selection_explanation(

        best_model=best_model_name,

        detailed_metrics=detailed_metrics,

        problem_type=problem_type,

        training_times=training_times,

    )

    return results, best_model_name, best_model_object, extras





def _compute_extras(model, X_test, y_test, problem_type, feature_names):

    extras = {}

    predictions = model.predict(X_test)



    if problem_type == "Classification":

        extras["confusion_matrix"] = confusion_matrix(y_test, predictions).tolist()

        if hasattr(model, "predict_proba"):

            try:

                y_proba = model.predict_proba(X_test)

                classes = sorted(set(y_test))

                if len(classes) == 2:

                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])

                    extras["roc_data"] = {

                        "fpr": fpr.tolist(),

                        "tpr": tpr.tolist(),

                        "auc": float(auc(fpr, tpr)),

                    }

            except Exception:

                extras["roc_data"] = None

    else:

        extras["confusion_matrix"] = None

        extras["roc_data"] = None



    if hasattr(model, "feature_importances_"):

        extras["feature_importance"] = dict(zip(feature_names, model.feature_importances_.tolist()))

    elif hasattr(model, "coef_"):

        coef = model.coef_

        if coef.ndim > 1:

            coef = np.abs(coef).mean(axis=0)

        extras["feature_importance"] = dict(zip(feature_names, np.abs(coef).tolist()))

    else:

        extras["feature_importance"] = {}



    return extras


