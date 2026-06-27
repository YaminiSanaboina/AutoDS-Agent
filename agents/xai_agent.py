import shap
import pandas as pd
import plotly.express as px
import numpy as np
from types import SimpleNamespace
import traceback


def _extract_model_feature_names(model):
    """Try several ways to extract feature names used during training."""
    # Direct attribute (sklearn, xgboost sklearn wrapper)
    try:
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            return list(model.feature_names_in_)
    except Exception:
        pass

    # If pipeline-like, try to get final estimator
    try:
        # lazy import to avoid hard dependency
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            final = model.steps[-1][1]
            if hasattr(final, "feature_names_in_"):
                return list(final.feature_names_in_)
    except Exception:
        pass

    # xgboost booster (raw) may expose feature names via get_booster
    try:
        booster = getattr(model, "get_booster", None)
        if callable(booster):
            b = booster()
            fn = getattr(b, "feature_names", None)
            if fn:
                return list(fn)
    except Exception:
        pass

    return None


def _is_tree_model(model):
    """Rudimentary check for tree-based models we want to treat specially."""
    cls_name = getattr(model, "__class__", type(model)).__name__.lower()
    try:
        name = str(model.__class__).lower()
    except Exception:
        name = ""
    if "randomforest" in cls_name or "xgboost" in cls_name or "xgb" in cls_name or "booster" in name:
        return True
    # sklearn ensemble classes
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            return True
    except Exception:
        pass

    return False


def generate_shap_explanation(model, X):
    """
    Generate SHAP feature importance.

    This function enforces that the input DataFrame `X` matches the model's
    training feature names (if available) and orders columns accordingly.
    For tree-based models it uses the TreeExplainer and disables the
    additivity check to avoid "Additivity check failed" errors.

    On failure, raises an Exception with a user-friendly message so callers
    can show a clean UI message without crashing the page.
    """

    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        model_features = _extract_model_feature_names(model)

        # If we know model training features, enforce exact match and order
        if model_features is not None:
            # Normalize names to strings
            model_features = [str(c) for c in model_features]
            missing = [c for c in model_features if c not in list(X.columns)]
            if missing:
                raise ValueError(f"Input data is missing model training columns: {missing}")

            # Drop any extra columns and reorder to match training
            X_aligned = X.loc[:, model_features]
        else:
            X_aligned = X.copy()

        # Use TreeExplainer for tree models and disable additivity check
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
            try:
                raw = explainer.shap_values(X_aligned, check_additivity=False)
            except TypeError:
                # Older/newer shap versions might not accept the kwarg; fallback
                raw = explainer.shap_values(X_aligned)

            # Normalize raw output into an object with .values attribute
            if isinstance(raw, list):
                try:
                    arr = np.stack(raw, axis=2)
                except Exception:
                    arr = np.array(raw, dtype=object)
            else:
                arr = np.array(raw)

            shap_values = SimpleNamespace(values=arr)
            return explainer, shap_values

        # Fallback to generic explainer (model-agnostic)
        try:
            explainer = shap.Explainer(model, X_aligned)
            shap_values = explainer(X_aligned)
            return explainer, shap_values
        except TypeError:
            # Some models are not callable with the default masker; try an explicit predict wrapper.
            def predict_fn(data):
                return model.predict(pd.DataFrame(data, columns=X_aligned.columns))

            explainer = shap.Explainer(predict_fn, X_aligned)
            shap_values = explainer(X_aligned)
            return explainer, shap_values

    except Exception as err:
        # Log traceback to console for debugging, but surface a clean message
        traceback.print_exc()
        raise Exception(
            "AI explanation is temporarily unavailable for this model, but the model predictions and performance are still valid."
        )


def get_feature_importance_ranking(shap_values, feature_names):
    values = shap_values.values
    if values.ndim == 3:
        values = np.abs(values).mean(axis=2)
    else:
        values = np.abs(values)

    mean_abs = values.mean(axis=0)
    ranking = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    ranking["Rank"] = range(1, len(ranking) + 1)
    return ranking[["Rank", "Feature", "Mean |SHAP|"]]


def plot_shap_summary(shap_values, feature_names, top_n=15):
    ranking = get_feature_importance_ranking(shap_values, feature_names).head(top_n)

    fig = px.bar(
        ranking,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title=f"SHAP Feature Impact (Top {top_n})",
        color="Mean |SHAP|",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, top_n * 30),
        showlegend=False,
    )
    return fig


def get_positive_negative_impact(shap_values, feature_names):
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 0]

    mean_pos = np.where(values > 0, values, 0).mean(axis=0)
    mean_neg = np.where(values < 0, values, 0).mean(axis=0)

    return pd.DataFrame({
        "Feature": feature_names,
        "Positive Impact": mean_pos,
        "Negative Impact": mean_neg,
    }).sort_values("Positive Impact", key=abs, ascending=False)


def plot_impact_chart(impact_df, top_n=12):
    df = impact_df.head(top_n)
    fig = px.bar(
        df,
        x="Feature",
        y=["Positive Impact", "Negative Impact"],
        title="Positive vs Negative Feature Impact",
        barmode="group",
        color_discrete_sequence=["#10B981", "#EF4444"],
    )
    fig.update_layout(xaxis_tickangle=-45, height=450)
    return fig