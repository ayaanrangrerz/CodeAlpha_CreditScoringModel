"""Train and evaluate credit-risk classifiers on OpenML's German Credit data."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the classic 1,000-row German Credit dataset from OpenML."""
    print("Downloading/loading OpenML dataset: credit-g ...")
    bunch = fetch_openml(name="credit-g", version=1, as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.astype(str).str.strip().str.lower()
    y = y.map({"good": "good", "bad": "bad", "1": "good", "2": "bad"})
    if y.isna().any():
        raise ValueError("Unexpected target labels in dataset.")
    return X, y


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """Build preprocessing inside the pipeline to prevent data leakage."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def bad_class_roc_auc(estimator, features, labels) -> float:
    """Calculate ROC-AUC with the bad-risk class explicitly treated as positive."""
    bad_index = list(estimator.classes_).index("bad")
    probabilities = estimator.predict_proba(features)[:, bad_index]
    return float(roc_auc_score((labels == "bad").astype(int), probabilities))


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "bad_recall": make_scorer(recall_score, pos_label="bad", zero_division=0),
        "bad_roc_auc": bad_class_roc_auc,
    }
    results = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(X_train, estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, return_train_score=False
        )
        results[name] = {
            "cv_bad_recall_mean": float(scores["test_bad_recall"].mean()),
            "cv_bad_recall_std": float(scores["test_bad_recall"].std()),
            "cv_bad_roc_auc_mean": float(scores["test_bad_roc_auc"].mean()),
            "cv_bad_roc_auc_std": float(scores["test_bad_roc_auc"].std()),
        }
        print(f"{name}: CV bad recall={results[name]['cv_bad_recall_mean']:.3f}, "
              f"CV ROC-AUC={results[name]['cv_roc_auc_mean']:.3f}")

    # Select using training-only cross-validation; keep the holdout test set untouched.
    best_name = max(
        results,
        key=lambda name: (
            results[name]["cv_bad_recall_mean"],
            results[name]["cv_bad_roc_auc_mean"],
        ),
    )
    best_model = build_pipeline(X_train, candidates[best_name])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    bad_idx = list(best_model.named_steps["classifier"].classes_).index("bad")
    bad_probs = best_model.predict_proba(X_test)[:, bad_idx]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "bad_precision": float(precision_score(y_test, y_pred, pos_label="bad", zero_division=0)),
        "bad_recall": float(recall_score(y_test, y_pred, pos_label="bad", zero_division=0)),
        "bad_f1": float(f1_score(y_test, y_pred, pos_label="bad", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "bad").astype(int), bad_probs)),
    }

    joblib.dump({
        "pipeline": best_model,
        "feature_columns": X.columns.tolist(),
        "classes": list(best_model.named_steps["classifier"].classes_),
        "selected_model": best_name,
    }, MODEL_DIR / "credit_scoring_model.joblib")

    metadata = {}
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    for column in X.columns:
        if column in numeric_features:
            series = pd.to_numeric(X[column], errors="coerce")
            metadata[column] = {
                "type": "numeric", "min": float(series.min()),
                "max": float(series.max()), "median": float(series.median()),
                "integer": bool(pd.api.types.is_integer_dtype(X[column])),
            }
        else:
            metadata[column] = {
                "type": "categorical",
                "options": sorted(X[column].dropna().astype(str).unique().tolist()),
            }
    (REPORT_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    summary = {
        "dataset": "OpenML credit-g (Statlog German Credit)",
        "rows": int(len(X)), "features": int(X.shape[1]),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "selection_method": "5-fold stratified cross-validation on training split",
        "selection_priority": "mean bad-class recall, then mean ROC-AUC",
        "selected_model": best_name,
        "cross_validation_comparison": results,
        "holdout_test_metrics": test_metrics,
        "classification_report": classification_report(
            y_test, y_pred, labels=["good", "bad"], output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": ["good", "bad"],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["good", "bad"]).tolist(),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSelected model: {best_name}")
    print(f"Holdout test metrics: {test_metrics}")
    print(f"Saved model: {MODEL_DIR / 'credit_scoring_model.joblib'}")


if __name__ == "__main__":
    main()
