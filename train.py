"""Train and evaluate credit-risk classifiers on OpenML's German Credit data."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
    mapping = {"good": "good", "bad": "bad", "1": "good", "2": "bad"}
    y = y.map(mapping)
    if y.isna().any():
        raise ValueError("Unexpected target labels in dataset. Inspect OpenML target values.")
    return X, y


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

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

    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        class_index = list(pipeline.named_steps["classifier"].classes_).index("bad")
        bad_probability = pipeline.predict_proba(X_test)[:, class_index]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "bad_precision": float(precision_score(y_test, pred, pos_label="bad", zero_division=0)),
            "bad_recall": float(recall_score(y_test, pred, pos_label="bad", zero_division=0)),
            "bad_f1": float(f1_score(y_test, pred, pos_label="bad", zero_division=0)),
            "roc_auc": float(roc_auc_score((y_test == "bad").astype(int), bad_probability)),
        }
        fitted[name] = pipeline
        print(f"{name}: accuracy={results[name]['accuracy']:.3f}, "
              f"bad_recall={results[name]['bad_recall']:.3f}, "
              f"ROC-AUC={results[name]['roc_auc']:.3f}")

    # The dataset's documented cost matrix makes missed bad-risk applicants costly.
    # Select using bad-class recall, then ROC-AUC as a tie-breaker.
    best_name = max(results, key=lambda n: (results[n]["bad_recall"], results[n]["roc_auc"]))
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)
    bad_idx = list(best_model.named_steps["classifier"].classes_).index("bad")
    bad_probs = best_model.predict_proba(X_test)[:, bad_idx]

    artifact = {
        "pipeline": best_model,
        "feature_columns": X.columns.tolist(),
        "classes": list(best_model.named_steps["classifier"].classes_),
        "selected_model": best_name,
    }
    joblib.dump(artifact, MODEL_DIR / "credit_scoring_model.joblib")

    metadata = {}
    for col in X.columns:
        if col in numeric_features:
            series = pd.to_numeric(X[col], errors="coerce")
            low, high = float(series.min()), float(series.max())
            metadata[col] = {
                "type": "numeric", "min": low, "max": high,
                "median": float(series.median()),
                "integer": bool(pd.api.types.is_integer_dtype(X[col])),
            }
        else:
            values = sorted(X[col].dropna().astype(str).unique().tolist())
            metadata[col] = {"type": "categorical", "options": values}
    (REPORT_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    summary = {
        "dataset": "OpenML credit-g (Statlog German Credit)",
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "good_applicants": int((y == "good").sum()),
        "bad_applicants": int((y == "bad").sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_model": best_name,
        "model_comparison": results,
        "classification_report": classification_report(
            y_test, y_pred, labels=["good", "bad"], output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": ["good", "bad"],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["good", "bad"]).tolist(),
        "roc_auc": float(roc_auc_score((y_test == "bad").astype(int), bad_probs)),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'credit_scoring_model.joblib'}")
    print(f"Saved metrics: {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
