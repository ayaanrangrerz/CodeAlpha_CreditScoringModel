"""Smoke tests for the trained CreditLens artifact and inference pipeline."""
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"
METADATA_PATH = ROOT / "reports" / "feature_metadata.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"


class ModelSmokeTest(unittest.TestCase):
    def test_training_outputs_exist(self):
        for path in (MODEL_PATH, METADATA_PATH, METRICS_PATH):
            self.assertTrue(path.exists(), f"Missing {path.name}; run python train.py first.")

    def test_pipeline_predicts_one_sample(self):
        artifact = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text())
        sample = {}
        for feature in artifact["feature_columns"]:
            meta = metadata[feature]
            sample[feature] = (
                meta["options"][0] if meta["type"] == "categorical" else meta["median"]
            )
        frame = pd.DataFrame([sample], columns=artifact["feature_columns"])
        prediction = artifact["pipeline"].predict(frame)
        probabilities = artifact["pipeline"].predict_proba(frame)
        self.assertEqual(len(prediction), 1)
        self.assertIn(str(prediction[0]), {"good", "bad"})
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_metrics_document_holdout_and_cv(self):
        metrics = json.loads(METRICS_PATH.read_text())
        self.assertIn("cross_validation_comparison", metrics)
        self.assertIn("holdout_test_metrics", metrics)
        self.assertEqual(metrics["test_rows"], 200)


if __name__ == "__main__":
    unittest.main()
