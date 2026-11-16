"""Basic checks for the trained model artifact."""
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"


class ModelSmokeTest(unittest.TestCase):
    def test_model_artifact_exists_and_has_expected_keys(self):
        self.assertTrue(MODEL_PATH.exists(), "Run `python train.py` before running tests.")
        artifact = joblib.load(MODEL_PATH)
        self.assertIn("pipeline", artifact)
        self.assertIn("feature_columns", artifact)
        self.assertIn("selected_model", artifact)

    def test_pipeline_predicts_one_sample(self):
        self.assertTrue(MODEL_PATH.exists(), "Run `python train.py` before running tests.")
        artifact = joblib.load(MODEL_PATH)
        metadata_path = ROOT / "reports" / "feature_metadata.json"
        import json
        metadata = json.loads(metadata_path.read_text())
        sample = {}
        for feature in artifact["feature_columns"]:
            meta = metadata[feature]
            sample[feature] = meta["options"][0] if meta["type"] == "categorical" else meta["median"]
        frame = pd.DataFrame([sample], columns=artifact["feature_columns"])
        prediction = artifact["pipeline"].predict(frame)
        self.assertEqual(len(prediction), 1)
        self.assertIn(str(prediction[0]), {"good", "bad"})


if __name__ == "__main__":
    unittest.main()
