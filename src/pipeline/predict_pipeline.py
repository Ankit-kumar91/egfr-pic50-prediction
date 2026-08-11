"""Shared inference pipeline for the deployed web app (api/ and streamlit app.py).

Loads the three models the app exposes -- Random Forest, the CheMeleon
fine-tune, and the tuned Chemprop D-MPNN -- and predicts pIC50 with an
uncertainty interval and an applicability domain flag for any subset of
them. Both the FastAPI backend and the Streamlit UI import PredictionPipeline
directly rather than duplicating this logic.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem, DataStructs

from src.components.feature_engineering import featurize
from src.components.model_evaluator import morgan_fingerprints

# Not importing model_trainer: it pulls in xgboost/lightgbm/torch_geometric
# at import time, which this inference-only path doesn't need.

_ROOT = Path(__file__).resolve().parents[2]

# "chemprop" here is the tuned D-MPNN (models/dmpnn), not the untuned
# default (models/chemprop) -- the one reported in the comparison results.
MODEL_PATHS = {
    "random_forest": _ROOT / "models" / "random_forest_scaffold.joblib",
    "chemeleon": _ROOT / "models" / "chemeleon" / "scaffold" / "model_0" / "best.pt",
    "chemprop": _ROOT / "models" / "dmpnn" / "scaffold" / "model_0" / "best.pt",
}
CHEMPROP_MODELS = {"chemeleon", "chemprop"}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "chemeleon": "CheMeleon",
    "chemprop": "Chemprop D-MPNN",
}

# RMSE values from results/model_comparison_scaffold_test.csv, hardcoded so
# the sidebar doesn't need to reload the test set to describe the models.
MODEL_DESCRIPTIONS = {
    "random_forest": {
        "tagline": "Classical ML on molecular fingerprints",
        "detail": (
            "Random Forest over ECFP fingerprints and RDKit descriptors. "
            "The best performer on this dataset -- fixed fingerprints "
            "still beat learned graph representations at this data scale."
        ),
        "rmse": 0.889,
    },
    "chemeleon": {
        "tagline": "Fine-tuned chemical foundation model",
        "detail": (
            "A Chemprop D-MPNN initialized from CheMeleon, pretrained on "
            "roughly 1 million PubChem molecules, then fine-tuned on this "
            "EGFR dataset."
        ),
        "rmse": 0.934,
    },
    "chemprop": {
        "tagline": "Directed message passing neural network",
        "detail": (
            "A Chemprop D-MPNN trained from random initialization on this "
            "dataset only, with hyperparameters tuned via Ray Tune."
        ),
        "rmse": 0.989,
    },
}

TRAIN_SMILES_PATH = _ROOT / "data" / "splits" / "scaffold_train.csv"
AD_THRESHOLD = (
    0.4  # matches configs/config.yaml: applicability_domain.tanimoto_threshold
)

# The chemprop console script isn't necessarily on PATH (e.g. if this
# process wasn't launched from a fully activated env) -- resolve it next
# to the current interpreter instead.
_CHEMPROP_CLI = Path(sys.executable).with_name("chemprop")
if not _CHEMPROP_CLI.exists():
    _CHEMPROP_CLI = "chemprop"  # fall back to PATH lookup


class PredictionPipeline:
    """Loads model artifacts lazily and caches them across calls."""

    def __init__(self) -> None:
        self._rf_model = None
        self._train_fps = None

    def available_models(self) -> list[str]:
        return [name for name, path in MODEL_PATHS.items() if path.exists()]

    def _rf(self):
        if self._rf_model is None:
            self._rf_model = joblib.load(MODEL_PATHS["random_forest"])
        return self._rf_model

    def _train_fingerprints(self):
        if self._train_fps is None:
            train_smiles = pd.read_csv(TRAIN_SMILES_PATH)["smiles"].tolist()
            self._train_fps = morgan_fingerprints(train_smiles)
        return self._train_fps

    def _applicability_domain(self, smiles_list: list[str]) -> list[dict]:
        query_fps = morgan_fingerprints(smiles_list)
        train_fps = self._train_fingerprints()
        out = []
        for fp in query_fps:
            similarity = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            out.append(
                {
                    "nearest_neighbor_similarity": similarity,
                    "in_domain": similarity >= AD_THRESHOLD,
                    "threshold": AD_THRESHOLD,
                }
            )
        return out

    def _predict_random_forest(self, smiles_list: list[str]) -> list[dict]:
        df = pd.DataFrame({"smiles": smiles_list})
        X = featurize(df, kind="ecfp", n_bits=2048)
        y_pred, y_pis = self._rf().predict_interval(X)  # mapie SplitConformalRegressor
        y_lo, y_hi = y_pis[:, 0, 0], y_pis[:, 1, 0]
        return [
            {"pic50": float(p), "lower": float(lo), "upper": float(hi)}
            for p, lo, hi in zip(y_pred, y_lo, y_hi)
        ]

    def _predict_chemprop(self, smiles_list: list[str], checkpoint: Path) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            in_csv, out_csv = tmp / "query.csv", tmp / "preds.csv"
            pd.DataFrame({"smiles": smiles_list}).to_csv(in_csv, index=False)
            # Same `chemprop predict` CLI used at training time, rather than
            # chemprop's internal inference API, which changes across versions.
            subprocess.run(
                [
                    str(_CHEMPROP_CLI),
                    "predict",
                    "-i",
                    str(in_csv),
                    "-s",
                    "smiles",
                    "--model-paths",
                    str(checkpoint),
                    "--uncertainty-method",
                    "mve",
                    "-o",
                    str(out_csv),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            preds = pd.read_csv(out_csv)
        results = []
        for pic50, variance in zip(preds["pIC50"], preds["pIC50_unc"]):
            std = float(variance) ** 0.5
            results.append(
                {
                    "pic50": float(pic50),
                    "lower": float(pic50 - 1.96 * std),
                    "upper": float(pic50 + 1.96 * std),
                }
            )
        return results

    def predict_batch(
        self, smiles_list: list[str], model_names: list[str]
    ) -> list[dict]:
        """Predict pIC50 for every SMILES with every requested model.

        Invalid SMILES are reported per-row (`error` key) instead of
        failing the whole batch, so one bad row in an uploaded CSV doesn't
        block the rest.
        """
        unknown = set(model_names) - set(MODEL_PATHS)
        if unknown:
            raise ValueError(f"Unknown model name(s): {sorted(unknown)}")

        valid_mask = [Chem.MolFromSmiles(s) is not None for s in smiles_list]
        valid_smiles = [s for s, ok in zip(smiles_list, valid_mask) if ok]

        ad = self._applicability_domain(valid_smiles) if valid_smiles else []
        per_model: dict[str, list[dict]] = {}
        for name in model_names:
            if not valid_smiles:
                per_model[name] = []
            elif name == "random_forest":
                per_model[name] = self._predict_random_forest(valid_smiles)
            else:
                per_model[name] = self._predict_chemprop(
                    valid_smiles, MODEL_PATHS[name]
                )

        results = []
        valid_i = 0
        for smiles, ok in zip(smiles_list, valid_mask):
            if not ok:
                results.append(
                    {"smiles": smiles, "error": "RDKit could not parse this SMILES"}
                )
                continue
            results.append(
                {
                    "smiles": smiles,
                    "applicability_domain": ad[valid_i],
                    "predictions": {
                        name: per_model[name][valid_i] for name in model_names
                    },
                }
            )
            valid_i += 1
        return results

    def predict(self, smiles: str, model_names: list[str]) -> dict:
        """Single-molecule convenience wrapper around predict_batch."""
        return self.predict_batch([smiles], model_names)[0]
