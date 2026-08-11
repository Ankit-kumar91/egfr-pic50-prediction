"""Regression metrics and applicability-domain (AD) checks."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R^2 and Spearman rho.

    Reported together rather than R^2 alone: R^2 can look strong on a
    narrow pIC50 range even when the model is off by a log unit, while
    Spearman rho checks whether the predicted ranking of compounds (what
    matters for prioritizing which to synthesize) is preserved.
    """
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }


def morgan_fingerprints(smiles_list, radius: int = 2, n_bits: int = 2048):
    """Morgan (ECFP-like) fingerprints used as the similarity representation
    for applicability domain checks.

    Public (not module-private) since callers that score many queries
    against the same training set, e.g. src/pipeline/predict_pipeline.py,
    should compute the training fingerprints once and reuse them rather
    than recomputing on every applicability_domain() call.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def applicability_domain(
    query_smiles, train_smiles, threshold: float = 0.4
) -> np.ndarray:
    """Max Tanimoto similarity of each query molecule to the training set.

    A query molecule structurally far from anything the model was trained
    on (max similarity below `threshold`) is outside the model's applicability
    domain: the prediction may not be reliable, so it should carry a warning
    rather than be silently trusted. This function only flags the condition;
    the caller decides what to do with the flag (never a hard rejection, since
    a low-similarity prediction can still be useful context).

    Returns (max_similarity, in_domain) arrays, one value per query molecule.
    """
    train_fps = morgan_fingerprints(train_smiles)
    query_fps = morgan_fingerprints(query_smiles)
    max_sim = np.array(
        [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in query_fps]
    )
    return max_sim, max_sim >= threshold
