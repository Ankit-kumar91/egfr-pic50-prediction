"""Regression metrics and applicability-domain (AD) checks."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R², Spearman rho — always reported together per project convention."""
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }


def _morgan_fps(smiles_list, radius: int = 2, n_bits: int = 2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def applicability_domain(
    query_smiles, train_smiles, threshold: float = 0.4
) -> np.ndarray:
    """Max Tanimoto similarity of each query molecule to the training set.

    Per CLAUDE.md: below `threshold` means the prediction should carry an
    AD warning, not a hard rejection — caller decides what to do with the flag.
    Returns (max_similarity, in_domain) arrays.
    """
    train_fps = _morgan_fps(train_smiles)
    query_fps = _morgan_fps(query_smiles)
    max_sim = np.array(
        [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in query_fps]
    )
    return max_sim, max_sim >= threshold
