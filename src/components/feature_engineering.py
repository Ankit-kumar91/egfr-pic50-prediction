"""SMILES -> features for classical ML: ECFP fingerprints and RDKit descriptors."""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Compact, standard drug-discovery descriptor set (avoids Descriptors.CalcMolDescriptors's
# ~200 columns, most of which are redundant with each other for a QSAR model this size).
_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}


def ecfp_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (ECFP) fingerprint as a dense bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    return np.array([f(mol) for f in _DESCRIPTOR_FUNCS.values()], dtype=float)


def featurize(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    kind: str = "ecfp",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """kind: 'ecfp', 'descriptors', or 'both'."""
    smiles = df[smiles_col].tolist()
    parts = []
    if kind in ("ecfp", "both"):
        parts.append(np.array([ecfp_fingerprint(s, radius, n_bits) for s in smiles]))
    if kind in ("descriptors", "both"):
        parts.append(np.array([rdkit_descriptors(s) for s in smiles]))
    if not parts:
        raise ValueError(f"Unknown feature kind: {kind}")
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def descriptor_names(kind: str, n_bits: int = 2048) -> list[str]:
    names = []
    if kind in ("ecfp", "both"):
        names += [f"ecfp_{i}" for i in range(n_bits)]
    if kind in ("descriptors", "both"):
        names += list(_DESCRIPTOR_FUNCS)
    return names
