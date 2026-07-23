"""Generate scaffold and random train/val/test splits for the curated dataset.

Scaffold split (primary, per CLAUDE.md methodology rules): groups compounds by
Bemis-Murcko scaffold, then greedily assigns whole scaffold groups (largest
first) to train/val/test so no scaffold leaks across splits.

Random split is generated only for contrast against the scaffold split.
"""

from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    scaffolds: dict[str, list[int]] = {}
    for idx, smiles in df["smiles"].items():
        scaffolds.setdefault(_scaffold(smiles), []).append(idx)

    # Largest scaffold groups first, tie-broken deterministically by seed shuffle.
    groups = list(scaffolds.values())
    rng = __import__("random").Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n = len(df)
    n_train, n_val = int(train_frac * n), int(val_frac * n)

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def random_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    train, rest = train_test_split(df, train_size=train_frac, random_state=seed)
    val_size = val_frac / (1 - train_frac)
    val, test = train_test_split(rest, train_size=val_size, random_state=seed)
    return train, val, test


def main() -> None:
    config = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
    seed = config["seed"]
    train_frac = config["split"]["train_frac"]
    val_frac = config["split"]["val_frac"]

    df = pd.read_csv(_ROOT / config["data"]["curated_csv"])
    out_dir = _ROOT / config["data"]["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, splitter in [("scaffold", scaffold_split), ("random", random_split)]:
        train, val, test = splitter(df, train_frac, val_frac, seed)
        for part_name, part in [("train", train), ("val", val), ("test", test)]:
            path = out_dir / f"{name}_{part_name}.csv"
            part.to_csv(path, index=False)
            print(f"{path.relative_to(_ROOT)}: {len(part)} rows")


if __name__ == "__main__":
    main()
