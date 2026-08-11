"""Generate scaffold and random train/val/test splits for the curated dataset.

Scaffold split (the primary evaluation split): groups compounds by their
Bemis-Murcko scaffold, then greedily assigns whole scaffold groups (largest
first) to train/val/test so no scaffold leaks across splits. This is the
realistic test of generalization for drug discovery, since a real
prospective screen encounters scaffolds the model has never seen, and a
random split would let near-duplicate analogs of the same scaffold appear
in both train and test, inflating the apparent accuracy.

Random split is generated only for contrast against the scaffold split, to
illustrate how much a naive split overstates performance.
"""

from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]


def _scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES: the molecule's ring system with side chains
    stripped off, used as the grouping key so structurally related analogs
    land in the same split."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    # Group row indices by scaffold so every compound sharing a scaffold
    # is assigned to the same split as a unit.
    scaffolds: dict[str, list[int]] = {}
    for idx, smiles in df["smiles"].items():
        scaffolds.setdefault(_scaffold(smiles), []).append(idx)

    # Shuffle first (seeded, for reproducibility) so groups of equal size
    # aren't always ordered the same way, then sort largest first. Filling
    # train with the biggest scaffold groups first keeps small/rare
    # scaffolds concentrated in val/test, which is the harder, more
    # realistic generalization test.
    groups = list(scaffolds.values())
    rng = __import__("random").Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n = len(df)
    n_train, n_val = int(train_frac * n), int(val_frac * n)

    # Greedily fill train, then val, then whatever's left goes to test.
    # Group sizes vary, so the resulting fractions are only approximately
    # train_frac/val_frac, not exact.
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
    """IID random split, generated only as a baseline to contrast against
    the scaffold split above — never the headline result for this project."""
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

    # Write both split strategies so downstream notebooks can load whichever
    # they need; scaffold_test.csv is the one used for headline metrics.
    for name, splitter in [("scaffold", scaffold_split), ("random", random_split)]:
        train, val, test = splitter(df, train_frac, val_frac, seed)
        for part_name, part in [("train", train), ("val", val), ("test", test)]:
            path = out_dir / f"{name}_{part_name}.csv"
            part.to_csv(path, index=False)
            print(f"{path.relative_to(_ROOT)}: {len(part)} rows")


if __name__ == "__main__":
    main()
