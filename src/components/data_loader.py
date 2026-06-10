"""ChEMBL bioactivity fetcher and curation pipeline for EGFR pIC50 prediction."""

import logging
import logging.handlers
import math
import time
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# Project root is two levels up from src/components/data_loader.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_DATA_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_LOG_DIR = _PROJECT_ROOT / "logs"

_PAGE_SIZE = 1000
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5  # seconds; multiplied by attempt number

_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_units",
    "standard_relation",
    "standard_type",
    "pchembl_value",
    "assay_chembl_id",
    "document_chembl_id",
    "data_validity_comment",
]

# Intermediate activity zone: weakly active but not clearly inactive.
# Excluded to get a cleaner activity cliff.
_IC50_INTERMEDIATE_MIN = 1000  # nM (exclusive lower bound)
_IC50_INTERMEDIATE_MAX = 10_000  # nM (inclusive upper bound)

# Max allowed difference between our pIC50 and ChEMBL's pchembl_value.
_PCHEMBL_TOLERANCE = 0.1

# pIC50 physical plausibility bounds (2 = 10 mM, 12 = 1 pM).
_PIC50_MIN = 2.0
_PIC50_MAX = 12.0

# Minimum heavy-atom count; anything below is too fragment-like for QSAR.
_MIN_HEAVY_ATOMS = 5


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """Set up console + rotating file logging. Call once at the top of your script/notebook.

    Logs are written to: logs/data_curation.log
    Console shows INFO+; file captures DEBUG+ with timestamps.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / "data_curation.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # allow all levels; handlers filter individually

    # Avoid adding duplicate handlers if called more than once (common in notebooks)
    if root.handlers:
        return

    # Console: clean, minimal format
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

    # File: full detail with timestamps, rotates at 5 MB, keeps 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
    )

    root.addHandler(console)
    root.addHandler(file_handler)
    logging.info("Logging started — log file: %s", log_file)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class ChEMBLFetcher:
    """Fetch and curate IC50 bioactivity data from ChEMBL for a single target.

    Typical usage::

        setup_logging()
        fetcher = ChEMBLFetcher("CHEMBL203")

        raw = fetcher.fetch_raw()
        fetcher.save_raw(raw)

        curated = fetcher.curate(raw)
        fetcher.save_curated(curated)
    """

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.curation_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_raw(self) -> pd.DataFrame:
        """Fetch raw IC50 records from ChEMBL. Returns an uncurated DataFrame."""
        logger.info("Starting fetch for target: %s", self.target_id)
        try:
            qs = new_client.activity.filter(
                target_chembl_id=self.target_id,
                assay_type="B",
                standard_type="IC50",
                standard_relation="=",
                standard_units="nM",
                target_organism="Homo sapiens",
            ).only(_FIELDS)

            total = qs.count()
            logger.info("%s: %d raw IC50 records found", self.target_id, total)

            if total == 0:
                logger.warning(
                    "No records found for %s — returning empty DataFrame",
                    self.target_id,
                )
                return pd.DataFrame(columns=_FIELDS)

            records: list[dict] = []
            for offset in range(0, total, _PAGE_SIZE):
                batch = self._fetch_page(qs, offset)
                records.extend(batch)
                logger.info("  fetched %d / %d", min(offset + _PAGE_SIZE, total), total)

            df = pd.DataFrame(records)
            self._log_step("raw fetch", removed=0, remaining=len(df))
            logger.info("Fetch complete: %d rows", len(df))
            return df

        except RuntimeError:
            raise  # already wrapped with context in _fetch_page
        except Exception as exc:
            logger.error(
                "Unexpected error during fetch for %s: %s",
                self.target_id,
                exc,
                exc_info=True,
            )
            raise

    def curate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full curation pipeline. Returns an ML-ready DataFrame.

        Steps (in order):
          1. Validate API filters (IC50, exact relation, nM)
          2. Drop missing / non-positive IC50 or SMILES
          3. Remove intermediate-activity zone (1,000–10,000 nM)
          4. Flag ChEMBL data-validity warnings (kept, logged)
          5. Compute pIC50 = 9 − log10(IC50_nM)
          6. Filter pIC50 to physical range [2, 12]
          7. Validate against pchembl_value (flag disagreements)
          8. Standardize SMILES: remove salts → keep largest fragment → RDKit canonical
          9. Remove inorganic molecules (no carbon atoms)
         10. Remove fragment-like molecules (< 5 heavy atoms)
         11. Deduplicate: same compound across assays → mean pIC50
        """
        if df.empty:
            logger.warning("curate() received an empty DataFrame — returning as-is")
            return df

        logger.info("Starting curation pipeline (%d input rows)", len(df))
        try:
            df = df.copy()
            df = self._validate_api_filters(df)
            df = self._drop_missing_values(df)
            df = self._remove_intermediate_activity(df)
            df = self._flag_validity_comments(df)
            df = self._compute_pic50(df)
            df = self._filter_pic50_range(df)
            df = self._validate_pchembl(df)
            df = self._standardize_smiles(df)
            df = self._remove_inorganic_mols(df)
            df = self._remove_small_molecules(df)
            df = self._deduplicate(df)
        except Exception as exc:
            logger.error("Curation pipeline failed: %s", exc, exc_info=True)
            raise

        self._print_curation_summary()

        return df[
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "pic50",
                "n_measurements",
                "ic50_nM_mean",
                "pchembl_agreement",
            ]
        ]

    def save_raw(self, df: pd.DataFrame) -> Path:
        """Save the raw fetched DataFrame to data/raw/<target_id>_raw.parquet."""
        _DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = _DATA_RAW_DIR / f"{self.target_id}_raw.parquet"
        try:
            df.to_parquet(path, index=False)
            logger.info("Raw data saved → %s  (%d rows)", path, len(df))
        except Exception as exc:
            logger.error("Failed to save raw data to %s: %s", path, exc)
            raise
        return path

    def save_curated(self, df: pd.DataFrame) -> Path:
        """Save the curated DataFrame to data/processed/<target_id>_curated.parquet."""
        _DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = _DATA_PROCESSED_DIR / f"{self.target_id}_curated.parquet"
        try:
            df.to_parquet(path, index=False)
            logger.info("Curated data saved → %s  (%d rows)", path, len(df))
        except Exception as exc:
            logger.error("Failed to save curated data to %s: %s", path, exc)
            raise
        return path

    # ------------------------------------------------------------------
    # Curation steps (called in order by curate())
    # ------------------------------------------------------------------

    def _validate_api_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Confirm API-level filters were honoured (sanity check)."""
        before = len(df)
        mask = (
            df["standard_type"].eq("IC50")
            & df["standard_relation"].eq("=")
            & df["standard_units"].eq("nM")
        )
        df = df[mask].copy()
        self._log_step("validate API filters (IC50 / exact / nM)", before, len(df))
        return df

    def _drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing / non-positive IC50 or missing SMILES."""
        before = len(df)
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value", "canonical_smiles"])
        df = df[df["standard_value"] > 0].copy()
        self._log_step("drop missing / non-positive IC50 or SMILES", before, len(df))
        return df

    def _remove_intermediate_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the 1,000–10,000 nM zone — ambiguous activity, adds noise."""
        before = len(df)
        is_intermediate = (df["standard_value"] > _IC50_INTERMEDIATE_MIN) & (
            df["standard_value"] <= _IC50_INTERMEDIATE_MAX
        )
        df = df[~is_intermediate].copy()
        self._log_step(
            f"remove intermediate activity ({_IC50_INTERMEDIATE_MIN}–{_IC50_INTERMEDIATE_MAX} nM)",
            before,
            len(df),
        )
        return df

    def _flag_validity_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log rows ChEMBL has flagged as problematic (kept, not dropped)."""
        if "data_validity_comment" not in df.columns:
            return df
        flagged = df["data_validity_comment"].notna() & df["data_validity_comment"].ne(
            ""
        )
        n_flagged = flagged.sum()
        if n_flagged:
            comments = df.loc[flagged, "data_validity_comment"].value_counts().to_dict()
            logger.warning(
                "%d rows carry data_validity_comment (kept — review manually): %s",
                n_flagged,
                comments,
            )
        return df

    def _compute_pic50(self, df: pd.DataFrame) -> pd.DataFrame:
        """pIC50 = −log10(IC50 in molar) = 9 − log10(IC50 in nM)."""
        df["pic50"] = 9.0 - df["standard_value"].apply(math.log10)
        return df

    def _filter_pic50_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only pIC50 in [2, 12] — outside this range likely reflects assay artifacts."""
        before = len(df)
        df = df[(df["pic50"] >= _PIC50_MIN) & (df["pic50"] <= _PIC50_MAX)].copy()
        self._log_step(
            f"filter pIC50 range [{_PIC50_MIN}, {_PIC50_MAX}]", before, len(df)
        )
        return df

    def _validate_pchembl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare computed pIC50 against ChEMBL's pchembl_value.

        Difference > 0.1 likely means a unit mismatch. Rows are flagged in
        pchembl_agreement (False) but not removed — review them manually.
        """
        df["pchembl_agreement"] = True

        if "pchembl_value" not in df.columns:
            logger.info("pchembl_value column not present — skipping validation")
            return df

        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        has_pchembl = df["pchembl_value"].notna()
        logger.info(
            "pchembl_value available for %d / %d rows", has_pchembl.sum(), len(df)
        )

        if has_pchembl.sum() > 0:
            diff = (
                df.loc[has_pchembl, "pic50"] - df.loc[has_pchembl, "pchembl_value"]
            ).abs()
            disagreement = diff > _PCHEMBL_TOLERANCE
            n_disagree = disagreement.sum()
            if n_disagree:
                logger.warning(
                    "%d rows: computed pIC50 differs from pchembl_value by >%.1f — "
                    "possible unit mismatch. Flagged in 'pchembl_agreement'.",
                    n_disagree,
                    _PCHEMBL_TOLERANCE,
                )
            df.loc[
                has_pchembl & disagreement.reindex(df.index, fill_value=False),
                "pchembl_agreement",
            ] = False

        return df

    def _standardize_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove salt fragments (keep largest) and re-canonicalize via RDKit."""
        before = len(df)
        df["canonical_smiles"] = df["canonical_smiles"].apply(_standardize_smiles_str)
        df = df.dropna(subset=["canonical_smiles"]).copy()
        self._log_step("standardize SMILES (desalt + canonicalize)", before, len(df))
        return df

    def _remove_inorganic_mols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove molecules without any carbon atoms (metal salts, inorganic complexes)."""
        before = len(df)

        def has_carbon(smiles: str) -> bool:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None and any(
                    a.GetAtomicNum() == 6 for a in mol.GetAtoms()
                )
            except Exception:
                return False

        df = df[df["canonical_smiles"].apply(has_carbon)].copy()
        self._log_step("remove inorganic molecules (no carbon)", before, len(df))
        return df

    def _remove_small_molecules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove fragment-like molecules with fewer than _MIN_HEAVY_ATOMS heavy atoms."""
        before = len(df)

        def heavy_atom_count(smiles: str) -> int:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol.GetNumAtoms() if mol else 0
            except Exception:
                return 0

        df = df[
            df["canonical_smiles"].apply(heavy_atom_count) >= _MIN_HEAVY_ATOMS
        ].copy()
        self._log_step(
            f"remove small molecules (<{_MIN_HEAVY_ATOMS} heavy atoms)", before, len(df)
        )
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse duplicate SMILES: mean pIC50 in log space (= geometric mean IC50)."""
        before = len(df)
        df = df.groupby("canonical_smiles", as_index=False).agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pic50=("pic50", "mean"),
            n_measurements=("pic50", "count"),
            ic50_nM_mean=("standard_value", "mean"),
            pchembl_agreement=(
                "pchembl_agreement",
                "all",
            ),  # False if any row disagreed
        )
        self._log_step("deduplicate (unique compounds)", before, len(df))
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_step(self, step_name: str, before: int, remaining: int) -> None:
        removed = before - remaining
        self.curation_log.append(
            {"step": step_name, "removed": removed, "remaining": remaining}
        )
        logger.debug(
            "Step %-50s removed=%d  remaining=%d", step_name, removed, remaining
        )

    def _print_curation_summary(self) -> None:
        print("\n" + "=" * 68)
        print("  Curation Summary")
        print("=" * 68)
        print(f"  {'Step':<46} {'Removed':>8}  {'Remaining':>9}")
        print("-" * 68)
        for entry in self.curation_log:
            print(
                f"  {entry['step']:<46} {entry['removed']:>8}  {entry['remaining']:>9}"
            )
        print("=" * 68 + "\n")

    def _fetch_page(self, qs, offset: int) -> list[dict]:
        """Fetch one page with exponential-backoff retries."""
        end = offset + _PAGE_SIZE
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return list(qs[offset:end])
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"ChEMBL API failed at offset {offset} after {_MAX_RETRIES} retries"
                    ) from exc
                delay = _RETRY_BACKOFF * attempt
                logger.warning(
                    "Attempt %d/%d failed at offset %d (%s). Retrying in %ds…",
                    attempt,
                    _MAX_RETRIES,
                    offset,
                    exc,
                    delay,
                )
                time.sleep(delay)
        return []  # unreachable; satisfies type checker


# ------------------------------------------------------------------
# Module-level SMILES helper
# ------------------------------------------------------------------


def _standardize_smiles_str(smiles: str) -> str | None:
    """Remove salts (keep largest fragment) and return RDKit canonical SMILES.

    Returns None if RDKit cannot parse the molecule — those rows get dropped.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        # LargestFragmentChooser removes counterions (Na+, Cl-, etc.)
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None
