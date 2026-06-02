"""ChEMBL bioactivity fetcher — IC50 (nM, exact relation) for a given target."""

import logging
import math
import time

import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem

logger = logging.getLogger(__name__)

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
    "assay_chembl_id",
    "document_chembl_id",
    "data_validity_comment",
]


class ChEMBLFetcher:
    """Fetch and curate IC50 bioactivity data from ChEMBL for a single target.

    Filtering at the API level:
      - standard_type  = 'IC50'
      - standard_relation = '='   (exact values only; no '>', '<', '~')
      - standard_units = 'nM'

    Curation steps applied by :meth:`curate`:
      1. Coerce IC50 to numeric; drop nulls.
      2. Drop IC50 ≤ 0 (undefined in log space).
      3. Warn on any row flagged by ChEMBL's data_validity_comment.
      4. Compute pIC50 = 9 − log10(IC50_nM)  [= −log10(IC50_molar)].
      5. Re-canonicalize SMILES via RDKit (ChEMBL canonical ≠ RDKit canonical).
      6. Deduplicate on RDKit canonical SMILES:
           same compound, multiple assays → arithmetic mean pIC50 in log space
           (equivalent to geometric mean IC50).
    """

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_raw(self) -> pd.DataFrame:
        """Return a raw DataFrame straight from the ChEMBL API (no curation)."""
        qs = new_client.activity.filter(
            target_chembl_id=self.target_id,
            standard_type="IC50",
            standard_relation="=",
            standard_units="nM",
        ).only(_FIELDS)

        total = qs.count()
        logger.info("%s: %d raw IC50 records found", self.target_id, total)

        if total == 0:
            return pd.DataFrame(columns=_FIELDS)

        records: list[dict] = []
        for offset in range(0, total, _PAGE_SIZE):
            batch = self._fetch_page(qs, offset)
            records.extend(batch)
            logger.info(
                "  fetched %d / %d",
                min(offset + _PAGE_SIZE, total),
                total,
            )

        df = pd.DataFrame(records)
        logger.info("Raw fetch complete: %d rows", len(df))
        return df

    def curate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply curation pipeline; return ML-ready DataFrame."""
        if df.empty:
            logger.warning("curate() received an empty DataFrame — returning as-is")
            return df

        df = df.copy()
        df = self._validate_api_filters(df)
        df = self._coerce_and_drop_nulls(df)
        df = self._flag_validity_comments(df)
        df = self._compute_pic50(df)
        df = self._rdkit_canonicalize(df)
        df = self._deduplicate(df)

        return df[
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "pic50",
                "n_measurements",
                "ic50_nM_mean",
            ]
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        return []  # unreachable, satisfies type checker

    def _validate_api_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-fetch sanity check: confirm API filters were honoured."""
        bad_type = df["standard_type"].ne("IC50").sum()
        bad_rel = df["standard_relation"].ne("=").sum()
        bad_units = df["standard_units"].ne("nM").sum()
        for label, count in [
            ("non-IC50 type", bad_type),
            ("non-exact relation", bad_rel),
            ("non-nM units", bad_units),
        ]:
            if count:
                logger.warning("Unexpected rows with %s: %d — dropping", label, count)
        mask = (
            df["standard_type"].eq("IC50")
            & df["standard_relation"].eq("=")
            & df["standard_units"].eq("nM")
        )
        return df[mask].copy()

    def _coerce_and_drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["standard_value", "canonical_smiles"])
        dropped_null = before - len(df)

        before = len(df)
        df = df[df["standard_value"] > 0]
        dropped_nonpos = before - len(df)

        if dropped_null:
            logger.warning("Dropped %d rows: null IC50 or SMILES", dropped_null)
        if dropped_nonpos:
            logger.warning(
                "Dropped %d rows: IC50 ≤ 0 (undefined pIC50)", dropped_nonpos
            )
        return df

    def _flag_validity_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        if "data_validity_comment" not in df.columns:
            return df
        flagged = df["data_validity_comment"].notna() & df["data_validity_comment"].ne(
            ""
        )
        n_flagged = flagged.sum()
        if n_flagged:
            comments = df.loc[flagged, "data_validity_comment"].value_counts().to_dict()
            logger.warning(
                "%d rows carry ChEMBL data_validity_comment (kept, but review): %s",
                n_flagged,
                comments,
            )
        return df

    def _compute_pic50(self, df: pd.DataFrame) -> pd.DataFrame:
        # pIC50 = −log10(IC50 in molar) = 9 − log10(IC50 in nM)
        df["pic50"] = 9.0 - df["standard_value"].apply(math.log10)
        return df

    def _rdkit_canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df["canonical_smiles"] = df["canonical_smiles"].apply(_to_rdkit_smiles)
        before = len(df)
        df = df.dropna(subset=["canonical_smiles"])
        invalid = before - len(df)
        if invalid:
            logger.warning("Dropped %d rows: RDKit could not parse SMILES", invalid)
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.groupby("canonical_smiles", as_index=False).agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pic50=("pic50", "mean"),  # mean in log space = geometric mean IC50
            n_measurements=("pic50", "count"),
            ic50_nM_mean=("standard_value", "mean"),
        )
        after = len(df)
        logger.info(
            "Deduplication: %d measurements → %d unique compounds (%d removed)",
            before,
            after,
            before - after,
        )
        return df


# ------------------------------------------------------------------
# Module-level helper (used by _rdkit_canonicalize)
# ------------------------------------------------------------------


def _to_rdkit_smiles(smiles: str) -> str | None:
    """Return RDKit isomeric canonical SMILES, or None if unparseable."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None
