"""Fine-tune the CheMeleon foundation model on the EGFR pIC50 data.

CheMeleon is a D-MPNN pretrained on ~1M PubChem molecules (to predict Mordred
descriptors). `--from-foundation CHEMELEON` loads those pretrained message-
passing weights and continues training them on our small EGFR set, instead of
starting from random weights like scripts/train_chemprop.py does. This is the
whole reason to bother on a small dataset. Needs chemprop >= 2.2; the
checkpoint is downloaded and cached automatically on first use.

    python scripts/finetune_chemeleon.py --accelerator gpu --epochs 30

Note: chemprop 2.2.3's `--freeze-encoder` only works with `--checkpoint`, not
`--from-foundation` — combining them raises `ArgumentError` (verified by
actually running it, not just reading --help). So there's no CLI-only way to
freeze CheMeleon's encoder here. If fine-tuning overfits on this small a
dataset, the two options that do work are lowering --patience or lowering
--epochs, not freezing.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.components.model_evaluator import regression_metrics


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on val loss. Matches the GNN track for a "
        "fair comparison. Set to a huge number to effectively disable.",
    )
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    args = parser.parse_args()

    splits_dir = ROOT / config["data"]["splits_dir"]
    train_csv = splits_dir / f"{args.split}_train.csv"
    val_csv = splits_dir / f"{args.split}_val.csv"
    test_csv = splits_dir / f"{args.split}_test.csv"

    out_dir = ROOT / config["models_dir"] / "chemprop" / "chemeleon" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "chemprop",
        "train",
        "-i",
        str(train_csv),
        str(val_csv),
        str(test_csv),
        "-s",
        "smiles",
        "--target-columns",
        "pIC50",
        "-t",
        "regression-mve",
        "--metrics",
        "rmse",
        "mae",
        "r2",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
        "--data-seed",
        str(config["seed"]),
        "--pytorch-seed",
        str(config["seed"]),
        "--from-foundation",
        "CHEMELEON",
        # CheMeleon was pretrained with this atom featurizer; chemprop errors
        # out if it doesn't match. Happens to be the CLI default too, but
        # pinned explicitly since this one is a hard requirement, not a knob.
        "--multi-hot-atom-featurizer-mode",
        "V2",
        "-o",
        str(out_dir),
    ]
    run(train_cmd)

    checkpoint = out_dir / "model_0" / "best.pt"
    val_df = predict(checkpoint, val_csv, out_dir / "val_predictions.csv")
    test_df = predict(checkpoint, test_csv, out_dir / "test_predictions.csv")

    log_run(config, args, train_csv, val_df, test_df, checkpoint)


def predict(checkpoint: Path, csv_path: Path, pred_path: Path) -> pd.DataFrame:
    """Run chemprop predict with MVE uncertainty and merge back the truth column."""
    run(
        [
            "chemprop",
            "predict",
            "-i",
            str(csv_path),
            "-s",
            "smiles",
            "--model-paths",
            str(checkpoint),
            "--uncertainty-method",
            "mve",
            "-o",
            str(pred_path),
        ]
    )
    preds = pd.read_csv(pred_path)
    truth = pd.read_csv(csv_path)
    return preds.merge(
        truth[["smiles", "pIC50"]], on="smiles", suffixes=("_pred", "_true")
    )


def log_run(config, args, train_csv, val_df, test_df, checkpoint) -> None:
    """Compute metrics and log everything to W&B.

    AD is deliberately not computed here — it's checked later in a local
    notebook, not on the GCP training box.
    """
    val_metrics = regression_metrics(val_df["pIC50_true"], val_df["pIC50_pred"])
    test_metrics = regression_metrics(test_df["pIC50_true"], test_df["pIC50_pred"])
    train_smiles = pd.read_csv(train_csv)["smiles"]

    print("val:", val_metrics)
    print("test:", test_metrics)

    run = wandb.init(
        project=config["wandb"]["project"],
        job_type="chemprop",
        group=args.split,
        name=f"chemeleon_finetune_{args.split}",
        tags=["chemprop", args.split, "chemeleon"],
        config={
            "model_type": "chemeleon_finetune",
            "split_type": args.split,
            "descriptor_type": "learned_graph",
            "seed": config["seed"],
            "n_train": len(train_smiles),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "epochs": args.epochs,
            "patience": args.patience,
        },
    )
    wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    wandb.log({"test_mean_mve_variance": test_df["pIC50_unc"].mean()})

    artifact = wandb.Artifact(f"chemeleon-finetune-{args.split}", type="model")
    artifact.add_file(str(checkpoint))
    logged = run.log_artifact(artifact)
    run.link_artifact(logged, target_path="wandb-registry-model/egfr-pic50-chemeleon")
    wandb.finish()

    print(f"Checkpoint -> {checkpoint}")


if __name__ == "__main__":
    main()
