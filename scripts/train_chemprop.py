"""Train chemprop's D-MPNN via its CLI, evaluate consistently with the other
model tracks, and log to MLflow.

Local smoke test:
    python scripts/train_chemprop.py --epochs 5 --accelerator cpu

Full run on a GCP GPU box:
    python scripts/train_chemprop.py --accelerator gpu --devices 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.components.model_evaluator import applicability_domain, regression_metrics


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    args = parser.parse_args()

    splits_dir = ROOT / config["data"]["splits_dir"]
    train_csv = splits_dir / f"{args.split}_train.csv"
    val_csv = splits_dir / f"{args.split}_val.csv"
    test_csv = splits_dir / f"{args.split}_test.csv"

    out_dir = ROOT / config["models_dir"] / "chemprop" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
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
            "--accelerator",
            args.accelerator,
            "--devices",
            args.devices,
            "--data-seed",
            str(config["seed"]),
            "--pytorch-seed",
            str(config["seed"]),
            "-o",
            str(out_dir),
        ]
    )

    checkpoint = out_dir / "model_0" / "best.pt"

    def predict_with_uncertainty(csv_path: Path, tag: str) -> pd.DataFrame:
        pred_path = out_dir / f"{tag}_predictions.csv"
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
        true = pd.read_csv(csv_path)
        return preds.merge(
            true[["smiles", "pIC50"]], on="smiles", suffixes=("_pred", "_true")
        )

    val_df = predict_with_uncertainty(val_csv, "val")
    test_df = predict_with_uncertainty(test_csv, "test")

    val_metrics = regression_metrics(val_df["pIC50_true"], val_df["pIC50_pred"])
    test_metrics = regression_metrics(test_df["pIC50_true"], test_df["pIC50_pred"])

    train_smiles = pd.read_csv(train_csv)["smiles"]
    _, in_domain = applicability_domain(
        test_df["smiles"],
        train_smiles,
        threshold=config["applicability_domain"]["tanimoto_threshold"],
    )

    print("val:", val_metrics)
    print("test:", test_metrics)
    print("test AD in-domain frac:", in_domain.mean())

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/")
    )
    experiment = config["mlflow"]["chemprop_experiment_name"]
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=str(ROOT / "mlruns" / "artifacts")
        )
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"chemprop_dmpnn_{args.split}"):
        mlflow.log_params(
            {
                "model_type": "chemprop_dmpnn",
                "split_type": args.split,
                "descriptor_type": "learned_graph",
                "seed": config["seed"],
                "n_train": len(pd.read_csv(train_csv)),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "epochs": args.epochs,
                "accelerator": args.accelerator,
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("test_mean_mve_variance", test_df["pIC50_unc"].mean())
        mlflow.log_metric("test_ad_in_domain_frac", in_domain.mean())
        mlflow.log_artifact(str(checkpoint))

    print(f"Checkpoint -> {checkpoint}")


if __name__ == "__main__":
    main()
