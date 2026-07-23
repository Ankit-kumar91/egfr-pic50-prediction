"""Train the from-scratch MPNN on the scaffold split, log to MLflow.

Local smoke test:
    python scripts/train_gnn.py --epochs 5 --device cpu

Full run on a GCP GPU box (see configs/config.yaml for default hyperparams):
    python scripts/train_gnn.py --device cuda
"""

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.components.gnn_model import MPNN, graphs_from_df
from src.components.model_evaluator import applicability_domain, regression_metrics
from src.components.model_trainer import predict_gnn_with_uncertainty, train_gnn


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    gnn_cfg = config["gnn"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=gnn_cfg["epochs"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    args = parser.parse_args()

    splits_dir = ROOT / config["data"]["splits_dir"]
    train_df = pd.read_csv(splits_dir / f"{args.split}_train.csv")
    val_df = pd.read_csv(splits_dir / f"{args.split}_val.csv")
    test_df = pd.read_csv(splits_dir / f"{args.split}_test.csv")

    print(f"Building graphs ({args.split} split)...")
    train_g = graphs_from_df(train_df)
    val_g = graphs_from_df(val_df)
    test_g = graphs_from_df(test_df)
    print(f"  train={len(train_g)} val={len(val_g)} test={len(test_g)}")

    model = MPNN(
        hidden_dim=gnn_cfg["hidden_dim"],
        depth=gnn_cfg["depth"],
        dropout=gnn_cfg["dropout"],
    )

    print(f"Training on {args.device} for up to {args.epochs} epochs...")
    model, history = train_gnn(
        model,
        train_g,
        val_g,
        epochs=args.epochs,
        lr=gnn_cfg["lr"],
        batch_size=gnn_cfg["batch_size"],
        patience=gnn_cfg["patience"],
        device=args.device,
    )
    print(f"  stopped at epoch {history[-1]['epoch']}, best val_rmse in history")

    val_pred, _val_std, val_true = predict_gnn_with_uncertainty(
        model, val_g, n_mc=gnn_cfg["mc_dropout_samples"], device=args.device
    )
    test_pred, test_std, test_true = predict_gnn_with_uncertainty(
        model, test_g, n_mc=gnn_cfg["mc_dropout_samples"], device=args.device
    )
    val_metrics = regression_metrics(val_true, val_pred)
    test_metrics = regression_metrics(test_true, test_pred)

    test_smiles = [test_df["smiles"].iloc[i] for i in range(len(test_g))]
    train_smiles = [train_df["smiles"].iloc[i] for i in range(len(train_g))]
    _, in_domain = applicability_domain(
        test_smiles,
        train_smiles,
        threshold=config["applicability_domain"]["tanimoto_threshold"],
    )

    print("val:", val_metrics)
    print("test:", test_metrics)
    print("test AD in-domain frac:", in_domain.mean())

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/")
    )
    experiment = config["mlflow"]["gnn_experiment_name"]
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=str(ROOT / "mlruns" / "artifacts")
        )
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"mpnn_{args.split}"):
        mlflow.log_params(
            {
                "model_type": "mpnn_scratch",
                "split_type": args.split,
                "descriptor_type": "graph",
                "seed": config["seed"],
                "n_train": len(train_g),
                "n_val": len(val_g),
                "n_test": len(test_g),
                "device": args.device,
                "epochs_run": history[-1]["epoch"] + 1,
                **gnn_cfg,
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("test_mean_mc_std", test_std.mean())
        mlflow.log_metric("test_ad_in_domain_frac", in_domain.mean())
        mlflow.pytorch.log_model(model, name="model")

    models_dir = ROOT / config["models_dir"]
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model.cpu(), models_dir / f"mpnn_{args.split}.joblib")
    print(f"Saved -> {models_dir / f'mpnn_{args.split}.joblib'}")


if __name__ == "__main__":
    main()
