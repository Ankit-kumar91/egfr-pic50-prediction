"""Hyperparameter search for chemprop's D-MPNN via Ray Tune — GPU/CPU-hungry,
run this on a GCP VM (see scripts/gcp_setup.sh), not on the Mac.

    pip install -U "ray[tune]" optuna
    python scripts/hpopt_chemprop.py --num-samples 30 --num-gpus 1

--num-cpus/--num-gpus are the TOTAL resources on this machine that Ray may
use (passed to ray.init()), not a per-trial amount — chemprop's hpopt CLI has
no "per-trial GPU fraction" flag. With one GPU and the default
--max-concurrent-trials (unset -> 1), trials simply run one after another,
each using the whole GPU, which is what you want on a single-GPU VM.

Writes best_config.toml under the save dir. Then train the final model (with
proper eval + W&B logging) on it via train_chemprop.py:
    python scripts/train_chemprop.py --accelerator gpu \\
        --config-path models/chemprop/hpopt/scaffold/best_config.toml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    parser.add_argument(
        "--search-space",
        default="basic",
        choices=["basic", "learning_rate", "all"],
        help="basic=depth/ffn_num_layers/dropout/message_hidden_dim/ffn_hidden_dim",
    )
    parser.add_argument("--num-samples", type=int, default=30, help="Ray Tune trials")
    parser.add_argument("--num-cpus", type=int, default=4, help="Total CPUs on this VM")
    parser.add_argument("--num-gpus", type=int, default=1, help="Total GPUs on this VM")
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help="Trials to run at once. Leave unset on a single GPU (defaults to "
        "sequential, one trial using the whole GPU at a time).",
    )
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "auto"])
    args = parser.parse_args()

    splits_dir = ROOT / config["data"]["splits_dir"]
    train_csv = splits_dir / f"{args.split}_train.csv"
    val_csv = splits_dir / f"{args.split}_val.csv"
    test_csv = splits_dir / f"{args.split}_test.csv"

    save_dir = ROOT / config["models_dir"] / "chemprop" / "hpopt" / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "chemprop",
        "hpopt",
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
        "--search-parameter-keywords",
        args.search_space,
        "--raytune-num-samples",
        str(args.num_samples),
        "--raytune-num-cpus",
        str(args.num_cpus),
        "--raytune-num-gpus",
        str(args.num_gpus),
        "--raytune-search-algorithm",
        "optuna",
        "--hpopt-save-dir",
        str(save_dir),
        "--accelerator",
        args.accelerator,
        "--data-seed",
        str(config["seed"]),
        "--pytorch-seed",
        str(config["seed"]),
    ]
    if args.max_concurrent_trials:
        cmd += ["--raytune-max-concurrent-trials", str(args.max_concurrent_trials)]
    run(cmd)

    best_config = save_dir / "best_config.toml"
    print(f"\nBest config -> {best_config}")
    print("Train the final D-MPNN on it with:")
    print(
        f"  python scripts/train_chemprop.py --accelerator {args.accelerator} "
        f"--split {args.split} --config-path {best_config}"
    )


if __name__ == "__main__":
    main()
