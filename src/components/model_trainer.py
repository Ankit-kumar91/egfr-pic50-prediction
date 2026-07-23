"""Classical ML training with conformal prediction intervals.

Uses mapie's SplitConformalRegressor (already an env dependency) for
uncertainty instead of a hand-rolled ensemble — fit on train, conformalize on
val, predict point + interval on val/test.
"""

import copy

import numpy as np
import torch
from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader
from xgboost import XGBRegressor

_ESTIMATORS = {
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "random_forest": RandomForestRegressor,
}


def get_estimator(model_type: str, seed: int, **params):
    if model_type not in _ESTIMATORS:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose from {list(_ESTIMATORS)}"
        )
    cls = _ESTIMATORS[model_type]
    kwargs = {"random_state": seed, **params}
    if model_type == "lightgbm":
        kwargs.setdefault("verbose", -1)
    return cls(**kwargs)


def train_with_uncertainty(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_conf: np.ndarray,
    y_conf: np.ndarray,
    seed: int = 42,
    confidence_level: float = 0.9,
    **params,
) -> SplitConformalRegressor:
    """Fit on train, conformalize on val (X_conf/y_conf) -> intervals on any X."""
    estimator = get_estimator(model_type, seed, **params)
    model = SplitConformalRegressor(
        estimator=estimator, confidence_level=confidence_level, prefit=False
    )
    model.fit(X_train, y_train)
    model.conformalize(X_conf, y_conf)
    return model


def predict_with_interval(model: SplitConformalRegressor, X: np.ndarray):
    """Returns (y_pred, y_lower, y_upper)."""
    y_pred, y_pis = model.predict_interval(X)
    return y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0]


def train_gnn(
    model,
    train_graphs: list,
    val_graphs: list,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 15,
    device: str = "cpu",
):
    """Train an MPNN with Adam/MSE and early stopping on val RMSE.

    Returns (model with best-val weights loaded, history list of dicts).
    """
    model = model.to(device)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_sq_errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_sq_errors.append(
                    torch.nn.functional.mse_loss(pred, batch.y, reduction="none")
                )
        val_rmse = torch.cat(val_sq_errors).mean().sqrt().item()
        history.append({"epoch": epoch, "val_rmse": val_rmse})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_gnn_with_uncertainty(
    model, graphs: list, batch_size: int = 64, n_mc: int = 20, device: str = "cpu"
):
    """MC-dropout: keep dropout active over n_mc stochastic forward passes.

    Returns (y_pred_mean, y_std, y_true) in dataset order (loader is unshuffled).
    """
    model = model.to(device)
    model.train()  # dropout stays active
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            run_preds = []
            for batch in loader:
                run_preds.extend(model(batch.to(device)).cpu().tolist())
            preds.append(run_preds)
    preds = np.array(preds)  # (n_mc, n_samples)

    y_true = np.array([g.y.item() for g in graphs])
    return preds.mean(axis=0), preds.std(axis=0), y_true
