"""Classical ML training with conformal prediction intervals.

Uses mapie's SplitConformalRegressor (already an env dependency) for
uncertainty instead of a hand-rolled ensemble — fit on train, conformalize on
val, predict point + interval on val/test.
"""

import contextlib
import copy

import joblib
import numpy as np
import torch
from lightgbm import LGBMRegressor
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from xgboost import XGBRegressor

_ESTIMATORS = {
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "random_forest": RandomForestRegressor,
}

_PARAM_GRIDS = {
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 400],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
    },
    "random_forest": {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3],
    },
}


def get_estimator(model_type: str, seed: int, **params):
    """Build an unfitted sklearn-compatible regressor for one of the model
    classes compared in this project, with sensible defaults applied."""
    if model_type not in _ESTIMATORS:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose from {list(_ESTIMATORS)}"
        )
    cls = _ESTIMATORS[model_type]
    kwargs = {"random_state": seed, **params}
    if model_type == "lightgbm":
        kwargs.setdefault("verbose", -1)
    if model_type == "random_forest":
        # sklearn's RandomForestRegressor default is max_features=1.0 (every
        # split scans all features) — fine for a handful of columns, but on
        # 2048-bit ECFP vectors with unbounded depth it's what made the tuned
        # RF's 5-fold CV (and the search that produced it) so slow. "sqrt" is
        # the standard RF choice and can still be overridden via **params.
        kwargs.setdefault("max_features", "sqrt")
    return cls(**kwargs)


@contextlib.contextmanager
def _tqdm_joblib(total: int, desc: str):
    """Report joblib's parallel batch completions into a tqdm bar.

    Monkeypatches joblib's internal callback since no public progress hook
    exists. If a future joblib version changes this internal, fall back to
    RandomizedSearchCV(verbose=10)'s built-in text progress instead.
    """
    pbar = tqdm(total=total, desc=desc)

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback, joblib.parallel.BatchCompletionCallBack = (
        joblib.parallel.BatchCompletionCallBack,
        _Callback,
    )
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        pbar.close()


def search_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 42,
    n_iter: int = 20,
    n_splits: int = 5,
) -> tuple[dict, float]:
    """Randomized search, 5-fold CV. Returns (best_params, best_cv_rmse).

    Pass `groups` (e.g. Murcko scaffold per row) to use GroupKFold instead of
    plain KFold — keeps CV folds scaffold-disjoint so hyperparameter selection
    on the scaffold split isn't inflated by near-duplicate scaffolds landing in
    both the CV-train and CV-val fold.
    """
    cv = (
        GroupKFold(n_splits=n_splits)
        if groups is not None
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    search = RandomizedSearchCV(
        # n_jobs=1 here: RandomizedSearchCV below already parallelizes across
        # fits with n_jobs=-1. xgboost/lightgbm default to all-core internal
        # threading too, so leaving this unset oversubscribes the CPU
        # (n_jobs worker processes x all-core estimators) and was the actual
        # cause of multi-hour searches on a few thousand rows.
        get_estimator(model_type, seed, n_jobs=1),
        _PARAM_GRIDS[model_type],
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1,
    )
    with _tqdm_joblib(n_iter * n_splits, desc=f"{model_type} search"):
        search.fit(X_train, y_train, groups=groups)
    return search.best_params_, -search.best_score_


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
                break  # val RMSE plateaued for `patience` epochs; stop overfitting

    # Restore the weights from the best val epoch, not whatever epoch we stopped on.
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
