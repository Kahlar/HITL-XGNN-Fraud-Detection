"""Tabular baseline models for financial fraud detection: Random Forest, XGBoost, LightGBM, and MLP."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from src.common.logger import get_logger
from src.module4_gnn.loss import WeightedCrossEntropyLoss

logger = get_logger("Module4.Baselines")


class RandomForestBaseline:
    """Random Forest classifier with balanced class weighting."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        random_state: int = 42,
        class_weight: str = "balanced",
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        """Fits Random Forest on training matrix."""
        logger.info(f"Training Random Forest (n_estimators={self.n_estimators}, max_depth={self.max_depth})...")
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of illicit (class 1) probabilities."""
        probs = self.model.predict_proba(X)
        return probs[:, 1].astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Saves model artifact."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)
        logger.info(f"Saved Random Forest model to {p}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RandomForestBaseline":
        """Loads model artifact."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class XGBoostBaseline:
    """XGBoost gradient-boosted decision trees with training scale_pos_weight."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
            tree_method="hist",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15,
    ) -> "XGBoostBaseline":
        """Fits XGBoost on training matrix with optional validation early stopping."""
        if self.scale_pos_weight is None:
            num_pos = np.sum(y_train == 1)
            num_neg = np.sum(y_train == 0)
            self.scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
            self.model.set_params(scale_pos_weight=self.scale_pos_weight)

        logger.info(
            f"Training XGBoost (scale_pos_weight={self.scale_pos_weight:.2f}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})..."
        )

        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of illicit (class 1) probabilities."""
        probs = self.model.predict_proba(X)
        return probs[:, 1].astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Saves model artifact."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))
        logger.info(f"Saved XGBoost model to {p}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostBaseline":
        """Loads model artifact."""
        instance = cls()
        instance.model.load_model(str(path))
        return instance


class LightGBMBaseline:
    """LightGBM gradient-boosted trees with training scale_pos_weight."""

    def __init__(
        self,
        n_estimators: int = 150,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15,
    ) -> "LightGBMBaseline":
        """Fits LightGBM on training matrix with optional validation early stopping."""
        if self.scale_pos_weight is None:
            num_pos = np.sum(y_train == 1)
            num_neg = np.sum(y_train == 0)
            self.scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
            self.model.set_params(scale_pos_weight=self.scale_pos_weight)

        logger.info(
            f"Training LightGBM (scale_pos_weight={self.scale_pos_weight:.2f}, "
            f"learning_rate={self.learning_rate}, num_leaves={self.num_leaves})..."
        )

        callbacks = []
        if X_val is not None and y_val is not None and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of illicit (class 1) probabilities."""
        probs = self.model.predict_proba(X)
        return probs[:, 1].astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Saves model artifact."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)
        logger.info(f"Saved LightGBM model to {p}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LightGBMBaseline":
        """Loads model artifact."""
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class MLPNet(nn.Module):
    """Deep Multi-Layer Perceptron architecture with LayerNorm for stable tabular training."""

    def __init__(self, in_features: int = 165, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw classification logits of shape [N, 2]."""
        return self.net(x)


class MLPBaseline:
    """PyTorch MLP classifier with weighted cross-entropy and validation PR-AUC early stopping."""

    def __init__(
        self,
        in_features: int = 165,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(random_state)
        self.model = MLPNet(
            in_features=in_features,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 256,
        patience: int = 15,
    ) -> "MLPBaseline":
        """Trains MLP using training-derived class weights and validation PR-AUC tracking."""
        # Class weights from train labels strictly: w_pos = N_neg / N_pos
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # DataLoader
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        logger.info(
            f"Training MLP on {self.device} (in_dim={self.in_features}, "
            f"class_weights=[1.0, {pos_weight:.2f}], epochs={epochs})..."
        )

        best_val_score = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)

            # Validation evaluation on PR-AUC
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    vx = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    vout = self.model(vx)
                    vprobs = torch.softmax(vout, dim=1)[:, 1].cpu().numpy()
                    prec, rec, _ = precision_recall_curve(y_val, vprobs, pos_label=1)
                    val_prauc = float(auc(rec, prec))

                if val_prauc > best_val_score:
                    best_val_score = val_prauc
                    best_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(f"MLP early stopping at epoch {epoch}. Best Val PR-AUC: {best_val_score:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Returns 1D array of illicit (class 1) probabilities."""
        self.model.eval()
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_probs = []

        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.model(bx)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Saves model weights."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "in_features": self.in_features,
                "hidden_dim1": self.hidden_dim1,
                "hidden_dim2": self.hidden_dim2,
                "dropout": self.dropout,
            },
            p
        )
        logger.info(f"Saved MLP model to {p}")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "MLPBaseline":
        """Loads model weights."""
        p = Path(path)
        checkpoint = torch.load(p, map_location="cpu", weights_only=False)
        instance = cls(
            in_features=checkpoint["in_features"],
            hidden_dim1=checkpoint["hidden_dim1"],
            hidden_dim2=checkpoint["hidden_dim2"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        return instance
