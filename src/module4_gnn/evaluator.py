"""Unified model evaluation engine with validation threshold calibration."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.common.logger import get_logger

logger = get_logger("Module4.Evaluator")


@dataclass
class EvaluationResult:
    """Standardized container for binary classification metrics."""
    threshold: float
    precision_illicit: float
    recall_illicit: float
    f1_illicit: float
    pr_auc: float
    roc_auc: float
    accuracy: float
    confusion_matrix: List[List[int]]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    illicit_samples: int
    licit_samples: int

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelEvaluator:
    """Evaluates fraud detection models and calibrates decision thresholds on validation sets."""

    @staticmethod
    def calibrate_threshold(
        y_val: np.ndarray,
        y_prob_val: np.ndarray,
        num_thresholds: int = 200,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
    ) -> Tuple[float, float]:
        """
        Calibrates optimal decision threshold tau* on validation probabilities
        to maximize Illicit F1-score.
        
        Returns: (optimal_threshold: float, best_val_f1: float)
        """
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        best_f1 = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            y_pred = (y_prob_val >= thresh).astype(int)
            # Compute illicit F1 (pos_label=1)
            score = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(thresh)

        logger.info(f"Threshold calibrated on validation: tau*={best_thresh:.4f} (Val Illicit F1={best_f1:.4f})")
        return best_thresh, best_f1

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Evaluates predictions given true binary targets, class 1 probabilities, and a decision threshold.
        """
        y_true_np = np.asarray(y_true, dtype=int)
        y_prob_np = np.asarray(y_prob, dtype=np.float32)

        # Binary decisions based on frozen threshold
        y_pred = (y_prob_np >= threshold).astype(int)

        # Precision-Recall Curve & PR-AUC
        precisions, recalls, _ = precision_recall_curve(y_true_np, y_prob_np, pos_label=1)
        pr_auc_val = float(auc(recalls, precisions))

        # ROC Curve & ROC-AUC
        try:
            roc_auc_val = float(roc_auc_score(y_true_np, y_prob_np))
        except ValueError:
            roc_auc_val = 0.5

        # Confusion Matrix
        cm = confusion_matrix(y_true_np, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_true_np, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true_np, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true_np, y_pred, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_true_np, y_pred))

        return EvaluationResult(
            threshold=float(threshold),
            precision_illicit=round(prec, 4),
            recall_illicit=round(rec, 4),
            f1_illicit=round(f1, 4),
            pr_auc=round(pr_auc_val, 4),
            roc_auc=round(roc_auc_val, 4),
            accuracy=round(acc, 4),
            confusion_matrix=cm.tolist(),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            total_samples=len(y_true_np),
            illicit_samples=int(np.sum(y_true_np == 1)),
            licit_samples=int(np.sum(y_true_np == 0)),
        )
