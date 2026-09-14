"""Loss functions for class-imbalanced fraud classification."""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Class-weighted cross entropy loss."""

    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape [N, 2] or [N] (binary logit)
            targets: Binary targets of shape [N] (0 or 1)
        """
        if logits.dim() == 2 and logits.size(1) == 2:
            return F.cross_entropy(logits, targets, weight=self.weight)
        elif logits.dim() == 1 or logits.size(1) == 1:
            pos_weight = self.weight[1] / self.weight[0] if self.weight is not None else None
            return F.binary_cross_entropy_with_logits(logits.view(-1), targets.float(), pos_weight=pos_weight)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in fraud detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape [N, 2]
            targets: Target classes of shape [N] in {0, 1}
        """
        if logits.dim() == 1:
            logits = torch.stack([-logits, logits], dim=1)

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Target one-hot
        target_one_hot = F.one_hot(targets, num_classes=2).float()
        
        # p_t: probability of true class
        p_t = (probs * target_one_hot).sum(dim=1)
        log_p_t = (log_probs * target_one_hot).sum(dim=1)

        # alpha_t: weight for positive class (1) vs negative class (0)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # Focal loss formula
        loss = -alpha_t * ((1.0 - p_t) ** self.gamma) * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
