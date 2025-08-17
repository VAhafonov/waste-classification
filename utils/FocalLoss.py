import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        alpha: balancing factor (can be float or list of per-class weights)
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits (N, C)
        targets: ground truth labels (N,) with class indices [0..C-1]
        """
        # Compute log-probabilities
        log_probs = F.log_softmax(inputs, dim=1)  # (N, C)
        probs = torch.exp(log_probs)              # (N, C)

        # Gather probabilities of the true class
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)  # (N,)
        pt = probs.gather(1, targets).squeeze(1)          # (N,)

        # Alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.squeeze())
        else:
            at = self.alpha

        # Focal loss formula
        loss = -at * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
