import torch
import torch.nn.functional as F


def dice_coefficient_tensor(pred, target, eps=1e-6):
    """
    pred, target: [B,1,H,W] binary pred
    """
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def bce_dice_loss(logits, target, bce_weight=0.7, eps=1e-6):
    """
    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)

    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target_f = target.view(target.size(0), -1)

    intersection = (probs * target_f).sum(dim=1)
    union = probs.sum(dim=1) + target_f.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    dloss = 1.0 - dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dloss
