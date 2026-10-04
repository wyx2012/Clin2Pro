import torch
import torch.nn.functional as F


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    values = F.smooth_l1_loss(prediction, target, reduction="none", beta=delta)
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    values = (prediction - target).pow(2)
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_bce_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_pathway_profiles(
    protein_values: torch.Tensor,
    protein_mask: torch.Tensor,
    membership: torch.Tensor,
):
    weights = protein_mask.to(protein_values.dtype)
    denominator = weights.matmul(membership).clamp_min(1.0)
    profile = (protein_values * weights).matmul(membership) / denominator
    pathway_mask = weights.matmul(membership) > 0
    return profile, pathway_mask
