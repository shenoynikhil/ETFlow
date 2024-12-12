"""Loss Functions"""

from typing import Optional

import torch
from torch_geometric.utils import scatter


def correct_tensor_shape(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return ((prediction - target) ** 2).sum(dim=-1).mean(dim=0)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return (torch.abs(prediction - target)).sum(dim=-1).mean(dim=0)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor):
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return torch.norm(prediction - target, p=2, dim=-1).mean(dim=0)


def batchwise_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    """Mean Squared Error Loss
    This computes the average MSE loss per molecule and then
    averages over number of molecules in the batch.
    """
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        ((prediction - target) ** 2).sum(dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)


def batchwise_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        torch.norm(prediction - target, p=2, dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)
