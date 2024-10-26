from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .utils import GatedEquivariantBlock, act_class_mapping


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors_feat: Tensor):
        """Coordinate features Normalization"""
        # shape of coors_feat: (num_atoms, 3, hidden_channels)
        norm = coors_feat.norm(dim=1, keepdim=True)
        normed_coors = coors_feat / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model, reduce_op):
        super().__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        return scatter(x, batch, dim=0, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantVectorOutput(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        layer_norm: bool = False,
    ):
        super(EquivariantVectorOutput, self).__init__(
            allow_prior_model=False, reduce_op="sum"
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    layer_norm=layer_norm,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                    layer_norm=layer_norm,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        v = v.squeeze() + pos

        return x, v


class EquivariantVectorAndScalarOutput(OutputModel):
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum"):
        super(EquivariantVectorAndScalarOutput, self).__init__(
            allow_prior_model=False, reduce_op="sum"
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                ),
            ]
        )
        self.scalar_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class_mapping[activation](),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        x = self.scalar_network(x)
        v = v.squeeze() + pos
        return x, v
