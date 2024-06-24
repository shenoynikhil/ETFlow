from typing import Optional

import torch
from torch import Tensor

from ecgen.models.base_flow import BaseFlow
from ecgen.models.loss import batchwise_l2_loss
from ecgen.models.utils import (
    center_of_mass,
    extend_bond_index,
    rmsd_align,
    unsqueeze_like,
)
from ecgen.networks.torchmd_net import TorchMDDynamics


class BaseSFM(BaseFlow):
    """Energy-Based Flow Matching Model (BaseFlow)"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma"]

    def __init__(
        self,
        # flow matching network args
        network_type: str = "TorchMDDynamics",
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = True,
        clip_during_norm: bool = False,
        max_num_neighbors: int = 32,
        so3_equivariant: bool = True,
        # flow matching args
        sigma: float = 0.1,
        interpolation_type: str = "linear",
        prior_type: str = "gaussian",
        normalize_node_invariants=False,
        sample_time_dist: str = "uniform",
        separate_encoders: bool = False,
        harmonic_alpha: float = 1.0,
        path_type: str = "standard",
        parity_switch: Optional[str] = None,
        # grad norm max value
        grad_norm_max_val: float = 100.0,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        # optimizer
        optimizer_type: str = "Adam",
        lr: float = 1e-3,
        beta1: float = 0.95,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        # lr scheduler args
        lr_scheduler_type: Optional[str] = "plateau",
        factor: float = 0.6,
        patience: int = 10,
        first_cycle_steps: int = 1000,
        cycle_mult: float = 1.0,
        max_lr: float = 0.0001,
        min_lr: float = 1.0e-08,
        warmup_steps: int = 10000,
        gamma: float = 0.75,
        last_epoch: int = -1,
        lr_scheduler_monitor: str = "val/loss",
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
    ):
        super().__init__(
            network_type=network_type,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            attn_activation=attn_activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            reduce_op=reduce_op,
            qk_norm=qk_norm,
            output_layer_norm=output_layer_norm,
            clip_during_norm=clip_during_norm,
            max_num_neighbors=max_num_neighbors,
            so3_equivariant=so3_equivariant,
            # flow matching args
            sigma=sigma,
            interpolation_type=interpolation_type,
            prior_type=prior_type,
            normalize_node_invariants=normalize_node_invariants,
            sample_time_dist=sample_time_dist,
            harmonic_alpha=harmonic_alpha,
            path_type=path_type,
            parity_switch=parity_switch,
            # grad norm max value
            grad_norm_max_val=grad_norm_max_val,
            # make edge_type one-hot
            edge_one_hot=edge_one_hot,
            edge_one_hot_types=edge_one_hot_types,
            # optimizer
            optimizer_type=optimizer_type,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            factor=factor,
            patience=patience,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            gamma=gamma,
            last_epoch=last_epoch,
            lr_scheduler_monitor=lr_scheduler_monitor,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_frequency=lr_scheduler_frequency,
        )

        self.separate_encoders = separate_encoders

        # setup network
        if separate_encoders:
            if network_type == "TorchMDDynamics":
                network_cls = TorchMDDynamics
            else:
                raise NotImplementedError(
                    f"Network type {network_type} not implemented for BaseFlow"
                )

            self.score_network = network_cls(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
            )

    def eta_t(self, t):
        return 2 * self.sigma_t(t) / self.sigma**2

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros((x1.size(0),)).to(self.device)

        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # sample a gaussian centered around the interpolation of x1, x0
        x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch, return_eps=True)
        t = unsqueeze_like(t[batch], x1)

        # derivative of interpolate plus derivative of sigma function * noise
        u_t = self.dtIt(x0, x1, t) + self.sigma_dot_t(t) * eps

        return x_t, u_t, eps

    def sample_base_dist(self, size=None, edge_index=None, batch=None, smiles=None):
        if self.prior_type == "gaussian":
            assert size is not None
            return torch.randn(size=size, device=self.device)
        elif self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            return self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

    def sample_time(
        self,
        num_samples: int,
        low: float = 1e-4,
        high: float = 0.9999,
        stage: str = "train",
    ):
        # batch_size = batch.max().item() + 1
        if self.sample_time_dist == "uniform" or stage == "val":
            # TODO: remove this later on, to remain consistent with val metrics
            # clamp to ensure numerical stability
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        elif self.sample_time_dist == "logit_norm":
            return torch.sigmoid(torch.randn(size=(num_samples, 1))).to(self.device)
        else:
            raise NotImplementedError(
                f"Sample time distribution {self.sample_time_dist} not implemented"
            )

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        # center the positions at 0
        pos = center_of_mass(pos, batch=batch)

        # normalize node attributes
        # This has been empirically useful in EDM.
        # scale one-hot and charges so that model focuses on positions
        if self.normalize_node_invariants and node_attr is not None:
            node_attr = node_attr * 0.1

        # compute extended bond index
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=bond_index,
            batch=batch,
            bond_attr=edge_attr,
            device=self.device,
            cutoff=self.cutoff,
        )

        if self.separate_encoders:
            # compute energy and score from network
            v_t = self.network(
                z=z,
                t=t[batch],
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_type,
                node_attr=node_attr,
                batch=batch,
            )

            s_t = self.score_network(
                z=z,
                t=t[batch],
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_type,
                node_attr=node_attr,
                batch=batch,
            )
        else:
            # compute energy and score from network
            v_t, s_t = self.network(
                z=z,
                t=t[batch],
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_type,
                node_attr=node_attr,
                batch=batch,
            )

        return v_t, s_t

    @torch.enable_grad()
    def generic_step(self, batched_data, batch_idx: int, stage: str):
        # atomic numbers
        z, pos, bond_index, node_attr, edge_attr, batch = (
            batched_data["atomic_numbers"],
            batched_data["pos"],
            batched_data["edge_index"],
            batched_data.get("node_attr", None),  # optional
            batched_data.get("edge_attr", None),  # optional
            batched_data.get("batch", None),  # optional
        )
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # sample base distribution, either from harmonic or gaussian
        # x0 is sampling distribution and not data distribution
        x0 = self.sample_base_dist(
            size=pos.shape,
            edge_index=bond_index,
            batch=batch,
            smiles=batched_data.get("smiles", None),
        )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        if self.prior_type == "harmonic":
            x0 = rmsd_align(pos=x0, ref_pos=pos, batch=batch)

        # sample conditional vector field for positions
        x_t, u_t, eps = self.compute_conditional_vector_field(
            x0=x0, x1=pos, t=t, batch=batch
        )

        # run flow matching network
        v_t, s_t = self(
            z=z,
            t=t,
            pos=x_t,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
        )

        # regress against vector field
        flow_matching_loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")
        eta_t = self.eta_t(t)[batch]

        # score matching against -1 * eps
        score_matching_loss = batchwise_l2_loss(
            eta_t * s_t, -eps, batch=batch, reduce="mean"
        )

        loss = flow_matching_loss + score_matching_loss

        # log loss
        self.log_helper(
            f"{stage}/flow_matching_loss", flow_matching_loss, batch_size=batch_size
        )
        self.log_helper(
            f"{stage}/score_matching_loss", score_matching_loss, batch_size=batch_size
        )
        self.log_helper(f"{stage}/loss", loss, batch_size=batch_size)

        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    @torch.no_grad()
    def sample(
        self,
        z,
        bond_index,
        batch,
        node_attr=None,
        edge_attr=None,
        n_timesteps: int = 20,
        x0: Optional[Tensor] = None,
        use_sde: bool = True,
        eps: float = 1.0,
    ):
        t_schedule = torch.linspace(
            0.0001, 0.9999, steps=n_timesteps + 1, device=self.device
        )

        if use_sde:
            return self.sdeint(
                z=z,
                bond_index=bond_index,
                batch=batch,
                t_schedule=t_schedule,
                node_attr=node_attr,
                edge_attr=edge_attr,
                eps=eps,
                x0=x0,
            )

        return self.odeint(
            z=z,
            bond_index=bond_index,
            batch=batch,
            t_schedule=t_schedule,
            node_attr=node_attr,
            edge_attr=edge_attr,
            x0=x0,
        )

    @torch.no_grad()
    def odeint(
        self,
        z,
        bond_index,
        batch,
        t_schedule,
        node_attr=None,
        edge_attr=None,
        x0=None,
    ):
        num_atoms = z.size(0)

        # sample positions x0 from base distribution
        x = self.sample_base_dist(
            size=(num_atoms, 3), edge_index=bond_index, batch=batch
        )
        x = center_of_mass(x, batch=batch)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            v_t, _ = self(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )

            v_t = v_t.detach()  # detach v_t from graph
            x = x + delta_t * v_t

        return x

    @torch.no_grad()
    def sdeint(
        self,
        z,
        bond_index,
        batch,
        t_schedule,
        node_attr=None,
        edge_attr=None,
        eps: float = 1.0,
        x0=None,
    ):
        num_atoms = z.size(0)

        # sample positions x0 from base distribution
        x = self.sample_base_dist(
            size=(num_atoms, 3), edge_index=bond_index, batch=batch
        )
        x = center_of_mass(x, batch=batch)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)

            delta_t = self._compute_delta_t(t_schedule, t=i)

            v_t, s_t = self(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )

            # no need for 0.5 * sigma^2 since
            # it was incorporated in the loss
            x = x + delta_t * (v_t + eps * s_t)
            x = x.detach()

            dW = torch.randn_like(x)
            dW = center_of_mass(dW, batch=batch)
            dW = self.sigma_t(t) * torch.sqrt(2 * delta_t * eps) * dW
            x = x + dW

        return x
