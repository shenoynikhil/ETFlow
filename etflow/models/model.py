from typing import Optional

import torch
from loguru import logger as log
from torch import Tensor

from etflow.commons import signed_volume
from etflow.models.base import BaseModel
from etflow.models.loss import batchwise_l2_loss
from etflow.models.utils import (
    HarmonicSampler,
    center_of_mass,
    extend_bond_index,
    rmsd_align,
    unsqueeze_like,
)
from etflow.networks.torchmd_net import (
    TensorNetDynamics,
    TorchMDDynamics,
    TorchMDDynamicsWithScore,
)


class BaseFlow(BaseModel):
    """Energy-Based Flow Matching Model (BaseFlow)"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma", "gvp_squared"]
    __path_types__ = ["standard", "cond_ot_path", "pred_x1"]

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
        so3_equivariant: bool = False,
        equivariance_invariance_group: str = "SO(3)",
        dtype: str = torch.float32,
        # flow matching args
        sigma: float = 0.1,
        interpolation_type: str = "linear",
        normalize_node_invariants=False,
        prior_type: str = "gaussian",
        sample_time_dist: str = "uniform",
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
            optimizer_type=optimizer_type,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            grad_norm_max_val=grad_norm_max_val,
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

        # setup network
        if network_type == "TorchMDDynamics":
            self.network = TorchMDDynamics(
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
                so3_equivariant=so3_equivariant,
            )
        elif network_type == "TorchMDDynamicsWithScore":
            self.network = TorchMDDynamicsWithScore(
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
                so3_equivariant=so3_equivariant,
            )
        elif network_type == "TensorNetDynamics":
            self.network = TensorNetDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_num_neighbors=max_num_neighbors,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                equivariance_invariance_group=equivariance_invariance_group,
                clip_during_norm=clip_during_norm,
                reduce_op=reduce_op,
                dtype=dtype,
                output_layer_norm=output_layer_norm,
            )
        else:
            raise NotImplementedError(
                f"Network type {network_type} not implemented for BaseFlow"
            )

        self.sigma = sigma
        self.cutoff = cutoff_upper
        self.parity_switch = parity_switch
        self.normalize_node_invariants = normalize_node_invariants
        self.prior_type = prior_type
        self.interpolation_type = interpolation_type
        self.sample_time_dist = sample_time_dist
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.max_num_neighbors = max_num_neighbors
        self.path_type = path_type

        if parity_switch is not None:
            assert parity_switch in [
                "prior",
                "post_hoc",
            ], f"Parity switch {parity_switch} not implemented"
            log.info(
                f"Will be performing the following parity switch: {self.parity_switch}"
            )

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

        assert (
            self.interpolation_type in self.__interpolation_types__
        ), f"""\Interpolation type {interpolation_type} not available.
            This is the list of implemented interpolations {self.__interpolation_types__}.\n"""

        assert (
            self.path_type in self.__path_types__
        ), f"""\Interpolation type {path_type} not available.
            This is the list of implemented interpolations {self.__path_types__}.\n"""

        if prior_type == "harmonic":
            self.harmonic_sampler = HarmonicSampler(alpha=harmonic_alpha)

    def alpha_t(self, t):
        if self.interpolation_type == "linear":
            return t
        elif self.interpolation_type == "gvp":
            return torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return torch.sqrt(1 - self.sigma_t(t) ** 2) * torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_squared":
            return torch.sin(0.5 * torch.pi * t) ** 2

    def beta_t(self, t):
        if self.interpolation_type == "linear":
            return 1 - t
        elif self.interpolation_type == "gvp":
            return torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return torch.sqrt(1 - self.sigma_t(t) ** 2) * torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_squared":
            return torch.cos(0.5 * torch.pi * t) ** 2

    def alpha_dot_t(self, t):
        if self.interpolation_type == "linear":
            return 1
        elif self.interpolation_type == "gvp":
            return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return -self.sigma_dot_t(t) / torch.sqrt(
                1 - self.sigma_t(t) ** 2
            ) * torch.sin(0.5 * torch.pi * t) + 0.5 * torch.pi * torch.cos(
                0.5 * torch.pi * t
            )
        elif self.interpolation_type == "gvp_squared":
            return (
                torch.pi * torch.sin(0.5 * torch.pi * t) * torch.cos(0.5 * torch.pi * t)
            )

    def beta_dot_t(self, t):
        if self.interpolation_type == "linear":
            return -1
        elif self.interpolation_type == "gvp":
            return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return -self.sigma_dot_t(t) / torch.sqrt(
                1 - self.sigma_t(t) ** 2
            ) * torch.cos(0.5 * torch.pi * t) - 0.5 * torch.pi * torch.sin(
                0.5 * torch.pi * t
            )
        elif self.interpolation_type == "gvp_squared":
            return (
                -torch.pi
                * torch.cos(0.5 * torch.pi * t)
                * torch.sin(0.5 * torch.pi * t)
            )

    def interpolate(self, x0, x1, t):
        return self.alpha_t(t) * x1 + self.beta_t(t) * x0

    def dtIt(self, x0, x1, t):
        return self.alpha_dot_t(t) * x1 + self.beta_dot_t(t) * x0

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def lambda_t(self, t):
        return self.sigma_dot_t(t) - (
            (self.sigma_t(t) * self.alpha_dot_t(t)) / self.alpha_t(t)
        )

    def gamma_t(self, t):
        return self.beta_dot_t(t) - (
            (self.beta_t(t) * self.alpha_dot_t(t)) / self.alpha_t(t)
        )

    def compute_s_from_v(self, t, v, x0, x):
        numerator = (
            (self.alpha_dot_t(t) / self.alpha_t(t)) * x + self.gamma_t(t) * x0 - v
        )

        denominator = self.lambda_t(t) * self.sigma_t(t)
        return numerator / denominator

    def compute_v_from_s(self, t, s, x0, x):
        return (
            (self.alpha_dot_t(t) / self.alpha_t(t)) * x
            + self.gamma_t(t) * x0
            - self.lambda_t(t) * self.sigma_t(t) * s
        )

    def compute_s_from_x1(self, t, x1, x0, x):
        return (1 / (self.sigma_t(t) ** 2)) * (self.interpolate(x0, x1, t) - x)

    def sample_conditional_pt(
        self, x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor, return_eps: bool = False
    ):
        # Have this here in case sample_conditional_pt
        # is used outside of compute_conditional_vector_field
        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # unsqueeze t and then reshape to number of atoms
        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # linear interpolation between x0 and x1
        # mu_t = self.interpolation_fn(x0, x1, t)
        eps = torch.randn_like(x1)

        # center each around center of mass
        eps = center_of_mass(eps, batch=batch)
        mu_t = self.interpolate(x0=x0, x1=x1, t=t)

        # no noise at t = 0 or t = 1
        x_t = mu_t + self.sigma_t(t) * eps

        if return_eps:
            return x_t, eps

        return x_t

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
        if self.path_type == "cond_ot_path":
            coef = self.alpha_dot_t(t) / (1 - self.alpha_t(t))
            u_t = (
                coef * (x1 - self.interpolate(x0=x0, x1=x1, t=t))
                + self.sigma_dot_t(t) * eps
            )
        elif self.path_type == "pred_x1":
            u_t = x1
        else:
            u_t = self.dtIt(x0, x1, t) + self.sigma_dot_t(t) * eps

        return x_t, u_t

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=self.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor

    def sample_base_dist(
        self,
        size=None,
        edge_index=None,
        batch=None,
        smiles=None,
        chiral_index=None,
        chiral_nbr_index=None,
        chiral_tag=None,
    ):
        if self.prior_type == "gaussian":
            assert size is not None
            x0 = torch.randn(size=size, device=self.device)
        elif self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            x0 = self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

        # if parity switch, switch prior correctly
        if self.parity_switch == "prior":
            x0 = self.switch_parity_of_pos(
                x0, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x0

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
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )

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

        return v_t

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
            chiral_index=batched_data.get("chiral_index", None),
            chiral_nbr_index=batched_data.get("chiral_nbr_index", None),
            chiral_tag=batched_data.get("chiral_tag", None),
        )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        if self.prior_type == "harmonic":
            x0 = rmsd_align(pos=x0, ref_pos=pos, batch=batch)

        # check if x0 is nan
        if torch.isnan(x0).any():
            raise ValueError("x0 is NaN. Fix bug in harmonic alignment!")

        # sample conditional vector field for positions
        x_t, u_t = self.compute_conditional_vector_field(
            x0=x0, x1=pos, t=t, batch=batch
        )

        # run flow matching network
        v_t = self(
            z=z,
            t=t,
            pos=x_t,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
        )

        if self.path_type == "pred_x1":
            # Model outputs (pred_pos - pos) so
            # re-add the pos to get pred_pos
            v_t = center_of_mass(v_t + x_t, batch=batch)

        # regress against vector field
        loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")

        if torch.isnan(loss):
            raise ValueError("Loss is NaN, fix bug")

        # log loss
        self.log_helper(f"{stage}/flow_matching_loss", loss, batch_size=batch_size)
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
        chiral_index=None,
        chiral_nbr_index=None,
        chiral_tag=None,
        n_timesteps: int = 20,
        eps: float = 0.0,
        start_time=0.0001,
        end_time=0.9999,
    ):
        if eps > 0.0:
            return self.sdeint(
                z=z,
                bond_index=bond_index,
                batch=batch,
                node_attr=node_attr,
                edge_attr=edge_attr,
                n_timesteps=n_timesteps,
                start_time=start_time,
                end_time=end_time,
                eps=eps,
                chiral_index=chiral_index,
                chiral_nbr_index=chiral_nbr_index,
                chiral_tag=chiral_tag,
            )

        return self.odeint(
            z=z,
            bond_index=bond_index,
            batch=batch,
            node_attr=node_attr,
            edge_attr=edge_attr,
            n_timesteps=n_timesteps,
            start_time=start_time,
            end_time=end_time,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
        )

    @torch.no_grad()
    def odeint(
        self,
        z,
        bond_index,
        batch,
        node_attr=None,
        edge_attr=None,
        chiral_index=None,
        chiral_nbr_index=None,
        chiral_tag=None,
        n_timesteps: int = 20,
        start_time=0.0001,
        end_time=0.9999,
    ):
        t_schedule = torch.linspace(
            start_time, end_time, steps=n_timesteps + 1, device=self.device
        )
        num_atoms = z.size(0)

        x = self.sample_base_dist(
            size=(num_atoms, 3),
            edge_index=bond_index,
            batch=batch,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
        )
        x = center_of_mass(x, batch=batch)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            dt = self._compute_delta_t(t_schedule, t=i)

            v_t = self(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )

            if self.path_type == "pred_x1":
                s = t + dt

                v_t = center_of_mass(v_t + x, batch=batch)
                x1_hat = rmsd_align(v_t, x, batch=batch)

                x = dt / (1 - t) * x1_hat + (1 - s) / (1 - t) * x

            else:
                x = x + dt * v_t

        if self.parity_switch == "post_hoc":
            # perform parity switch
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x

    @torch.no_grad()
    def sdeint(
        self,
        z,
        bond_index,
        batch,
        node_attr=None,
        edge_attr=None,
        chiral_index=None,
        chiral_nbr_index=None,
        chiral_tag=None,
        n_timesteps: int = 20,
        eps: float = 0.0,
        start_time=0.0001,
        end_time=0.9999,
        x0: Optional[Tensor] = None,
    ):
        t_schedule = torch.linspace(
            start_time, end_time, steps=n_timesteps + 1, device=self.device
        )
        num_atoms = z.size(0)

        x = self.sample_base_dist(
            size=(num_atoms, 3),
            edge_index=bond_index,
            batch=batch,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
        )

        x = center_of_mass(x, batch=batch)
        x0 = x.clone()

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            dt = self._compute_delta_t(t_schedule, t=i)

            v_t = self(
                z=z,
                t=t,
                pos=x,
                bond_index=bond_index,
                edge_attr=edge_attr,
                node_attr=node_attr,
                batch=batch,
            )

            dW = torch.randn_like(x)
            dW = center_of_mass(dW, batch=batch)
            dW = torch.sqrt(2 * dt * eps) * dW

            if self.path_type == "pred_x1":
                s = t + dt

                v_t = center_of_mass(v_t + x, batch=batch)
                x1_hat = rmsd_align(v_t, x, batch=batch)

                s_t = self.compute_s_from_x1(t=t, x1=x1_hat, x0=x0, x=x)
                s_t = center_of_mass(s_t, batch=batch)
                s_t = 0.5 * self.sigma**2 * s_t

                x = dt / (1 - t) * x1_hat + (1 - s) / (1 - t) * x
                x = x + eps * s_t * dt + self.sigma_t(t) * dW

            else:
                s_t = self.compute_s_from_v(t=t, v=v_t, x0=x0, x=x)
                s_t = center_of_mass(s_t, batch=batch)
                s_t = 0.5 * self.sigma**2 * s_t
                x = x + dt * (v_t + eps * s_t)
                x = x + self.sigma_t(t) * dW

        if self.parity_switch == "post_hoc":
            # perform parity switch
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x
