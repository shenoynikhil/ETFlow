import os
from typing import Optional

try:
    from pydantic.v1 import BaseModel, Field, validator
except ImportError:
    from pydantic import BaseModel, validator, Field

import fsspec
from tqdm import tqdm
from typing_extensions import Literal

CACHE = "~/.cache/etflow"


def download_with_progress(url, destination, chunk_size=2**20):  # 1MB chunks
    """
    Download a file with progress bar using fsspec and tqdm.

    Args:
        url: Source URL
        destination: Local destination path
        chunk_size: Size of chunks to read/write in bytes
    """
    with fsspec.open(url, "rb") as source:
        # Get file size if available
        try:
            file_size = source.size
        except AttributeError:
            file_size = None

        # Create progress bar
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading to {destination}",
        ) as pbar:
            # Open destination file
            with open(destination, "wb") as target:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    target.write(chunk)
                    pbar.update(len(chunk))


class BaseConfigSchema(BaseModel):
    class Config(BaseModel.Config):
        case_insensitive = True
        extra = "forbid"


class CheckpointConfigSchema(BaseConfigSchema):
    type: str
    checkpoint_path: str
    cache: Optional[str] = CACHE
    _format: Literal[str] = ".ckpt"

    @validator("cache")
    def validate_cache(cls, value):
        if not value:
            value = "cache"
        if not os.path.exists(value):
            os.makedirs(value)
        return value

    def fetch_checkpoint(self) -> str:
        self.create_cache()
        if not self.checkpoint_exists():
            download_with_progress(self.checkpoint_path, self.local_path)
        else:
            print(f"Checkpoint found at {self.local_path}")
        return self

    @property
    def local_path(self) -> str:
        return os.path.join(self.cache, self.type + self._format)

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.local_path)

    def cache_exists(self) -> bool:
        return os.path.exists(self.cache)

    def create_cache(self):
        if not self.cache_exists():
            os.makedirs(self.cache)

    def set_cache(self, cache: str):
        if not cache:
            return
        self.cache = cache


class ModelArgsSchema(BaseConfigSchema):
    network_type: Literal["TorchMDDynamics"] = "TorchMDDynamics"
    hidden_channels: int = 160
    num_layers: int = 20
    num_rbf: int = 64
    rbf_type: Literal["expnorm"] = "expnorm"
    trainable_rbf: bool = True
    activation: Literal["silu"] = "silu"
    neighbor_embedding: bool = True
    cutoff_lower: float = 0.0
    cutoff_upper: float = 10.0
    max_z: int = 100
    node_attr_dim: int = 10
    edge_attr_dim: int = 1
    attn_activation: Literal["silu"] = "silu"
    num_heads: int = 8
    distance_influence: Literal["both"] = "both"
    reduce_op: Literal["sum"] = "sum"
    qk_norm: bool = True
    so3_equivariant: bool = False
    clip_during_norm: bool = True
    parity_switch: Literal["post_hoc"] = "post_hoc"
    output_layer_norm: bool = False

    # flow matching specific
    sigma: float = 0.1
    prior_type: Literal["harmonic"] = "harmonic"
    interpolation_type: Literal["linear"] = "linear"

    # optimizer args
    optimizer_type: Literal["AdamW"] = "AdamW"
    lr: float = 8.0e-4
    weight_decay: float = 1.0e-8

    # lr scheduler args
    lr_scheduler_type: Literal[
        "CosineAnnealingWarmupRestarts"
    ] = "CosineAnnealingWarmupRestarts"
    first_cycle_steps: int = 375_000
    cycle_mult: float = 1.0
    max_lr: float = 5.0e-4
    min_lr: float = 1.0e-8
    warmup_steps: int = 0
    gamma: float = 0.05
    last_epoch: int = -1
    lr_scheduler_monitor: Literal["val/loss"] = "val/loss"
    lr_scheduler_interval: Literal["step"] = "step"
    lr_scheduler_frequency: int = 1


class ModelConfigSchema(BaseConfigSchema):
    model: Literal["BaseFlow"] = "BaseFlow"
    model_args: ModelArgsSchema = ModelArgsSchema()
    checkpoint_config: CheckpointConfigSchema

    def model_dict(self):
        return self.dict(exclude={"checkpoint_config"})


class DRUGS_O3_CHECKPOINT(CheckpointConfigSchema):
    type: Literal["drugs-o3"] = "drugs-o3"
    checkpoint_path: str = (
        "https://zenodo.org/records/14226681/files/drugs-o3.ckpt?download=1"
    )


class DRUGS_SO3_CHECKPOINT(CheckpointConfigSchema):
    type: Literal["drugs-so3"] = "drugs-so3"
    checkpoint_path: str = (
        "https://zenodo.org/records/14226681/files/drugs-so3.ckpt?download=1"
    )


class QM9_O3_CHECKPOINT(CheckpointConfigSchema):
    type: Literal["qm9-o3"] = "qm9-o3"
    checkpoint_path: str = (
        "https://zenodo.org/records/14226681/files/qm9-o3.ckpt?download=1"
    )


class DRUGS_O3(ModelConfigSchema):
    checkpoint_config: CheckpointConfigSchema = Field(
        default_factory=DRUGS_O3_CHECKPOINT
    )


class DRUGS_SO3(ModelConfigSchema):
    checkpoint_config: CheckpointConfigSchema = Field(
        default_factory=DRUGS_SO3_CHECKPOINT
    )
    model_args: ModelConfigSchema = ModelArgsSchema(so3_equivariant=True)


class QM9_O3(ModelConfigSchema):
    checkpoint_config: CheckpointConfigSchema = Field(default_factory=QM9_O3_CHECKPOINT)
    model_args: ModelConfigSchema = ModelArgsSchema(
        output_layer_norm=True,
        lr=7.0e-4,
        first_cycle_steps=250_000,
        max_lr=7.0e-4,
    )


CONFIG_DICT = {
    "drugs-o3": DRUGS_O3,
    "drugs-so3": DRUGS_SO3,
    "qm9-o3": QM9_O3,
}

if __name__ == "__main__":
    print(DRUGS_O3().checkpoint_config)
