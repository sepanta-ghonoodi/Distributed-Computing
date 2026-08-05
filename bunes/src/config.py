from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class DataConfig:
    unified_path: str = 'data/processed/unified.parquet'
    cache_dir: str = 'data/processed/windows'
    source_hz: float = 10.0
    target_hz: float = 2.0
    obs_seconds: float = 10.0
    pred_seconds: float = 30.0
    window_stride: int = 4
    min_mean_speed: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 1234
    max_vehicles: int | None = None

    @property
    def obs_len(self) -> int:
        return int(round(self.obs_seconds * self.target_hz))

    @property
    def pred_len(self) -> int:
        return int(round(self.pred_seconds * self.target_hz))

    @property
    def dt(self) -> float:
        return 1.0 / self.target_hz

@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = 'gelu'
    norm_first: bool = True
    use_cv_prior: bool = True
    output_scale: float = 10.0

@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    lr: float = 0.0003
    weight_decay: float = 0.0001
    warmup_steps: int = 500
    grad_clip: float = 1.0
    loss: str = 'huber_position'
    teacher_forcing: bool = True
    scheduled_sampling: float = 0.0
    ss_ramp_epochs: int = 10
    num_workers: int = 4
    device: str = 'cuda'
    amp: bool = True
    seed: int = 42
    eval_every: int = 1
    early_stop_patience: int = 12
    out_dir: str = 'runs/phase1_baseline'

@dataclass
class PhysicsConfig:
    weight: float = 0.0
    a_max: float = 1.2
    b_comf: float = 2.2
    s0: float = 2.5
    t_headway: float = 1.4
    delta: float = 4.0
    a_clip: float = 6.0
    min_gap: float = 1.0
    horizon_steps: int | None = None

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> 'Config':
        with open(path, 'r') as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return Config(data=_build(DataConfig, raw.get('data', {}), 'data'), model=_build(ModelConfig, raw.get('model', {}), 'model'), train=_build(TrainConfig, raw.get('train', {}), 'train'), physics=_build(PhysicsConfig, raw.get('physics', {}), 'physics'))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

def _build(cls, values: dict[str, Any], section: str):
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in config section '{section}'. Valid keys: {sorted(valid)}")
    return cls(**values)
