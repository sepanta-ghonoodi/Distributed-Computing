"""Typed configuration objects for PAS-LGT.

Configs are plain dataclasses so they can be constructed in code (tests,
notebooks) or loaded from YAML (`configs/phase1_baseline.yaml`). Unknown keys in
the YAML raise immediately rather than being silently ignored — a silent typo in
a hyperparameter name is one of the more expensive bugs in this kind of project.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # Where the unified parquet lives (produced by src/data/preprocess.py or
    # scripts/make_dummy_data.py). All datasets are normalised into one schema
    # so nothing downstream knows whether it is NGSIM, highD or synthetic.
    unified_path: str = "data/processed/unified.parquet"
    cache_dir: str = "data/processed/windows"

    # Raw NGSIM/highD are 10 Hz / 25 Hz. Decoding 30 s at 10 Hz means 300
    # autoregressive steps, which is slow and drifts badly. We resample to
    # `target_hz` (2 Hz default => 20 observed / 60 predicted steps). Raising
    # this is a knob, not a rewrite.
    source_hz: float = 10.0
    target_hz: float = 2.0

    obs_seconds: float = 10.0   # T_obs from the plan
    pred_seconds: float = 30.0  # T_pred from the plan

    window_stride: int = 4      # stride between windows, in target_hz frames
    min_mean_speed: float = 1.0  # m/s; drop parked/stop-and-go-only windows

    # Split by vehicle id (not by window) so that windows from the same vehicle
    # never straddle train/val — otherwise the metrics are optimistic.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 1234

    max_vehicles: int | None = None  # subsample for quick experiments

    @property
    def obs_len(self) -> int:
        return int(round(self.obs_seconds * self.target_hz))

    @property
    def pred_len(self) -> int:
        return int(round(self.pred_seconds * self.target_hz))

    @property
    def dt(self) -> float:
        """Seconds between consecutive frames after resampling."""
        return 1.0 / self.target_hz


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    # Pre-LN ("norm_first") trains far more stably without a long warmup.
    norm_first: bool = True

    # Anchor the prediction on a constant-velocity extrapolation of the last
    # observed step (the "History Message" of PIT-IDM). With the head zero-
    # initialised the model starts as an exact constant-velocity predictor.
    use_cv_prior: bool = True
    # Typical magnitude of the *offset* from that anchor, in metres, over the
    # full horizon. Only used to keep decoder inputs and head outputs at O(1).
    output_scale: float = 10.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Loss on absolute positions, which is what ADE/FDE measure.
    # "huber_position" is far less sensitive to the rare hard-braking window
    # than plain MSE, which otherwise lets a handful of outliers dominate.
    loss: str = "huber_position"

    teacher_forcing: bool = True
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    seed: int = 42

    eval_every: int = 1
    early_stop_patience: int = 12
    out_dir: str = "runs/phase1_baseline"


@dataclass
class PhysicsConfig:
    """Phase 3 — IDM regulariser.

    `weight` of 0 disables the term entirely, which is how the Phase 1/2
    baseline is trained under otherwise identical settings.
    """

    weight: float = 0.0
    # Generic literature values, deliberately different from the parameters the
    # synthetic simulator uses — see src/physics/idm.py for why.
    a_max: float = 1.2
    b_comf: float = 2.2
    s0: float = 2.5
    t_headway: float = 1.4
    delta: float = 4.0
    a_clip: float = 6.0
    min_gap: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return Config(
            data=_build(DataConfig, raw.get("data", {}), "data"),
            model=_build(ModelConfig, raw.get("model", {}), "model"),
            train=_build(TrainConfig, raw.get("train", {}), "train"),
            physics=_build(PhysicsConfig, raw.get("physics", {}), "physics"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, values: dict[str, Any], section: str):
    """Instantiate `cls`, rejecting keys it does not declare."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{section}'. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**values)
