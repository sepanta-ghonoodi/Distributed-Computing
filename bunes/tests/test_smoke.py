"""Fast smoke tests: shapes, invertibility, and teacher-forcing/rollout parity.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import DataConfig, ModelConfig
from src.data.dataset import build_windows
from src.data.preprocess import add_lane_offset, compute_kinematics, fill_leader_speed, resample
from src.data.schema import NUM_FEATURES
from src.data.synthetic import generate_highway
from src.data.transforms import from_agent_frame, to_agent_frame
from src.models.seq2seq_transformer import build_model


def test_agent_frame_roundtrip():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 20, 2)) * 50
    origin = rng.normal(size=(8, 2)) * 100
    theta = rng.uniform(-np.pi, np.pi, 8)
    back = from_agent_frame(to_agent_frame(pts, origin, theta), origin, theta)
    assert np.allclose(pts, back, atol=1e-9)


def test_agent_frame_preserves_distance():
    """ADE computed in the agent frame must equal ADE in the world frame."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(4, 10, 2)), rng.normal(size=(4, 10, 2))
    origin, theta = rng.normal(size=(4, 2)), rng.uniform(-np.pi, np.pi, 4)
    d_world = np.linalg.norm(a - b, axis=-1)
    d_agent = np.linalg.norm(
        to_agent_frame(a, origin, theta) - to_agent_frame(b, origin, theta), axis=-1
    )
    assert np.allclose(d_world, d_agent, atol=1e-9)


def _tiny_bundle():
    df = generate_highway(n_vehicles=12, n_lanes=3, duration_s=120.0, seed=0)
    df = resample(df, 10.0, 2.0)
    df = compute_kinematics(df, dt=0.5)
    df = fill_leader_speed(df)
    df = add_lane_offset(df)
    cfg = DataConfig(obs_seconds=10.0, pred_seconds=30.0, target_hz=2.0, window_stride=4)
    return build_windows(df, cfg), cfg


def test_window_shapes_and_target_consistency():
    bundle, cfg = _tiny_bundle()
    assert bundle.features.shape[1:] == (cfg.obs_len, NUM_FEATURES)
    assert bundle.fut_delta.shape[1:] == (cfg.pred_len, 2)
    # Positions must be the exact cumulative sum of the displacements.
    assert np.allclose(bundle.fut_delta.cumsum(1), bundle.fut_pos, atol=1e-3)
    # The agent-frame origin is the last observed point, so rel_x/rel_y end at 0.
    assert np.allclose(bundle.features[:, -1, 0:2], 0.0, atol=1e-4)


def test_teacher_forcing_matches_rollout():
    """Feeding ground truth through rollout must reproduce the forward pass.

    This is the invariant that breaks silently when the decoder's token
    convention drifts between training and inference — the single most common
    way a Seq2Seq trajectory model ends up 'training fine, predicting garbage'.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, nhead=4, num_encoder_layers=1,
                      num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
    model = build_model(NUM_FEATURES, cfg).eval()

    b, obs, pred = 3, 20, 8
    src = torch.randn(b, obs, NUM_FEATURES)
    cv = torch.randn(b, 2)

    with torch.no_grad():
        free = model.rollout(src, pred, cv)
        # Teacher-force the model's *own* free-running output.
        tgt_in = torch.cat([torch.zeros(b, 1, 2), free["delta"][:, :-1]], dim=1)
        forced = model(src, tgt_in, cv)

    assert torch.allclose(forced, free["delta"], atol=1e-5)


def test_rollout_hook_is_applied():
    """The Phase 2 Link-Projection hook must actually steer the rollout."""
    cfg = ModelConfig(d_model=32, nhead=4, num_encoder_layers=1,
                      num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
    model = build_model(NUM_FEATURES, cfg).eval()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.5], [15.0, -0.5]])

    def snap_to_centreline(pos, step):
        # Stand-in for the real snapper: force the lateral coordinate to zero.
        return torch.stack([pos[:, 0], torch.zeros_like(pos[:, 1])], dim=-1)

    out = model.rollout(src, 8, cv, step_hook=snap_to_centreline)
    assert torch.allclose(out["pos"][:, :, 1], torch.zeros(2, 8), atol=1e-6)
    # And the returned displacements must still integrate to those positions.
    assert torch.allclose(out["delta"].cumsum(1), out["pos"], atol=1e-4)
