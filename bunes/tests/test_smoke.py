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


def _tiny_model():
    cfg = ModelConfig(d_model=32, nhead=4, num_encoder_layers=1,
                      num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
    return build_model(NUM_FEATURES, cfg).eval()


def test_teacher_forcing_matches_rollout():
    """Feeding the model's own output back through forward() must reproduce it.

    This is the invariant that breaks silently when the decoder's token
    convention drifts between training and inference — the single most common
    way a Seq2Seq trajectory model ends up 'training fine, predicting garbage'.
    """
    torch.manual_seed(0)
    model = _tiny_model()

    b, obs, pred = 3, 20, 8
    src = torch.randn(b, obs, NUM_FEATURES)
    cv = torch.randn(b, 2)

    with torch.no_grad():
        free = model.rollout(src, pred, cv)
        # Teacher-force on the model's *own* free-running positions.
        forced = model(src, free["pos"], cv)

    assert torch.allclose(forced, free["pos"], atol=1e-5)


def test_untrained_model_is_constant_velocity():
    """The zero-initialised head must make the model an exact CV predictor.

    This guards the anchoring that keeps the rollout from diverging: if the
    head ever stops being zero-initialised, training no longer starts from a
    sane reference and the failure is silent.
    """
    model = _tiny_model()
    src = torch.randn(4, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.2], [12.0, -0.3], [30.0, 0.0], [7.5, 1.0]])

    out = model.rollout(src, 60, cv)
    steps = torch.arange(1, 61, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(out["pos"], cv.unsqueeze(1) * steps, atol=1e-4)


def test_rollout_does_not_integrate_its_own_error():
    """A perturbation at one step must not accumulate into later steps.

    The first version of the model predicted per-step displacements and
    integrated them, so a single bad step shifted every subsequent position.
    With positions anchored to the constant-velocity prior, corrupting step k
    must leave step k+1 onwards anchored, not permanently offset.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.0], [15.0, 0.0]])

    def kick(pos, step):
        # Shove step 3 fifty metres sideways, leave everything else alone.
        return pos + torch.tensor([0.0, 50.0]) if step == 3 else pos

    clean = model.rollout(src, 12, cv)["pos"]
    kicked = model.rollout(src, 12, cv, step_hook=kick)["pos"]

    drift = (kicked - clean)[:, -1, :].abs().max()
    assert drift < 5.0, f"a single 50 m perturbation still moved the last step by {drift:.1f} m"


def test_frame_transforms_torch_match_numpy():
    """The torch transforms used by Link Projection must agree with the numpy
    pair used at preprocessing time — otherwise snapping happens in the wrong
    frame and silently corrupts every corrected step."""
    from src.map.highway import from_agent_frame_t, to_agent_frame_t

    rng = np.random.default_rng(3)
    pts = rng.normal(size=(16, 2)) * 40
    origin = rng.normal(size=(16, 2)) * 200
    theta = rng.uniform(-np.pi, np.pi, 16)

    np_out = to_agent_frame(pts[:, None, :], origin, theta)[:, 0, :]
    t_out = to_agent_frame_t(
        torch.tensor(pts), torch.tensor(origin), torch.tensor(theta)
    ).numpy()
    assert np.allclose(np_out, t_out, atol=1e-9)

    back = from_agent_frame_t(
        torch.tensor(t_out, dtype=torch.float64), torch.tensor(origin), torch.tensor(theta)
    ).numpy()
    assert np.allclose(back, pts, atol=1e-9)


def test_link_projection_snaps_to_lane_centres():
    """Full-strength snapping must put every corrected point on a centreline."""
    from src.map.highway import (
        LaneCentreMap,
        from_agent_frame_t,
        make_link_projection_hook,
    )

    lane_map = LaneCentreMap(torch.tensor([0.0, 3.7, 7.4, 11.1]))
    model = _tiny_model()

    b = 4
    src = torch.randn(b, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.4]] * b)
    origin = torch.randn(b, 2) * 100
    theta = torch.rand(b) * 0.2 - 0.1

    hook = make_link_projection_hook(lane_map, origin, theta, blend=1.0, start_step=0)
    pos = model.rollout(src, 12, cv, step_hook=hook)["pos"]

    # Every predicted point, back in world coordinates, must sit on a centreline.
    flat = pos.reshape(-1, 2)
    o = origin.repeat_interleave(12, dim=0)
    th = theta.repeat_interleave(12, dim=0)
    world_y = from_agent_frame_t(flat, o, th)[:, 1]
    dist = (world_y.view(-1, 1) - lane_map.centres.view(1, -1)).abs().min(dim=1).values
    assert dist.max() < 1e-4, f"max distance to a lane centre was {dist.max():.4f} m"


def test_rollout_hook_is_applied():
    """The Phase 2 Link-Projection hook must actually steer the rollout."""
    model = _tiny_model()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.5], [15.0, -0.5]])

    def snap_to_centreline(pos, step):
        # Stand-in for the real snapper: force the lateral coordinate to zero.
        return torch.stack([pos[:, 0], torch.zeros_like(pos[:, 1])], dim=-1)

    out = model.rollout(src, 8, cv, step_hook=snap_to_centreline)
    assert torch.allclose(out["pos"][:, :, 1], torch.zeros(2, 8), atol=1e-6)
    # The reported displacements must still integrate back to those positions.
    assert torch.allclose(out["delta"].cumsum(1), out["pos"], atol=1e-4)
