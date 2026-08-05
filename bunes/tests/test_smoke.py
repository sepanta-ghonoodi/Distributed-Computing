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
    assert np.allclose(pts, back, atol=1e-09)

def test_agent_frame_preserves_distance():
    rng = np.random.default_rng(1)
    (a, b) = (rng.normal(size=(4, 10, 2)), rng.normal(size=(4, 10, 2)))
    (origin, theta) = (rng.normal(size=(4, 2)), rng.uniform(-np.pi, np.pi, 4))
    d_world = np.linalg.norm(a - b, axis=-1)
    d_agent = np.linalg.norm(to_agent_frame(a, origin, theta) - to_agent_frame(b, origin, theta), axis=-1)
    assert np.allclose(d_world, d_agent, atol=1e-09)

def _tiny_bundle():
    df = generate_highway(n_vehicles=12, n_lanes=3, duration_s=120.0, seed=0)
    df = resample(df, 10.0, 2.0)
    df = compute_kinematics(df, dt=0.5)
    df = fill_leader_speed(df)
    df = add_lane_offset(df)
    cfg = DataConfig(obs_seconds=10.0, pred_seconds=30.0, target_hz=2.0, window_stride=4)
    return (build_windows(df, cfg), cfg)

def test_window_shapes_and_target_consistency():
    (bundle, cfg) = _tiny_bundle()
    assert bundle.features.shape[1:] == (cfg.obs_len, NUM_FEATURES)
    assert bundle.fut_delta.shape[1:] == (cfg.pred_len, 2)
    assert np.allclose(bundle.fut_delta.cumsum(1), bundle.fut_pos, atol=0.001)
    assert np.allclose(bundle.features[:, -1, 0:2], 0.0, atol=0.0001)

def _tiny_model():
    cfg = ModelConfig(d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
    return build_model(NUM_FEATURES, cfg).eval()

def test_teacher_forcing_matches_rollout():
    torch.manual_seed(0)
    model = _tiny_model()
    (b, obs, pred) = (3, 20, 8)
    src = torch.randn(b, obs, NUM_FEATURES)
    cv = torch.randn(b, 2)
    with torch.no_grad():
        free = model.rollout(src, pred, cv)
        forced = model(src, free['pos'], cv)
    assert torch.allclose(forced, free['pos'], atol=1e-05)

def test_scheduled_sampling_mixes_without_leaking_gradients():
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[12.0, 0.0], [12.0, 0.0]])
    tgt = torch.randn(2, 8, 2).cumsum(1)
    with torch.no_grad():
        guess = model(src, tgt, cv)
    take = torch.rand_like(guess[..., :1]) < 0.5
    mixed = torch.where(take, guess.detach(), tgt)
    assert not mixed.requires_grad
    keep = (~take).expand_as(tgt)
    assert torch.equal(mixed[keep], tgt[keep])

def test_untrained_model_is_constant_velocity():
    model = _tiny_model()
    src = torch.randn(4, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.2], [12.0, -0.3], [30.0, 0.0], [7.5, 1.0]])
    out = model.rollout(src, 60, cv)
    steps = torch.arange(1, 61, dtype=torch.float32).view(1, -1, 1)
    assert torch.allclose(out['pos'], cv.unsqueeze(1) * steps, atol=0.0001)

def test_rollout_does_not_integrate_its_own_error():
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.0], [15.0, 0.0]])

    def kick(pos, step):
        return pos + torch.tensor([0.0, 50.0]) if step == 3 else pos
    clean = model.rollout(src, 12, cv)['pos']
    kicked = model.rollout(src, 12, cv, step_hook=kick)['pos']
    drift = (kicked - clean)[:, -1, :].abs().max()
    assert drift < 5.0, f'a single 50 m perturbation still moved the last step by {drift:.1f} m'

def test_idm_equilibrium_gives_zero_acceleration():
    from src.physics.idm import IDMParams, idm_acceleration
    p = IDMParams()
    v = torch.tensor([30.0])
    a = idm_acceleration(v, v0=v.clone(), gap=torch.tensor([1000000.0]), dv=torch.zeros(1), p=p)
    assert a.abs().item() < 0.001

def test_idm_brakes_when_closing_on_a_leader():
    from src.physics.idm import IDMParams, idm_acceleration
    p = IDMParams()
    a = idm_acceleration(v=torch.tensor([30.0]), v0=torch.tensor([33.0]), gap=torch.tensor([8.0]), dv=torch.tensor([10.0]), p=p)
    assert a.item() < -1.0, f'expected braking, got {a.item():.3f} m/s^2'

def test_idm_loss_prefers_a_physically_consistent_trajectory():
    from src.physics.idm import IDMParams, idm_physics_loss
    (dt, T) = (0.5, 20)
    cv = torch.tensor([[12.0, 0.0]])
    steps = torch.arange(1, T + 1, dtype=torch.float32).view(1, -1, 1)
    steady = cv.unsqueeze(1) * steps
    speeding = steady * torch.linspace(1.0, 1.6, T).view(1, -1, 1)
    kwargs = dict(cv_delta=cv, leader_gap=torch.tensor([25.0]), leader_speed=torch.tensor([24.0]), desired_speed=torch.tensor([24.0]), dt=dt, p=IDMParams())
    (steady_loss, valid) = idm_physics_loss(pred_pos=steady, **kwargs)
    (speeding_loss, _) = idm_physics_loss(pred_pos=speeding, **kwargs)
    assert valid.item() == 1.0
    assert steady_loss < speeding_loss

def test_idm_horizon_limit_ignores_later_steps():
    from src.physics.idm import IDMParams, idm_physics_loss
    T = 40
    cv = torch.tensor([[12.0, 0.0]])
    steps = torch.arange(1, T + 1, dtype=torch.float32).view(1, -1, 1)
    good = cv.unsqueeze(1) * steps
    wrecked = good.clone()
    wrecked[:, 10:, 0] += torch.linspace(0.0, 300.0, T - 10)
    kwargs = dict(cv_delta=cv, leader_gap=torch.tensor([30.0]), leader_speed=torch.tensor([24.0]), desired_speed=torch.tensor([24.0]), dt=0.5, p=IDMParams())
    (limited_good, _) = idm_physics_loss(pred_pos=good, horizon_steps=10, **kwargs)
    (limited_bad, _) = idm_physics_loss(pred_pos=wrecked, horizon_steps=10, **kwargs)
    (full_bad, _) = idm_physics_loss(pred_pos=wrecked, horizon_steps=None, **kwargs)
    assert torch.allclose(limited_good, limited_bad, atol=1e-06)
    assert full_bad > limited_bad

def test_idm_loss_is_zero_without_a_leader():
    from src.physics.idm import IDMParams, idm_physics_loss
    pos = torch.randn(3, 10, 2).cumsum(1)
    (loss, valid) = idm_physics_loss(pred_pos=pos, cv_delta=torch.tensor([[12.0, 0.0]] * 3), leader_gap=torch.full((3,), float('nan')), leader_speed=torch.full((3,), float('nan')), desired_speed=torch.full((3,), 30.0), dt=0.5, p=IDMParams())
    assert torch.isfinite(loss) and loss.item() == 0.0
    assert valid.item() == 0.0

def test_idm_gradients_stay_finite_with_tiny_gaps():
    from src.physics.idm import IDMParams, idm_physics_loss
    pos = (torch.arange(1, 21, dtype=torch.float32).view(1, -1, 1) * torch.tensor([[[6.0, 0.0]]])).repeat(4, 1, 1).requires_grad_(True)
    (loss, valid) = idm_physics_loss(pred_pos=pos, cv_delta=torch.tensor([[6.0, 0.0]] * 4), leader_gap=torch.tensor([0.6, 1.5, 3.0, 40.0]), leader_speed=torch.tensor([2.0, 2.0, 5.0, 20.0]), desired_speed=torch.tensor([25.0] * 4), dt=0.5, p=IDMParams())
    loss.backward()
    assert torch.isfinite(loss), 'loss went non-finite on congested gaps'
    assert torch.isfinite(pos.grad).all(), 'gradient went non-finite on congested gaps'
    assert pos.grad.abs().max() < 1000.0, f'gradient still huge: {pos.grad.abs().max():.1f}'

def test_idm_partial_leader_coverage_does_not_poison_the_batch():
    from src.physics.idm import IDMParams, idm_physics_loss
    pos = (torch.arange(1, 21, dtype=torch.float32).view(1, -1, 1) * torch.tensor([[[8.0, 0.0]]])).repeat(3, 1, 1).requires_grad_(True)
    (loss, valid) = idm_physics_loss(pred_pos=pos, cv_delta=torch.tensor([[8.0, 0.0]] * 3), leader_gap=torch.tensor([25.0, float('nan'), 30.0]), leader_speed=torch.tensor([15.0, float('nan'), 16.0]), desired_speed=torch.tensor([25.0, 25.0, 25.0]), dt=0.5, p=IDMParams())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pos.grad).all()
    assert abs(valid.item() - 2 / 3) < 1e-06

def test_frame_transforms_torch_match_numpy():
    from src.map.highway import from_agent_frame_t, to_agent_frame_t
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(16, 2)) * 40
    origin = rng.normal(size=(16, 2)) * 200
    theta = rng.uniform(-np.pi, np.pi, 16)
    np_out = to_agent_frame(pts[:, None, :], origin, theta)[:, 0, :]
    t_out = to_agent_frame_t(torch.tensor(pts), torch.tensor(origin), torch.tensor(theta)).numpy()
    assert np.allclose(np_out, t_out, atol=1e-09)
    back = from_agent_frame_t(torch.tensor(t_out, dtype=torch.float64), torch.tensor(origin), torch.tensor(theta)).numpy()
    assert np.allclose(back, pts, atol=1e-09)

def test_link_projection_snaps_to_lane_centres():
    from src.map.highway import LaneCentreMap, from_agent_frame_t, make_link_projection_hook
    lane_map = LaneCentreMap(torch.tensor([0.0, 3.7, 7.4, 11.1]))
    model = _tiny_model()
    b = 4
    src = torch.randn(b, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.4]] * b)
    origin = torch.randn(b, 2) * 100
    theta = torch.rand(b) * 0.2 - 0.1
    hook = make_link_projection_hook(lane_map, origin, theta, blend=1.0, start_step=0)
    pos = model.rollout(src, 12, cv, step_hook=hook)['pos']
    flat = pos.reshape(-1, 2)
    o = origin.repeat_interleave(12, dim=0)
    th = theta.repeat_interleave(12, dim=0)
    world_y = from_agent_frame_t(flat, o, th)[:, 1]
    dist = (world_y.view(-1, 1) - lane_map.centres.view(1, -1)).abs().min(dim=1).values
    assert dist.max() < 0.0001, f'max distance to a lane centre was {dist.max():.4f} m'

def test_rollout_hook_is_applied():
    model = _tiny_model()
    src = torch.randn(2, 20, NUM_FEATURES)
    cv = torch.tensor([[15.0, 0.5], [15.0, -0.5]])

    def snap_to_centreline(pos, step):
        return torch.stack([pos[:, 0], torch.zeros_like(pos[:, 1])], dim=-1)
    out = model.rollout(src, 8, cv, step_hook=snap_to_centreline)
    assert torch.allclose(out['pos'][:, :, 1], torch.zeros(2, 8), atol=1e-06)
    assert torch.allclose(out['delta'].cumsum(1), out['pos'], atol=0.0001)
