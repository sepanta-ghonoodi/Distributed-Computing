from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config, DataConfig, ModelConfig, PhysicsConfig, TrainConfig
from .data.dataset import load_splits
from .data.schema import NUM_FEATURES
from .engine import constant_velocity_baseline, evaluate
from .map.highway import LaneCentreMap, make_link_projection_hook, off_road_rate
from .models.seq2seq_transformer import build_model
REPORT_KEYS = ['ade', 'fde', 'ade@10s', 'ade@30s', 'rmse_long', 'rmse_lat', 'miss_rate@5m']

def config_from_dict(d: dict) -> Config:
    return Config(data=DataConfig(**d['data']), model=ModelConfig(**d['model']), train=TrainConfig(**d['train']), physics=PhysicsConfig(**d.get('physics', {})))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/phase1_colab/best.pt')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--blends', type=float, nargs='+', default=[0.0, 0.5, 1.0], help='Link-Projection strengths to compare (0 = Phase 1 baseline)')
    ap.add_argument('--start-step', type=int, default=2, help='skip snapping for the first N steps (vehicle may be mid-lane-change)')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = config_from_dict(ckpt['config'])
    device = torch.device('cuda' if cfg.train.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    (datasets, _) = load_splits(cfg.data)
    loader = DataLoader(datasets[args.split], batch_size=cfg.train.batch_size * 2, shuffle=False)
    model = build_model(NUM_FEATURES, cfg.model).to(device)
    model.load_state_dict(ckpt['model'])
    lane_map = LaneCentreMap.from_parquet(cfg.data.unified_path).to(device)
    print(f'lane map | {len(lane_map.centres)} centrelines at {[round(float(c), 2) for c in lane_map.centres]} m, mean spacing {lane_map.lane_width:.2f} m\n')
    rows: dict[str, dict[str, float]] = {}
    preds_for_plot: dict[str, torch.Tensor] = {}
    truth = origin = theta = None
    for blend in args.blends:
        name = 'Phase 1 (no snap)' if blend == 0.0 else f'+ LinkProj b={blend:g}'

        def factory(batch, b=blend):
            if b == 0.0:
                return None
            return make_link_projection_hook(lane_map, batch['origin'], batch['theta'], blend=b, start_step=args.start_step)
        rep = evaluate(model, loader, cfg.train, device, pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz, hook_factory=factory, desc=name, return_predictions=True)
        pred = rep.pop('_pred_pos')
        truth = rep.pop('_true_pos')
        (origin, theta) = (rep.pop('_origin'), rep.pop('_theta'))
        rep['off_road'] = off_road_rate(pred, origin, theta, lane_map.to('cpu'))
        lane_map.to(device)
        rows[name] = rep
        preds_for_plot[name] = pred
    rows['constant velocity'] = constant_velocity_baseline(loader, cfg.data.target_hz, device)
    cv_pos = []
    for batch in loader:
        k = torch.arange(1, cfg.data.pred_len + 1, dtype=torch.float32).view(1, -1, 1)
        cv_pos.append(batch['cv_delta'].unsqueeze(1) * k)
    preds_for_plot['constant velocity'] = torch.cat(cv_pos, 0)
    keys = REPORT_KEYS + ['off_road']
    width = max((len(n) for n in rows)) + 2
    header = f"{'model':<{width}}" + ''.join((f'{k:>13}' for k in keys))
    print(f'\n=== {args.split} set ({len(datasets[args.split]):,} windows) ===')
    print(header)
    print('-' * len(header))
    for (name, rep) in rows.items():
        cells = ''.join((f"{rep.get(k, float('nan')):>13.3f}" for k in keys))
        print(f'{name:<{width}}{cells}')
    out_dir = Path(args.ckpt).parent
    with open(out_dir / f'ablation_{args.split}.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_dir / f'ablation_{args.split}.json'}")
    base = rows['Phase 1 (no snap)']
    share = base['rmse_lat'] ** 2 / (base['rmse_lat'] ** 2 + base['rmse_long'] ** 2)
    print(f'\nerror budget | longitudinal {100 * (1 - share):.1f}%  lateral {100 * share:.1f}%\nLink Projection acts only on the lateral share, so it drives the\noff-road rate towards zero without moving ADE much. The longitudinal\nshare is where the ADE gap lives; the IDM regulariser of Phase 3 was\nour attempt at it and made things worse on both datasets, so this\nremains open.')
    if args.plot:
        plot_error_curves(preds_for_plot, truth, cfg.data.target_hz, out_dir / f'ablation_{args.split}.png')
        plot_lateral_zoom(preds_for_plot, truth, cfg.data.target_hz, out_dir / f'ablation_lateral_{args.split}.png')

def plot_error_curves(preds: dict[str, torch.Tensor], truth: torch.Tensor, target_hz: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = torch.arange(1, truth.size(1) + 1, dtype=torch.float32) / target_hz
    (fig, axes) = plt.subplots(1, 3, figsize=(14, 4))
    for (name, p) in preds.items():
        resid = p - truth
        style = dict(lw=2.0, ls='-' if 'constant' not in name else '--')
        axes[0].plot(t, torch.linalg.vector_norm(resid, dim=-1).mean(0), label=name, **style)
        axes[1].plot(t, resid[..., 0].pow(2).mean(0).sqrt(), label=name, **style)
        axes[2].plot(t, resid[..., 1].pow(2).mean(0).sqrt(), label=name, **style)
    titles = ['total displacement error', 'longitudinal RMSE  (Phase 3 target)', 'lateral RMSE  (Phase 2 target)']
    for (ax, title) in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('prediction horizon [s]')
        ax.set_ylabel('error [m]')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')

def plot_lateral_zoom(preds: dict[str, torch.Tensor], truth: torch.Tensor, target_hz: float, out_path: Path, n: int=4) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    span = truth[..., 1].max(dim=1).values - truth[..., 1].min(dim=1).values
    idx = span.argsort(descending=True)[:n]
    t = torch.arange(1, truth.size(1) + 1, dtype=torch.float32) / target_hz
    (fig, axes) = plt.subplots(n, 1, figsize=(10, 2.3 * n), sharex=True)
    axes = [axes] if n == 1 else axes
    for (ax, i) in zip(axes, idx):
        ax.plot(t, truth[i, :, 1], '-', lw=2.5, color='k', label='ground truth')
        for (name, p) in preds.items():
            if 'constant' in name:
                continue
            ax.plot(t, p[i, :, 1], '--', lw=1.6, label=name)
        ax.set_ylabel('lateral [m]')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('prediction horizon [s]')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')
if __name__ == '__main__':
    main()
