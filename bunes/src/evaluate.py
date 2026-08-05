from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data.dataset import load_splits
from .data.schema import NUM_FEATURES
from .engine import constant_velocity_baseline, evaluate
from .models.seq2seq_transformer import build_model

def plot_examples(pred_pos, true_pos, out_path: Path, n: int=6) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(n, len(pred_pos))
    (fig, axes) = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=False)
    axes = [axes] if n == 1 else axes
    for (ax, p, t) in zip(axes, pred_pos[:n], true_pos[:n]):
        ax.plot(t[:, 0], t[:, 1], '-', lw=2, label='ground truth')
        ax.plot(p[:, 0], p[:, 1], '--', lw=2, label='prediction')
        ax.scatter([0], [0], marker='o', s=30, zorder=3, label='last observed')
        ax.set_ylabel('lateral [m]')
        ax.set_ylim(-6, 6)
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[-1].set_xlabel('longitudinal [m] (agent frame)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/phase1_baseline/best.pt')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = _config_from_dict(ckpt['config'])
    device = torch.device(cfg.train.device if cfg.train.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    (datasets, _) = load_splits(cfg.data)
    loader = DataLoader(datasets[args.split], batch_size=cfg.train.batch_size * 2, shuffle=False)
    model = build_model(NUM_FEATURES, cfg.model).to(device)
    model.load_state_dict(ckpt['model'])
    report = evaluate(model, loader, cfg.train, device, pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz, desc=args.split, return_predictions=True)
    (pred_pos, true_pos) = (report.pop('_pred_pos'), report.pop('_true_pos'))
    (report.pop('_origin'), report.pop('_theta'))
    cv = constant_velocity_baseline(loader, cfg.data.target_hz, device)
    print(f'\n=== {args.split} set ({len(datasets[args.split]):,} windows) ===')
    print(f"{'metric':<16}{'PAS-LGT (Phase 1)':>20}{'const-velocity':>18}")
    for k in sorted(report):
        print(f"{k:<16}{report[k]:>20.3f}{cv.get(k, float('nan')):>18.3f}")
    out_dir = Path(args.ckpt).parent
    with open(out_dir / f'metrics_{args.split}.json', 'w') as f:
        json.dump({'model': report, 'constant_velocity': cv}, f, indent=2)
    if args.plot:
        plot_examples(pred_pos, true_pos, out_dir / f'examples_{args.split}.png')

def _config_from_dict(d: dict) -> Config:
    from .config import DataConfig, ModelConfig, PhysicsConfig, TrainConfig
    return Config(data=DataConfig(**d['data']), model=ModelConfig(**d['model']), train=TrainConfig(**d['train']), physics=PhysicsConfig(**d.get('physics', {})))
if __name__ == '__main__':
    main()
