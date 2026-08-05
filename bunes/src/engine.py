from __future__ import annotations
import math
from typing import Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import PhysicsConfig, TrainConfig
from .metrics import full_report
from .models.seq2seq_transformer import StepHook, TrajectoryTransformer
from .physics.idm import IDMParams, idm_physics_loss

def trajectory_loss(pred_pos: torch.Tensor, tgt_pos: torch.Tensor, kind: str='huber_position') -> torch.Tensor:
    if kind == 'mse_position':
        return nn.functional.mse_loss(pred_pos, tgt_pos)
    if kind == 'huber_position':
        return nn.functional.smooth_l1_loss(pred_pos, tgt_pos, beta=1.0)
    if kind == 'ade':
        return torch.linalg.vector_norm(pred_pos - tgt_pos, dim=-1).mean()
    raise ValueError(f"Unknown loss '{kind}'")

def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total = max(1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model: TrajectoryTransformer, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler, scaler: torch.amp.GradScaler, cfg: TrainConfig, device: torch.device, epoch: int, physics: PhysicsConfig | None=None, dt: float=0.5) -> dict[str, float]:
    model.train()
    (running, running_data, running_phy, running_valid, n) = (0.0, 0.0, 0.0, 0.0, 0)
    use_physics = physics is not None and physics.weight > 0.0
    idm_params = IDMParams(a_max=physics.a_max, b_comf=physics.b_comf, s0=physics.s0, t_headway=physics.t_headway, delta=physics.delta, a_clip=physics.a_clip) if use_physics else None
    ss_p = cfg.scheduled_sampling * min(1.0, epoch / max(1, cfg.ss_ramp_epochs))
    bar = tqdm(loader, desc=f'epoch {epoch:03d} [train]', leave=False)
    for batch in bar:
        batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == 'cuda'):
            decoder_truth = batch['tgt_pos']
            if ss_p > 0.0:
                with torch.no_grad():
                    guess = model(batch['src'], batch['tgt_pos'], batch['cv_delta'])
                take_guess = torch.rand_like(guess[..., :1]) < ss_p
                decoder_truth = torch.where(take_guess, guess.detach(), batch['tgt_pos'])
            pred_pos = model(batch['src'], decoder_truth, batch['cv_delta'])
            data_loss = trajectory_loss(pred_pos, batch['tgt_pos'], cfg.loss)
            if use_physics:
                (phy_loss, valid) = idm_physics_loss(pred_pos.float(), batch['cv_delta'].float(), batch['leader_gap'].float(), batch['leader_speed'].float(), batch['desired_speed'].float(), dt=dt, p=idm_params, min_gap=physics.min_gap, horizon_steps=physics.horizon_steps)
                loss = data_loss + physics.weight * phy_loss
            else:
                phy_loss = torch.zeros((), device=device)
                valid = torch.zeros((), device=device)
                loss = data_loss
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        bs = batch['src'].size(0)
        running += loss.item() * bs
        running_data += data_loss.item() * bs
        running_phy += phy_loss.detach().item() * bs
        running_valid += valid.detach().item() * bs
        n += bs
        bar.set_postfix(loss=f'{running / max(1, n):.3f}', phy=f'{running_phy / max(1, n):.3f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
    return {'train_loss': running / max(1, n), 'train_data_loss': running_data / max(1, n), 'ss_prob': ss_p, 'train_phy_loss': running_phy / max(1, n), 'leader_coverage': running_valid / max(1, n)}

@torch.no_grad()
def evaluate(model: TrajectoryTransformer, loader: DataLoader, cfg: TrainConfig, device: torch.device, pred_len: int, target_hz: float, hook_factory: Callable[[dict[str, torch.Tensor]], StepHook | None] | None=None, desc: str='eval', return_predictions: bool=False) -> dict[str, Any]:
    model.eval()
    (preds, trues, origins, thetas) = ([], [], [], [])
    for batch in tqdm(loader, desc=f'[{desc}]', leave=False):
        batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}
        hook = hook_factory(batch) if hook_factory is not None else None
        out = model.rollout(batch['src'], pred_len, batch['cv_delta'], step_hook=hook)
        preds.append(out['pos'].float().cpu())
        trues.append(batch['tgt_pos'].float().cpu())
        origins.append(batch['origin'].float().cpu())
        thetas.append(batch['theta'].float().cpu())
    pred_pos = torch.cat(preds, 0)
    true_pos = torch.cat(trues, 0)
    report: dict[str, Any] = full_report(pred_pos, true_pos, target_hz)
    if return_predictions:
        report['_pred_pos'] = pred_pos
        report['_true_pos'] = true_pos
        report['_origin'] = torch.cat(origins, 0)
        report['_theta'] = torch.cat(thetas, 0)
    return report

@torch.no_grad()
def constant_velocity_baseline(loader: DataLoader, target_hz: float, device: torch.device) -> dict[str, float]:
    (preds, trues) = ([], [])
    for batch in loader:
        cv = batch['cv_delta'].to(device)
        steps = batch['tgt_pos'].size(1)
        idx = torch.arange(1, steps + 1, device=device, dtype=cv.dtype)
        pos = cv.unsqueeze(1) * idx.view(1, -1, 1)
        preds.append(pos.float().cpu())
        trues.append(batch['tgt_pos'].float())
    return full_report(torch.cat(preds, 0), torch.cat(trues, 0), target_hz)
