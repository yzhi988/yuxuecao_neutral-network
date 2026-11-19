"""Training script for SOH regression.

This script trains a neural network to predict battery State of Health (SOH).

Components:
- Data pipeline: MITdata dataloader
- Model: physics-temporal decoupled encoder + prediction head
- Objective: data loss (MSE) + optional feature-space regularization
- Evaluation: MAE / MAPE / MSE / RMSE / R2
- Repeated runs: run multiple independent trainings with different random seeds
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from dataloader.dataloader import MITdata
from utils.util import AverageMeter, get_logger, eval_metrix

from Model.SOHModel_final import ModelConfig, SOHRegressor, get_device


def set_seed(seed: int) -> None:
    """Best-effort reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Final SOH model training')

    # data
    p.add_argument('--data', type=str, default='MIT')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--normalization_method', type=str, default='robust', choices=['min-max', 'z-score', 'robust'])

    # training
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--early_stop', type=int, default=30)
    p.add_argument('--warmup_lr', type=float, default=1e-3)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--final_lr', type=float, default=5e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # model
    p.add_argument('--input_dim', type=int, default=17)
    p.add_argument('--enc_layers', type=int, default=3)
    p.add_argument('--enc_hidden_dim', type=int, default=128)
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--enc_dropout', type=float, default=0.2)
    p.add_argument('--head_dropout', type=float, default=0.2)

    # feature reg
    p.add_argument('--disable_feature_reg', action='store_true')
    p.add_argument('--lambda_smooth', type=float, default=0.01)
    p.add_argument('--lambda_energy', type=float, default=0.005)
    p.add_argument('--lambda_causal', type=float, default=0.005)


    # repeated runs
    p.add_argument('--num_runs', type=int, default=5, help='Number of repeated runs (different seeds)')
    p.add_argument('--seed', type=int, default=42, help='Base random seed; run_i uses seed+ i')

    # logging/saving
    p.add_argument('--log_dir', type=str, default='logging.txt')
    p.add_argument('--save_folder', type=str, default='results/final_model')

    return p.parse_args()


def load_MIT_data(args: argparse.Namespace, small_sample: int | None = None):
    root = 'data/MIT data'
    train_list, test_list = [], []

    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            battery_id = int(f.split('-')[-1].split('.')[0])
            if battery_id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))

    if small_sample is not None:
        train_list = train_list[:small_sample]

    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)

    return {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3'],
    }


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = get_device()

        os.makedirs(args.save_folder, exist_ok=True)
        log_path = os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_path)

        cfg = ModelConfig(
            input_dim=args.input_dim,
            enc_layers=args.enc_layers,
            enc_hidden_dim=args.enc_hidden_dim,
            emb_dim=args.emb_dim,
            enc_dropout=args.enc_dropout,
            head_dropout=args.head_dropout,
            feature_reg=(not args.disable_feature_reg),
            lambda_smooth=args.lambda_smooth,
            lambda_energy=args.lambda_energy,
            lambda_causal=args.lambda_causal,
        )

        self.model = SOHRegressor(cfg).to(self.device)
        self.mse = nn.MSELoss()

        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=args.epochs,
            pct_start=0.3,
            div_factor=args.lr / args.warmup_lr,
            final_div_factor=args.warmup_lr / args.final_lr,
        )

        self.best_model = None
        self._save_config(cfg)

    def _save_config(self, cfg: ModelConfig) -> None:
        payload = {
            'args': vars(self.args),
            'model_config': asdict(cfg),
            'device': self.device,
            'run': {
                'run_id': getattr(self.args, 'run_id', None),
                'seed': getattr(self.args, 'seed', None)
            }
        }
        with open(os.path.join(self.args.save_folder, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def train_one_epoch(self, epoch: int, loader) -> dict:
        self.model.train()

        data_meter = AverageMeter()
        reg_meter = AverageMeter()

        for x1, x2, y1, y2 in loader:
            x1, x2 = x1.to(self.device), x2.to(self.device)
            y1, y2 = y1.to(self.device), y2.to(self.device)

            u1, h1, _ = self.model(x1)
            u2, h2, _ = self.model(x2)

            loss_data = 0.5 * (self.mse(u1, y1) + self.mse(u2, y2))

            loss_reg = torch.zeros((), device=self.device)
            if self.model.feature_regularizer is not None:
                t1, t2 = x1[:, -1:], x2[:, -1:]
                reg = self.model.feature_regularizer(h1, h2, t1, t2)
                loss_reg = reg['total_reg']

            loss = loss_data + loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            data_meter.update(loss_data.item())
            reg_meter.update(loss_reg.item())

        return {'data': data_meter.avg, 'reg': reg_meter.avg}

    @torch.no_grad()
    def evaluate_mse(self, loader) -> float:
        self.model.eval()
        losses = []
        for x1, _, y1, _ in loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)
            u1, _, _ = self.model(x1)
            losses.append(self.mse(u1, y1).item())
        return float(np.mean(losses))

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        true_labels, pred_labels = [], []
        for x1, _, y1, _ in loader:
            x1 = x1.to(self.device)
            u1, _, _ = self.model(x1)
            true_labels.append(y1.numpy())
            pred_labels.append(u1.cpu().numpy())
        return np.concatenate(true_labels), np.concatenate(pred_labels)

    def fit(self, trainloader, validloader, testloader):
        min_valid = float('inf')
        early_stop_counter = 0

        for epoch in range(1, self.args.epochs + 1):
            losses = self.train_one_epoch(epoch, trainloader)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"[Train] epoch:{epoch} lr:{lr:.6f} | data:{losses['data']:.6f} | reg:{losses['reg']:.6f}"
            )

            valid_mse = self.evaluate_mse(validloader)
            self.logger.info(f"[Valid] epoch:{epoch} MSE:{valid_mse:.6f}")

            if valid_mse < min_valid:
                min_valid = valid_mse
                early_stop_counter = 0

                true_label, pred_label = self.test(testloader)
                mae, mape, mse, rmse, r2 = eval_metrix(pred_label, true_label)
                self.logger.info(
                    f"[Test] MSE:{mse:.8f}, MAE:{mae:.6f}, MAPE:{mape:.6f}, RMSE:{rmse:.6f}, R2:{r2:.6f}"
                )

                np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)

                self.best_model = {
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'valid_mse': valid_mse,
                    'test_metrics': {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2},
                }
                torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))
            else:
                early_stop_counter += 1

            if self.args.early_stop is not None and early_stop_counter > self.args.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self._clear_logger()

    def _clear_logger(self):
        if len(self.logger.handlers) > 0:
            self.logger.removeHandler(self.logger.handlers[0])
            self.logger.handlers.clear()


def main():
    args = get_args()

    base_save_folder = args.save_folder
    os.makedirs(base_save_folder, exist_ok=True)

    # Load data once; reuse across runs
    dataloader = load_MIT_data(args)

    all_metrics = []

    for run_id in range(1, args.num_runs + 1):
        run_seed = int(args.seed) + run_id
        set_seed(run_seed)

        run_folder = base_save_folder if args.num_runs == 1 else os.path.join(base_save_folder, f'run_{run_id}')
        os.makedirs(run_folder, exist_ok=True)

        # create a shallow args-like object for this run
        run_args = argparse.Namespace(**vars(args))
        run_args.save_folder = run_folder
        run_args.run_id = run_id
        run_args.seed = run_seed

        trainer = Trainer(run_args)
        trainer.fit(dataloader['train'], dataloader['valid'], dataloader['test'])

        if trainer.best_model is not None:
            m = trainer.best_model['test_metrics']
            all_metrics.append(m)

    # summarize if repeated runs
    if args.num_runs > 1 and len(all_metrics) > 0:
        def _mean_std(xs):
            xs = np.array(xs, dtype=float)
            return float(xs.mean()), float(xs.std())

        mse_mean, mse_std = _mean_std([m['MSE'] for m in all_metrics])
        mae_mean, mae_std = _mean_std([m['MAE'] for m in all_metrics])
        rmse_mean, rmse_std = _mean_std([m['RMSE'] for m in all_metrics])
        r2_mean, r2_std = _mean_std([m['R2'] for m in all_metrics])
        mape_mean, mape_std = _mean_std([m['MAPE'] for m in all_metrics])

        summary = {
            'num_runs': len(all_metrics),
            'MSE': {'mean': mse_mean, 'std': mse_std},
            'MAE': {'mean': mae_mean, 'std': mae_std},
            'RMSE': {'mean': rmse_mean, 'std': rmse_std},
            'MAPE': {'mean': mape_mean, 'std': mape_std},
            'R2': {'mean': r2_mean, 'std': r2_std},
        }

        with open(os.path.join(base_save_folder, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('Statistical Summary saved to:', os.path.join(base_save_folder, 'summary.json'))
        print('MSE :', f"{mse_mean:.8f} ± {mse_std:.8f}")
        print('MAE :', f"{mae_mean:.6f} ± {mae_std:.6f}")
        print('RMSE:', f"{rmse_mean:.6f} ± {rmse_std:.6f}")
        print('MAPE:', f"{mape_mean:.6f} ± {mape_std:.6f}")
        print('R2  :', f"{r2_mean:.6f} ± {r2_std:.6f}")


if __name__ == '__main__':
    main()

