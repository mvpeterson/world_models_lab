"""Two-stage split-update gated residual predictive autoencoder with EMA target encoder.
---------------
Core motivation
---------------
The fixed-latent split-update model was the first stable regime:
- stage 1 pretrains a plain autoencoder
- stage 2 freezes the latent space and learns corruption-induced latent updates

But once the same encoder is both:
1) the object that defines delta_tgt = z_cor - z0, and
2) the object receiving gradients from the predictive objective,

stage-2 fine-tuning becomes unstable.

This script is the next clean variant:
- keep the split-update predictor
- keep weighted delta regression + cosine alignment
- introduce a *target encoder* that is never optimized directly
- update that target encoder by EMA from the online encoder

So stage 2 becomes:
- online encoder/decoder/gate/predictors are trainable
- target encoder defines the corruption-induced latent target
- online encoder is softly anchored to the target encoder

----------------------------------
Probe-selected checkpoint variant.
----------------------------------

This script is based on the split-update two-stage AE trainer, but changes the
stage-2 model-selection logic:
- final checkpoint can still be saved as usual
- stage-2 best checkpoint selection is based on validation linear-probe accuracy
  instead of reconstruction BCE

Why this matters:
- reconstruction can stay strong while representation quality drifts
- if downstream linear separability is the real objective, stage-2 checkpoint
  selection should follow probe accuracy directly

----------------
Stage 2 sketch
----------------
Online branch:
    z0_online = encoder(x)
    g = sigmoid(gate(z0_online))
    g_eff = gate_floor + (1 - gate_floor) * g
    d_base = predictor_base(z0_online)
    d_gate = predictor_gate(z0_online)
    delta_pred = delta_scale * (d_base + g_eff * d_gate)
    z_cor_pred = z0_online + delta_pred

Target branch (no grad):
    z0_tgt = target_encoder(x)
    z_cor_tgt = target_encoder(x_cor)
    delta_tgt = z_cor_tgt - z0_tgt

Primary stage-2 losses:
    loss_pred   = mse(delta_pred, delta_tgt)
    loss_cos    = 1 - cosine(delta_pred, delta_tgt)
    loss_mag    = mse(mean_abs(delta_pred), mean_abs(delta_tgt))
    loss_anchor = mse(z0_online, z0_tgt)

Optional reconstruction remains on the online autoencoder:
    loss_recon = BCE(decoder(z0_online), x)

Selection metric:
    probe_acc = frozen linear probe accuracy on the test split

-------------
Example:
-------------

python3 test_ae_gate_residual_probe_select.py \
  --stage1-epochs 20 \
  --stage2-epochs 40 \
  --run-mode both \
  --freeze-decoder-stage2 1 \
  --stage2-lr 1e-4 \
  --delta-scale 0.12 \
  --pred-loss mse \
  --pred-cos-weight 0.3 \
  --delta-l2-weight 1e-5 \
  --ema-decay 0.995 \
  --latent-anchor-weight 0.1

"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_mnist import ClassifierEvalCfg, corrupt_batch, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402
from common_conv import ConvDecoder, ConvEncoder, ConvPredictor, ConvStackCfg, UnitKernelConv2d  # noqa: E402


class ResidualConvGate(nn.Module):
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(z))


class TwoStageSplitUpdateEMATargetGateAE(nn.Module):
    def __init__(
        self,
        cfg: ConvStackCfg,
        pred_hidden: int | None = None,
        delta_scale: float = 1.0,
        gate_floor: float = 0.25,
        gate_bias: bool = True,
        bound_delta: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.delta_scale = float(delta_scale)
        self.gate_floor = float(gate_floor)
        self.bound_delta = bool(bound_delta)

        self.encoder = ConvEncoder(UnitKernelConv2d, cfg)
        self.target_encoder = ConvEncoder(UnitKernelConv2d, cfg)
        self.decoder = ConvDecoder(cfg, out_channels=cfg.in_channels)

        c = cfg.channels[-1]
        self.gate = ResidualConvGate(c, bias=gate_bias)
        self.predictor_base = ConvPredictor(
            c,
            hidden=pred_hidden,
            nonlinearity=cfg.nonlinearity,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.predictor_gate = ConvPredictor(
            c,
            hidden=pred_hidden,
            nonlinearity=cfg.nonlinearity,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.init_target_encoder_from_online()
        self.freeze_target_encoder()

    @torch.no_grad()
    def init_target_encoder_from_online(self) -> None:
        self.target_encoder.load_state_dict(copy.deepcopy(self.encoder.state_dict()))

    def freeze_target_encoder(self) -> None:
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder_ema(self, decay: float) -> None:
        for p_tgt, p_online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_tgt.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def compute_gate(self, z0: torch.Tensor) -> torch.Tensor:
        return self.gate(z0)

    def compute_gate_effective(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.compute_gate(z0)
        g_eff = self.gate_floor + (1.0 - self.gate_floor) * g
        return g, g_eff

    def predict_base_delta(self, z0: torch.Tensor) -> torch.Tensor:
        d_base = self.predictor_base(z0)
        if self.bound_delta:
            d_base = torch.tanh(d_base)
        return d_base

    def predict_gated_delta(self, z0: torch.Tensor) -> torch.Tensor:
        d_gate = self.predictor_gate(z0)
        if self.bound_delta:
            d_gate = torch.tanh(d_gate)
        return d_gate

    def predict_corrupted_latent(self, z0: torch.Tensor):
        g, g_eff = self.compute_gate_effective(z0)
        d_base = self.predict_base_delta(z0)
        d_gate = self.predict_gated_delta(z0)
        delta_pred = self.delta_scale * (d_base + g_eff * d_gate)
        z_cor_pred = z0 + delta_pred
        return z_cor_pred, delta_pred, g, g_eff, d_base, d_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z0 = self.encode_full(x)
        return self.decode(z0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_full(x)


def save_checkpoint(path: str, model: nn.Module, cfg: ConvStackCfg, args: Any, stage: str, extra: dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "stage": stage,
        "model_state": model.state_dict(),
        "cfg": {
            "in_channels": cfg.in_channels,
            "channels": tuple(cfg.channels),
            "kernel_size": cfg.kernel_size,
            "stride": cfg.stride,
            "padding": cfg.padding,
            "nonlinearity": cfg.nonlinearity,
        },
        "args": vars(args).copy(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"[ckpt] saved: {path}")


def load_checkpoint(path: str, model: nn.Module, device: torch.device, strict: bool = True):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    print(f"[ckpt] loaded: {path} (stage={ckpt.get('stage', 'unknown')})")
    return ckpt


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def make_optimizer(model: TwoStageSplitUpdateEMATargetGateAE, lr: float, wd: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected for optimizer")
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


@torch.no_grad()
def eval_recon_bce(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    for x, _y in loader:
        x = x.to(device)
        x_hat = model(x)
        loss_sum += float(F.binary_cross_entropy(x_hat, x, reduction="sum").item())
        n += int(x.numel())
    return loss_sum / max(1, n)


def eval_probe_preserve_state(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    probe_cfg: ClassifierEvalCfg,
) -> float:
    was_training = model.training
    req_grad = [p.requires_grad for p in model.parameters()]

    acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)

    model.train(was_training)
    for p, flag in zip(model.parameters(), req_grad):
        p.requires_grad_(flag)
    return acc


def train_stage1_ae(
    model: TwoStageSplitUpdateEMATargetGateAE,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
):
    set_requires_grad(model.encoder, True)
    set_requires_grad(model.decoder, True)
    set_requires_grad(model.gate, False)
    set_requires_grad(model.predictor_base, False)
    set_requires_grad(model.predictor_gate, False)
    set_requires_grad(model.target_encoder, False)

    model.to(device)
    opt = make_optimizer(model, lr=lr, wd=wd)

    for ep in range(1, epochs + 1):
        model.train()
        recon_avg = 0.0
        n = 0

        for x, _y in train_loader:
            x = x.to(device)
            z0 = model.encode_full(x)
            x_hat = model.decode(z0)
            loss_recon = F.binary_cross_entropy(x_hat, x)

            opt.zero_grad(set_to_none=True)
            loss_recon.backward()
            opt.step()

            recon_avg += float(loss_recon.item())
            n += 1

        model.init_target_encoder_from_online()
        model.freeze_target_encoder()
        te = eval_recon_bce(model, test_loader, device)
        print(f"[stage1] epoch {ep:03d} | recon(bce)={recon_avg/max(1,n):.6f} | recon_eval/test={te:.6f}")


def train_stage2_predictor(
    model: TwoStageSplitUpdateEMATargetGateAE,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    corrupt_mode: str,
    corrupt_max_frac: float,
    corrupt_noise_std: float,
    freeze_encoder: bool,
    freeze_decoder: bool,
    pred_weight: float,
    pred_loss: str,
    pred_weight_power: float,
    pred_weight_clip: float,
    pred_cos_weight: float,
    gate_binarize_weight: float,
    gate_l1_weight: float,
    delta_l2_weight: float,
    gate_branch_l2_weight: float,
    recon_weight: float,
    latent_anchor_weight: float,
    latent_pred_match_weight: float,
    pred_mag_weight: float,
    ema_decay: float,
    probe_select_stage2: bool,
    probe_every: int,
    probe_cfg: ClassifierEvalCfg,
    best_stage2_save_path: str,
    args: Any,
):
    set_requires_grad(model.encoder, not freeze_encoder)
    set_requires_grad(model.decoder, not freeze_decoder)
    set_requires_grad(model.gate, True)
    set_requires_grad(model.predictor_base, True)
    set_requires_grad(model.predictor_gate, True)
    set_requires_grad(model.target_encoder, False)
    model.freeze_target_encoder()

    model.to(device)
    opt = make_optimizer(model, lr=lr, wd=wd)

    best_probe_acc = float("-inf")
    best_probe_epoch = -1
    best_probe_recon = float("nan")

    for ep in range(1, epochs + 1):
        model.train()
        recon_avg = 0.0
        pred_avg = 0.0
        gate_avg = 0.0
        gate_bin_avg = 0.0
        gate_std_avg = 0.0
        gate_min_avg = 0.0
        gate_max_avg = 0.0
        dbase_abs_avg = 0.0
        dgate_abs_avg = 0.0
        delta_abs_avg = 0.0
        delta_tgt_abs_avg = 0.0
        delta_ratio_avg = 0.0
        delta_cos_avg = 0.0
        loss_cos_avg = 0.0
        delta_std_avg = 0.0
        delta_tgt_std_avg = 0.0
        delta_avg = 0.0
        pred_w_avg = 0.0
        pred_w_max_avg = 0.0
        anchor_avg = 0.0
        pred_match_avg = 0.0
        target_drift_avg = 0.0
        mag_loss_avg = 0.0
        n = 0

        for x, _y in train_loader:
            x = x.to(device)
            x_cor = corrupt_batch(x, mode=corrupt_mode, max_frac=corrupt_max_frac, noise_std=corrupt_noise_std)

            z0_online = model.encode_full(x)
            x_hat = model.decode(z0_online)
            loss_recon = F.binary_cross_entropy(x_hat, x)

            with torch.no_grad():
                z0_tgt = model.encode_target(x)
                z_cor_tgt = model.encode_target(x_cor)
                delta_tgt = z_cor_tgt - z0_tgt

            z_cor_pred, delta_pred, g, _g_eff, d_base, d_gate = model.predict_corrupted_latent(z0_online)

            if pred_loss == "mse":
                pred_w = torch.ones_like(delta_tgt)
                loss_pred = F.mse_loss(delta_pred, delta_tgt)
            elif pred_loss == "smooth_l1":
                pred_w = torch.ones_like(delta_tgt)
                loss_pred = F.smooth_l1_loss(delta_pred, delta_tgt)
            elif pred_loss == "weighted_mse":
                pred_w = (delta_tgt.abs() + 1e-8).pow(pred_weight_power)
                pred_w = pred_w / (pred_w.mean().detach() + 1e-8)
                if pred_weight_clip > 0:
                    pred_w = pred_w.clamp(max=pred_weight_clip)
                loss_pred = (pred_w * (delta_pred - delta_tgt).pow(2)).mean()
            else:
                raise ValueError(f"Unknown pred_loss: {pred_loss}")

            delta_pred_flat = delta_pred.flatten(start_dim=1)
            delta_tgt_flat = delta_tgt.flatten(start_dim=1)
            loss_cos = 1.0 - F.cosine_similarity(delta_pred_flat, delta_tgt_flat, dim=1, eps=1e-8).mean()

            loss_gate_binarize = (g * (1.0 - g)).mean()
            loss_gate = g.mean()
            loss_delta = delta_pred.pow(2).mean()
            loss_gate_branch = d_gate.pow(2).mean()
            loss_anchor = F.mse_loss(z0_online, z0_tgt)
            loss_pred_match = F.mse_loss(z_cor_pred, z_cor_tgt)
            mag_pred = delta_pred.abs().mean(dim=(1, 2, 3))
            mag_tgt = delta_tgt.abs().mean(dim=(1, 2, 3))
            loss_mag = F.mse_loss(mag_pred, mag_tgt)

            loss = (
                recon_weight * loss_recon
                + pred_weight * loss_pred
                + pred_cos_weight * loss_cos
                + pred_mag_weight * loss_mag
                + gate_binarize_weight * loss_gate_binarize
                + gate_l1_weight * loss_gate
                + delta_l2_weight * loss_delta
                + gate_branch_l2_weight * loss_gate_branch
                + latent_anchor_weight * loss_anchor
                + latent_pred_match_weight * loss_pred_match
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_target_encoder_ema(decay=ema_decay)

            recon_avg += float(loss_recon.item())
            pred_avg += float(loss_pred.item())
            gate_avg += float(loss_gate.item())
            gate_bin_avg += float(loss_gate_binarize.item())
            gate_std_avg += float(g.std(unbiased=False).item())
            gate_min_avg += float(g.min().item())
            gate_max_avg += float(g.max().item())
            dbase_abs_avg += float(d_base.abs().mean().item())
            dgate_abs_avg += float(d_gate.abs().mean().item())
            delta_abs_avg += float(delta_pred.abs().mean().item())
            delta_tgt_abs_avg += float(delta_tgt.abs().mean().item())
            delta_ratio_avg += float((delta_pred.abs().mean() / (delta_tgt.abs().mean() + 1e-8)).item())
            delta_std_avg += float(delta_pred.std(unbiased=False).item())
            delta_tgt_std_avg += float(delta_tgt.std(unbiased=False).item())
            cos = F.cosine_similarity(delta_pred_flat, delta_tgt_flat, dim=1, eps=1e-8).mean()
            delta_cos_avg += float(cos.item())
            loss_cos_avg += float(loss_cos.item())
            delta_avg += float(loss_delta.item())
            pred_w_avg += float(pred_w.mean().item())
            pred_w_max_avg += float(pred_w.max().item())
            anchor_avg += float(loss_anchor.item())
            pred_match_avg += float(loss_pred_match.item())
            target_drift_avg += float((z0_online.detach() - z0_tgt).abs().mean().item())
            mag_loss_avg += float(loss_mag.item())
            n += 1

        te = eval_recon_bce(model, test_loader, device)
        log = (
            f"[stage2-ema] epoch {ep:03d} | "
            f"recon(bce)={recon_avg/max(1,n):.6f} | "
            f"pred(mse)={pred_avg/max(1,n):.6f} | "
            f"gate(mean)={gate_avg/max(1,n):.6f} | "
            f"gate(bin)={gate_bin_avg/max(1,n):.6f} | "
            f"gate(std)={gate_std_avg/max(1,n):.6f} | "
            f"gate(min)={gate_min_avg/max(1,n):.6f} | "
            f"gate(max)={gate_max_avg/max(1,n):.6f} | "
            f"d_base(abs)={dbase_abs_avg/max(1,n):.3e} | "
            f"d_gate(abs)={dgate_abs_avg/max(1,n):.3e} | "
            f"delta(abs)={delta_abs_avg/max(1,n):.3e} | "
            f"delta_tgt(abs)={delta_tgt_abs_avg/max(1,n):.3e} | "
            f"delta(ratio)={delta_ratio_avg/max(1,n):.3e} | "
            f"delta(cos)={delta_cos_avg/max(1,n):.4f} | "
            f"loss(cos)={loss_cos_avg/max(1,n):.4f} | "
            f"delta(std)={delta_std_avg/max(1,n):.3e} | "
            f"delta_tgt(std)={delta_tgt_std_avg/max(1,n):.3e} | "
            f"delta(reg)={delta_avg/max(1,n):.3e} | "
            f"pred_w(mean)={pred_w_avg/max(1,n):.3e} | "
            f"pred_w(max)={pred_w_max_avg/max(1,n):.3e} | "
            f"latent(anchor)={anchor_avg/max(1,n):.3e} | "
            f"latent(pred_match)={pred_match_avg/max(1,n):.3e} | "
            f"delta(mag_loss)={mag_loss_avg/max(1,n):.3e} | "
            f"latent(drift)={target_drift_avg/max(1,n):.3e} | "
            f"recon_eval/test={te:.6f}"
        )

        if probe_select_stage2 and probe_every > 0 and (ep % probe_every == 0 or ep == epochs):
            probe_acc = eval_probe_preserve_state(model, train_loader, test_loader, device, probe_cfg)
            log += f" | probe_acc/test={probe_acc:.4f}"
            if probe_acc > best_probe_acc:
                best_probe_acc = float(probe_acc)
                best_probe_epoch = int(ep)
                best_probe_recon = float(te)
                save_checkpoint(
                    best_stage2_save_path,
                    model,
                    model.cfg,
                    args,
                    stage="stage2-best-probe",
                    extra={
                        "best_probe_acc": best_probe_acc,
                        "best_probe_epoch": best_probe_epoch,
                        "recon_eval_test_at_save": best_probe_recon,
                    },
                )
                log += " | best_probe=1"

        print(log)

    if probe_select_stage2:
        if best_probe_epoch >= 0:
            print(
                f"[stage2-best] epoch={best_probe_epoch:03d} | "
                f"probe_acc/test={best_probe_acc:.4f} | recon_eval/test={best_probe_recon:.6f}"
            )
        else:
            print("[stage2-best] no probe checkpoint was evaluated/saved")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"], help="dataset source")
    ap.add_argument("--seed", type=int, default=111)

    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])

    ap.add_argument("--run-mode", type=str, default="both", choices=["both", "stage1-only", "stage2-only"])
    ap.add_argument("--checkpoint-dir", type=str, default="./_checkpoints")
    ap.add_argument("--stage1-save", type=str, default="")
    ap.add_argument("--stage2-save", type=str, default="")
    ap.add_argument("--stage2-best-save", type=str, default="")
    ap.add_argument("--load-checkpoint", type=str, default="")
    ap.add_argument("--load-strict", type=int, default=1)

    ap.add_argument("--stage1-epochs", type=int, default=20)
    ap.add_argument("--stage1-lr", type=float, default=4e-4)
    ap.add_argument("--stage1-wd", type=float, default=1e-4)

    ap.add_argument("--stage2-epochs", type=int, default=20)
    ap.add_argument("--stage2-lr", type=float, default=4e-4)
    ap.add_argument("--stage2-wd", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder-stage2", type=int, default=0)
    ap.add_argument("--freeze-decoder-stage2", type=int, default=1)
    ap.add_argument("--recon-weight-stage2", type=float, default=0.0)

    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--gate-floor", type=float, default=0.25)
    ap.add_argument("--gate-bias", type=int, default=1)
    ap.add_argument("--bound-delta", type=int, default=0)
    ap.add_argument("--pred-hidden", type=int, default=1)

    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--pred-loss", type=str, default="weighted_mse", choices=["mse", "smooth_l1", "weighted_mse"])
    ap.add_argument("--pred-weight-power", type=float, default=0.5)
    ap.add_argument("--pred-weight-clip", type=float, default=10.0)
    ap.add_argument("--pred-cos-weight", type=float, default=0.4)
    ap.add_argument("--gate-binarize-weight", type=float, default=0.0)
    ap.add_argument("--gate-l1-weight", type=float, default=0.0)
    ap.add_argument("--delta-l2-weight", type=float, default=1e-5)
    ap.add_argument("--gate-branch-l2-weight", type=float, default=0.0)

    ap.add_argument("--corrupt-mode", type=str, default="mask", choices=["mask", "noise"])
    ap.add_argument("--corrupt-max-frac", type=float, default=0.25)
    ap.add_argument("--corrupt-noise-std", type=float, default=0.2)

    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--latent-anchor-weight", type=float, default=0.1)
    ap.add_argument("--latent-pred-match-weight", type=float, default=0.0)
    ap.add_argument("--pred-mag-weight", type=float, default=0.0)

    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)
    ap.add_argument("--probe-print-every-epoch", type=int, default=0)
    ap.add_argument("--probe-select-stage2", type=int, default=1)
    ap.add_argument("--probe-every", type=int, default=1)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Two-Stage Split-Update Trainable-Gate AE with EMA Target Encoder (probe-selected checkpoints)")
    print("device:", device)

    data_dir = "./_data"
    train_loader, test_loader = get_mnist_loaders(data_dir, args.batch_size, dataset=args.dataset, seed=args.seed)

    cfg = ConvStackCfg(
        in_channels=1,
        channels=tuple(int(x) for x in args.channels.split(",") if x.strip()),
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        nonlinearity=args.nonlinearity,
    )
    print("config:", cfg)

    pred_hidden = args.pred_hidden if args.pred_hidden > 0 else None

    model = TwoStageSplitUpdateEMATargetGateAE(
        cfg,
        pred_hidden=pred_hidden,
        delta_scale=args.delta_scale,
        gate_floor=args.gate_floor,
        gate_bias=bool(args.gate_bias),
        bound_delta=bool(args.bound_delta),
    )

    if args.load_checkpoint:
        load_checkpoint(args.load_checkpoint, model, device, strict=bool(args.load_strict))

    default_stage1_ckpt = os.path.join(args.checkpoint_dir, "trainable_gate_ae_split_update_ema_target_stage1.pt")
    default_stage2_ckpt = os.path.join(args.checkpoint_dir, "trainable_gate_ae_split_update_ema_target_stage2.pt")
    default_stage2_best_ckpt = os.path.join(args.checkpoint_dir, "trainable_gate_ae_split_update_ema_target_stage2_best_probe.pt")
    stage1_save_path = args.stage1_save or default_stage1_ckpt
    stage2_save_path = args.stage2_save or default_stage2_ckpt
    stage2_best_save_path = args.stage2_best_save or default_stage2_best_ckpt

    if args.run_mode in ("both", "stage1-only") and args.stage1_epochs > 0:
        train_stage1_ae(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            wd=args.stage1_wd,
        )
        save_checkpoint(stage1_save_path, model, cfg, args, stage="stage1")

    probe_cfg = ClassifierEvalCfg(
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        wd=args.probe_wd,
        print_every_epoch=bool(args.probe_print_every_epoch),
    )

    if args.run_mode == "stage1-only":
        acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
        print(f"last linear probe acc: {acc:.4f}")
        return

    if args.run_mode == "stage2-only" and not args.load_checkpoint:
        raise ValueError("stage2-only mode requires --load-checkpoint pointing to a pretrained stage1 checkpoint")

    if args.run_mode in ("both", "stage2-only") and args.stage2_epochs > 0:
        train_stage2_predictor(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            wd=args.stage2_wd,
            corrupt_mode=args.corrupt_mode,
            corrupt_max_frac=args.corrupt_max_frac,
            corrupt_noise_std=args.corrupt_noise_std,
            freeze_encoder=bool(args.freeze_encoder_stage2),
            freeze_decoder=bool(args.freeze_decoder_stage2),
            pred_weight=args.pred_weight,
            pred_loss=args.pred_loss,
            pred_weight_power=args.pred_weight_power,
            pred_weight_clip=args.pred_weight_clip,
            pred_cos_weight=args.pred_cos_weight,
            gate_binarize_weight=args.gate_binarize_weight,
            gate_l1_weight=args.gate_l1_weight,
            delta_l2_weight=args.delta_l2_weight,
            gate_branch_l2_weight=args.gate_branch_l2_weight,
            recon_weight=args.recon_weight_stage2,
            latent_anchor_weight=args.latent_anchor_weight,
            latent_pred_match_weight=args.latent_pred_match_weight,
            pred_mag_weight=args.pred_mag_weight,
            ema_decay=args.ema_decay,
            probe_select_stage2=bool(args.probe_select_stage2),
            probe_every=max(1, int(args.probe_every)),
            probe_cfg=probe_cfg,
            best_stage2_save_path=stage2_best_save_path,
            args=args,
        )
        save_checkpoint(stage2_save_path, model, cfg, args, stage="stage2-final")

    acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"last linear probe acc: {acc:.4f}")


if __name__ == "__main__":
    main()
