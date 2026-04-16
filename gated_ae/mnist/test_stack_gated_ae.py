"""Stacked Gated Autoencoders (layerwise latent reconstruction) + top JEPA-like loss.

As requested:
- Stack of L one-layer conv autoencoders.
- Each level l:
    z_full_l = enc_l(x_{l-1})                    (pre-gate latent)
    z_gated_l = TopKGate_l(z_full_l)             (post-gate latent)
    xhat_{l-1} = dec_l(z_full_l)                 (reconstruct input of this level)
    x_l = z_gated_l                              (input to next level)

- Loss:
    * level 1 (pixels): BCE(xhat0, x0)
    * levels 2..L: MSE(xhat_{l-1}, x_{l-1}) where x_{l-1} is the *gated* representation from previous level
    * top level: JEPA-like prediction in latent space (post-gate) with stop-grad target:
        pred(z_top(clean)) ~ stopgrad(z_top(corrupt))

- Hard top-k gates with STE gradients on every level.
- Linear probe evaluation on TOP-level PRE-GATE features (z_full_L).

This file is self-contained (does not modify common_conv.py).

Usage:
  python3 test_stack_gated_ae.py --epochs 20 --levels 3 --topk 16

Requires: torch, torchvision.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common_mnist import ClassifierEvalCfg, corrupt_batch, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402
from common_conv import UnitKernelConv2d  # noqa: E402


# ------------------------- basic blocks -------------------------


class TopKGate(nn.Module):
    """Channel-wise hard top-k gate with STE gradients."""

    def __init__(self, channels: int, k: int, temperature: float = 1.0, init: float = 0.0):
        super().__init__()
        assert 1 <= k <= channels
        self.channels = channels
        self.k = k
        self.temperature = float(temperature)
        self.logits = nn.Parameter(torch.full((channels,), float(init)))

    def mask(self, device=None, dtype=None) -> torch.Tensor:
        logits = self.logits
        if device is not None:
            logits = logits.to(device)
        if dtype is not None:
            logits = logits.to(dtype)

        idx = torch.topk(logits, self.k).indices
        hard = torch.zeros_like(logits)
        hard[idx] = 1.0
        soft = torch.sigmoid(logits / self.temperature)
        return (hard - soft).detach() + soft

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.mask(device=z.device, dtype=z.dtype)
        return z * m.view(1, -1, 1, 1)


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(name)


class ConvAELevel(nn.Module):
    """One level: conv encoder (unit-norm) + top-k gate + convtranspose decoder."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        nonlinearity: str,
        topk: int,
        gate_temperature: float,
        gate_init: float,
    ):
        super().__init__()
        self.enc = nn.Sequential(
            UnitKernelConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _act(nonlinearity),
        )
        self.gate = TopKGate(out_ch, k=topk, temperature=gate_temperature, init=gate_init)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(out_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _act(nonlinearity) if in_ch != 1 else nn.Sigmoid(),
        )

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def encode_gated(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.encode_full(x))

    def decode(self, z_full: torch.Tensor) -> torch.Tensor:
        return self.dec(z_full)


class ConvPredictor(nn.Module):
    """1x1 conv MLP predictor."""

    def __init__(self, channels: int, hidden: int | None = None, nonlinearity: str = "relu"):
        super().__init__()
        if hidden is None:
            hidden = max(32, channels)
        act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[nonlinearity if nonlinearity in ("relu", "gelu") else "relu"]
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            act,
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ------------------------- full model -------------------------


@dataclass
class StackCfg:
    in_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64)
    topk: tuple[int, ...] | None = None  # per-level top-k (len == len(channels)); None => default c_out//2
    kernel_size: int = 6
    stride: int = 2
    padding: int = 2
    nonlinearity: str = "relu"


class StackedGatedAE(nn.Module):
    def __init__(
        self,
        cfg: StackCfg,
        gate_temperature: float = 1.0,
        gate_init: float = 0.0,
        pred_hidden: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        # per-level topk
        if cfg.topk is None:
            topk_list = [max(1, c // 2) for c in cfg.channels]
        else:
            if len(cfg.topk) != len(cfg.channels):
                raise ValueError(f"cfg.topk must have same length as cfg.channels ({len(cfg.channels)}), got {len(cfg.topk)}")
            topk_list = list(cfg.topk)

        self.levels = nn.ModuleList()
        c_in = cfg.in_channels
        for li, c_out in enumerate(cfg.channels):
            k = int(topk_list[li])
            if not (1 <= k <= c_out):
                raise ValueError(f"Bad topk at level {li}: topk={k}, c_out={c_out}")

            self.levels.append(
                ConvAELevel(
                    in_ch=c_in,
                    out_ch=c_out,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    nonlinearity=cfg.nonlinearity,
                    topk=k,
                    gate_temperature=gate_temperature,
                    gate_init=gate_init,
                )
            )
            c_in = c_out

        self.predictor = ConvPredictor(cfg.channels[-1], hidden=pred_hidden, nonlinearity=cfg.nonlinearity)

    def forward_levels(self, x0: torch.Tensor):
        """Forward through levels, returning per-level (x_in, z_full, z_gated, x_hat)."""
        xs = []
        x = x0
        for lvl in self.levels:
            z_full = lvl.encode_full(x)
            z_gated = lvl.gate(z_full)
            x_hat = lvl.decode(z_full)
            xs.append((x, z_full, z_gated, x_hat))
            x = z_gated
        return xs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # linear probe on top pre-gate features
        x_in = x
        z_full = None
        for lvl in self.levels:
            z_full = lvl.encode_full(x_in)
            z_gated = lvl.gate(z_full)
            x_in = z_gated
        return z_full


# ------------------------- training -------------------------


def train(
    model: StackedGatedAE,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    pred_weight: float,
    corrupt_mode: str,
    corrupt_max_frac: float,
    corrupt_noise_std: float,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def eval_pix_recon(loader) -> float:
        model.eval()
        loss_sum = 0.0
        n = 0
        for x0, _y in loader:
            x0 = x0.to(device)
            outs = model.forward_levels(x0)
            xhat0 = outs[0][3]
            loss_sum += float(F.binary_cross_entropy(xhat0, x0, reduction="sum").item())
            n += int(x0.numel())
        return loss_sum / max(1, n)

    for ep in range(1, epochs + 1):
        model.train()
        rec_pix_avg = 0.0
        rec_mid_avg = 0.0
        pred_avg = 0.0
        nb = 0

        for x0, _y in train_loader:
            x0 = x0.to(device)

            # Forward clean
            outs = model.forward_levels(x0)

            # (1) pixel recon loss: level 0 reconstructs pixels
            xhat0 = outs[0][3]
            loss_pix = F.binary_cross_entropy(xhat0, x0)

            # (2) intermediate recon losses: for l>=2 reconstruct x_{l-1} (which is gated output of prev level)
            # intermediate recon losses: may be empty if the stack has only 1 level
            loss_mid = torch.zeros((), device=x0.device)
            for li in range(1, len(outs)):
                x_in_li = outs[li][0]   # this is x_{li} = z_gated_{li-1}
                xhat_li = outs[li][3]   # recon of that input from z_full_li
                loss_mid = loss_mid + F.mse_loss(xhat_li, x_in_li)

            # (3) top JEPA-like pred loss on post-gate top latent
            x_cor = corrupt_batch(x0, mode=corrupt_mode, max_frac=corrupt_max_frac, noise_std=corrupt_noise_std)
            outs_cor = model.forward_levels(x_cor)

            z_top = outs[-1][2]      # z_gated_L (clean)
            z_tgt = outs_cor[-1][2]  # z_gated_L (corrupt)
            z_hat = model.predictor(z_top)
            loss_pred = F.mse_loss(z_hat, z_tgt.detach())

            loss = loss_pix + loss_mid + pred_weight * loss_pred

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            rec_pix_avg += float(loss_pix.item())
            rec_mid_avg += float(loss_mid.item())
            pred_avg += float(loss_pred.item())
            nb += 1

        te = eval_pix_recon(test_loader)
        print(
            f"epoch {ep:03d} | pix(BCE)={rec_pix_avg/max(1,nb):.6f} | mid(MSE)={rec_mid_avg/max(1,nb):.6f} | pred(MSE)={pred_avg/max(1,nb):.6f} | pix_test={te:.6f}"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"])
    ap.add_argument("--seed", type=int, default=111)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--channels", type=str, default="16,32,64,128,256")
    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])

    ap.add_argument("--topk", type=str, default="", help="per-level topk, e.g. '8,16,32'. Empty => default c_out//2")
    ap.add_argument("--gate-temperature", type=float, default=1.0)
    ap.add_argument("--gate-init", type=float, default=0.0)

    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--pred-hidden", type=int, default=16)

    ap.add_argument("--corrupt-mode", type=str, default="mask", choices=["mask", "noise"])
    ap.add_argument("--corrupt-max-frac", type=float, default=0.25)
    ap.add_argument("--corrupt-noise-std", type=float, default=0.2)

    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_dir = os.path.join(HERE, "_data")
    train_loader, test_loader = get_mnist_loaders(data_dir, args.batch_size, dataset=args.dataset, seed=args.seed)

    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())

    topk = None
    if args.topk.strip():
        topk = tuple(int(x) for x in args.topk.split(",") if x.strip())

    cfg = StackCfg(
        in_channels=1,
        channels=channels,
        topk=topk,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        nonlinearity=args.nonlinearity,
    )

    pred_hidden = args.pred_hidden if args.pred_hidden > 0 else None
    model = StackedGatedAE(
        cfg,
        gate_temperature=args.gate_temperature,
        gate_init=args.gate_init,
        pred_hidden=pred_hidden,
    )

    train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        pred_weight=args.pred_weight,
        corrupt_mode=args.corrupt_mode,
        corrupt_max_frac=args.corrupt_max_frac,
        corrupt_noise_std=args.corrupt_noise_std,
    )

    probe_cfg = ClassifierEvalCfg(epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, print_every_epoch=True)
    acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"final linear probe acc (top pre-gate): {acc:.4f}")


if __name__ == "__main__":
    main()
