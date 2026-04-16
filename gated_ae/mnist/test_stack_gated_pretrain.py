"""Progressive (layer-by-layer) pretraining wrapper for test_stack_gated_ae.py.

Requested behavior:
- Do not change test_stack_gated_ae.py.
- Reuse its StackedGatedAE + train() exactly as implemented.
- Pretrain incrementally by increasing the number of levels (channels).

Example: if --channels 16,32,64 then do stages:
  1) train model with channels=(16,)
  2) init encoder/decoder/gates from stage 1, train model with channels=(16,32)
  3) init from stage 2, train model with channels=(16,32,64)

We carry over all matching parameters (levels already present, predictor, etc.)
New level parameters are randomly initialized.

Usage:
  python3 test_stack_gated_pretrain.py --channels 16,32,64 --epochs 20

Requires: torch, torchvision.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common_mnist import ClassifierEvalCfg, eval_classifier, get_mnist_loaders, set_seed  # noqa: E402

# Reuse model + train() exactly as implemented
from test_stack_gated_ae import StackedGatedAE, StackCfg, train as train_once  # noqa: E402


def _load_matching(new_model: StackedGatedAE, prev_model: StackedGatedAE) -> None:
    """Best-effort partial load: copy all tensors with matching names+shapes."""
    prev_sd = prev_model.state_dict()
    new_sd = new_model.state_dict()

    filtered = {}
    for k, v in prev_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            filtered[k] = v

    new_model.load_state_dict(filtered, strict=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="a-mnist", choices=["mnist", "a-mnist"])
    ap.add_argument("--seed", type=int, default=111)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    #ap.add_argument("--channels", type=str, default="32,64,128,256,256")
    ap.add_argument("--channels", type=str, default="64,128,256")
    ap.add_argument("--topk", type=str, default="", help="per-level topk, e.g. '8,16,32'. Empty => default c_out//2")

    ap.add_argument("--kernel-size", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "gelu", "sigmoid", "tanh", "identity"])

    ap.add_argument("--gate-temperature", type=float, default=1.0)
    ap.add_argument("--gate-init", type=float, default=0.0)

    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--pred-hidden", type=int, default=0)

    ap.add_argument("--corrupt-mode", type=str, default="mask", choices=["mask", "noise"])
    ap.add_argument("--corrupt-max-frac", type=float, default=0.25)
    ap.add_argument("--corrupt-noise-std", type=float, default=0.2)

    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-lr", type=float, default=4e-4)
    ap.add_argument("--probe-wd", type=float, default=1e-4)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_dir = os.path.join(HERE, "_data")
    train_loader, test_loader = get_mnist_loaders(data_dir, args.batch_size, dataset=args.dataset, seed=args.seed)

    full_channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    if len(full_channels) < 1:
        raise ValueError("--channels must be non-empty")

    full_topk = None
    if args.topk.strip():
        full_topk = tuple(int(x) for x in args.topk.split(",") if x.strip())
        if len(full_topk) != len(full_channels):
            raise ValueError("--topk must match length of --channels")

    prev_model = None

    for stage in range(1, len(full_channels) + 1):
        ch = full_channels[:stage]
        tk = (full_topk[:stage] if full_topk is not None else None)

        print("\n" + "=" * 80)
        print(f"Stage {stage}/{len(full_channels)}: channels={ch}, topk={tk if tk is not None else 'default(c//2)'}")

        cfg = StackCfg(
            in_channels=1,
            channels=ch,
            topk=tk,
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

        if prev_model is not None:
            _load_matching(model, prev_model)

        train_once(
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

        prev_model = deepcopy(model).cpu()

    # final linear probe
    probe_cfg = ClassifierEvalCfg(epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, print_every_epoch=True)
    acc = eval_classifier(model, train_loader, test_loader, device, cfg=probe_cfg, frozen=True)
    print(f"final linear probe acc (top pre-gate): {acc:.4f}")


if __name__ == "__main__":
    main()
