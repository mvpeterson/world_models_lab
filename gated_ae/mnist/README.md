# Gated AEs vs JEPA on (A-)MNIST

This folder contains a small, self-contained set of experiments around **representation learning for world models**: how to learn features that capture *predictable structure* (useful for forecasting/control), rather than merely optimizing for reconstruction or classification.

The code here focuses on a clean MNIST-like setting (we use **A-MNIST** by default) to compare a few minimal baselines under a unified evaluation protocol (a linear classifier on frozen features).

This repository extends the methodological discussion in our paper for the case of non-linear multilayer models:
- **Is the reconstruction loss culprit? An attempt to outperform JEPA**: https://arxiv.org/abs/2603.14131

## What is being compared

All scripts share common utilities (`common_mnist.py`, `common_conv.py`) and differ only in the training objective / architecture.

1) **Supervised convnet (upper bound for linear probe)**
   - [`test_plain_conv.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_plain_conv.py)
   - End-to-end supervised training of a conv feature extractor + linear head.

2) **Vanilla autoencoder**
   - [`test_vanilla_ae.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_vanilla_ae.py)
   - Pretrain a standard convolutional AE with reconstruction loss, then evaluate features with a **frozen** linear probe.

3) **JEPA (student/teacher + predictor)**
   - [`test_jepa.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_jepa.py)
   - Pretrain a JEPA model by predicting teacher latents of a corrupted view from student latents, with EMA teacher updates.
   - No pixel-space reconstruction.

4) **Top-k gated predictive autoencoder (gated AE)**
   - [`test_topgate_pred_ae.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_topgate_pred_ae.py)
   - A standard AE (reconstruction uses the *full* latent), augmented with a trainable **hard top‑k gate** at the top latent level.
   - The gate selects a subset of latent channels that are trained to be predictable via an additional latent prediction loss.

5) **Stack of gated predictive autoencoders**
   - [`test_stack_gated_ae.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_stack_gated_ae.py)
   - Gates are added to each AE layer. The loss is the sum of losses at each layer.
   - [`test_stack_gated_pretrain.py`](https://github.com/Necr0x0Der/world_models_lab/blob/main/gated_ae/mnist/test_stack_gated_pretrain.py)
   - The stack is pretrained after adding each layer.

## Summary of observed results (5 conv layers)

Using a 5-layer conv backbone, the best linear-probe scores we observed are:

- (1) **Supervised convnet**: **0.993**
- (2) **Vanilla AE**: **0.9857**
- (3) **JEPA**: **0.983**
- (4) **Gated AE**: **0.9877**

A qualitative observation: for a **2-layer** network, JEPA training tends to be more stable and reaches better scores than in the 5-layer network; in that shallow regime, both JEPA and gated AEs consistently outperform vanilla AEs. However, top scores are obtained by Gated AE with a deeper network (which still underperforms w.r.t. the supervised convnet). We tested the very basic JEPA and Gated AE architectures for clearner comparison of their core ideas (no pixel-level loss vs explicit predictable features selection), and there should be a lot of room for improvement (which should be tested on more complex benchmarks).

## How to run

Each script accepts `--dataset` (`mnist` or `a-mnist`) and `--seed` for reproducibility.

Examples:

```bash
# Supervised baseline
python3 test_plain_conv.py --dataset a-mnist --seed 0

# Vanilla AE pretrain + linear probe
python3 test_vanilla_ae.py --dataset a-mnist --seed 0

# JEPA pretrain + linear probe
python3 test_jepa.py --dataset a-mnist --seed 0

# Gated predictive AE pretrain + linear probe
python3 test_topgate_pred_ae.py --dataset a-mnist --seed 0
```

The scripts can be run with the default settings without specifying any parameters.

## Notes

- The goal is not to “win MNIST”, but to keep the setting simple enough that we can reason about *why* certain objectives learn better representations.
- In this framing, **predictability** (implemented via latent-space prediction losses) is treated as a practical proxy for “signal”, while reconstruction can encourage encoding nuisance variability unless paired with appropriate inductive biases.
