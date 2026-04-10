# Experiment Note: From Collapsing Gated Residuals to a Stable Two-Stage Split-Update Model

This is AI generated experiment note

## Core hypothesis

Corruption effects in latent space can be modeled as a selective residual edit, where a learned gate determines which proposed latent corrections should actually apply.

We study corruption-induced latent prediction through a selective residual update mechanism, where a predictor proposes latent edits and a learned gate controls which edits are expressed.

## Goal
We wanted a latent predictive model where:

- `z0` = clean latent
- predictor proposes latent updates
- gate decides which updates are real
- final corrupted-latent prediction is a **residual selective update**

Core intended semantics:

```text
z_cor_pred = z0 + gated_update
```

The gate should act on the update itself, not just on the loss.

---

# 1. Initial idea: direct gated residual update

## Formulation
The first residual variant used:

```text
delta_pred = q * g * dz
z_cor_pred = z0 + delta_pred
```

where:
- `g = sigmoid(gate(z0))`
- `dz = predictor(z0)`
- `q = delta_scale`

Training target:
```text
delta_tgt = z_cor - z0
loss_pred = MSE(delta_pred, delta_tgt)
```

Reconstruction stayed separate:
```text
loss_recon = BCE(decoder(z0), x)
```

## What happened
This formulation consistently collapsed.

### Symptoms
- `dz(abs)` went to ~0
- `delta(abs)` went to ~0
- gate either stayed near ~0.5 or saturated uselessly
- predictive loss remained nontrivial, but branch effectively died

### Diagnosis
The branch had a bad multiplicative choke point:

```text
delta_pred = q * g * dz
```

So:
- if `g` shrinks → dead
- if `dz` shrinks → dead
- if both are modest but small → still dead

Dense MSE on latent delta made near-zero residuals a cheap local solution.

---

# 2. Tried improving the direct gated residual

We tested several fixes on the original residual path.

## 2.1 Gate statistics and diagnostics
Added logging for:
- `gate(mean/std/min/max)`
- `dz(abs)`
- `delta(abs)`
- `delta_tgt(abs)`
- `delta(reg)`

This clarified that the main failure was usually:
- **predictor collapse first**
- gate becoming irrelevant afterward

## 2.2 Gate binarization pressure
Added:
```text
loss_gate_binarize = mean(g * (1 - g))
```

to encourage more decisive 0/1 behavior.

### Result
Bad.
- gate became binary fast
- predictor still collapsed
- branch still dead

Conclusion:
- gate indecision was not the root cause
- making the gate sharper did not rescue the predictor

## 2.3 Gate floor
Changed forward modulation to:

```text
g_eff = gate_floor + (1 - gate_floor) * g
delta_pred = q * g_eff * dz
```

to prevent total shutdown.

### Result
Still bad.
- predictor kept collapsing
- gate no longer fully shut branch off, but that was not enough

Conclusion:
- the problem was deeper than gate closure

---

# 3. Introduced two-stage training

## Motivation
The latent target
```text
delta_tgt = z_cor - z0
```
was moving too much during joint learning because the encoder was changing.

So we split training into:

## Stage 1
Pretrain plain autoencoder:
```text
z0 = encoder(x)
x_hat = decoder(z0)
loss = BCE(x_hat, x)
```

## Stage 2
Freeze encoder/decoder and train the predictive branch in a fixed latent space.

### Result
This stabilized the target latent space, but **direct gated residual still collapsed**.

Important insight:
- joint optimization instability was real
- but not the whole problem

So the target space got cleaner, yet the branch still preferred near-zero updates.

---

# 4. Shifted to a split-update architecture

## Motivation
The fully gated residual path was too fragile.

So we changed the update to:

```text
delta_pred = delta_scale * (d_base + g_eff * d_gate)
```

where:
- `d_base` = always-on residual update head
- `d_gate` = selectively gated residual update head
- `g_eff = gate_floor + (1 - gate_floor) * g`

This preserved the intended semantics:
- gate acts on the update itself
- but predictor still has a non-gated backbone path

## Result
This was clearly better than `g * dz`.

### Early behavior
- branch started alive
- `d_base` and `d_gate` were nonzero
- gate began to matter

### But
With plain MSE, the model still drifted toward poor solutions:
- either collapse
- or weak target alignment

Conclusion:
- architecture improved
- loss was now the limiting factor

---

# 5. Changed the prediction loss: weighted MSE

## Motivation
The corruption-induced latent delta is small and sparse-ish in the pretrained latent space.

Observed:
```text
delta_tgt(abs) ~ 5e-3
```

Dense unweighted MSE makes “predict almost zero everywhere” too attractive.

## New idea
Weight the delta error by target magnitude.

### First version
```text
pred_w = abs(delta_tgt)
pred_w = pred_w / mean(pred_w)
loss_pred = mean(pred_w * (delta_pred - delta_tgt)^2)
```

### Result
This broke collapse:
- predictor woke up
- gate became active/selective
- prediction loss dropped

But a new failure appeared:
- updates became much too large
- `pred_w(max)` became huge (~200+)
- `delta(abs)` vastly overshot `delta_tgt(abs)`

Conclusion:
- weighting by target magnitude was the right direction
- but raw weighting was too sharp

---

# 6. Softened and clipped the weighted loss

## Modification
Changed weighted MSE to:

```text
pred_w = (abs(delta_tgt) + eps) ^ pred_weight_power
pred_w = pred_w / mean(pred_w)
pred_w = clamp(pred_w, max=pred_weight_clip)
loss_pred = mean(pred_w * (delta_pred - delta_tgt)^2)
```

with defaults:
- `pred_weight_power = 0.5`
- `pred_weight_clip = 10`

So effectively:
- sqrt weighting
- clipped max emphasis

## Result
This was a major stabilization.

### Improvements
- predictor stayed alive
- gate stayed active
- overshoot reduced substantially
- `delta(ratio)` fell from catastrophic values to a workable range

At this point the model was no longer collapsing or exploding blindly.

---

# 7. Added directional diagnostics

To understand whether the model was merely active or actually aligned with the target, we added:

- `delta(cos)` = cosine similarity between `delta_pred` and `delta_tgt`
- `delta(std)` / `delta_tgt(std)`
- `delta(ratio) = delta(abs) / delta_tgt(abs)`

## Findings
Even in stabilized runs:
- amplitude became reasonable
- gate became selective
- but cosine alignment remained modest

So the next issue was no longer collapse or scale — it was **directional mismatch**.

---

# 8. Added cosine alignment loss

## Modification
Added an extra directional term:

```text
loss_cos = 1 - cosine_similarity(delta_pred, delta_tgt)
```

Total stage-2 loss became:

```text
loss =
    recon_weight * loss_recon
    + pred_weight * loss_pred
    + pred_cos_weight * loss_cos
    + other regularizers
```

with `pred_cos_weight` initially tested at `0.1`, then `0.3`.

## Result
This gave the first clearly promising regime.

### With `pred_cos_weight = 0.1`
- amplitude calibration improved a lot
- gate remained selective
- cosine alignment improved somewhat

### With `pred_cos_weight = 0.3`
Best observed tradeoff so far:
- `delta(ratio)` near ~1.1 by later epochs
- `delta(cos)` around ~0.27
- gate strongly selective
- predictor alive
- no collapse
- no runaway overshoot

Conclusion:
- directional supervision was the missing piece after stabilization
- stronger cosine pressure improved calibration without killing learning

---

# Current best model

## Architecture
Two-stage split-update gated residual predictor:

### Stage 1
Pretrain AE:
```text
z0 = encoder(x)
x_hat = decoder(z0)
loss_recon = BCE(x_hat, x)
```

### Stage 2
Freeze encoder/decoder initially and train:

```text
g = sigmoid(gate(z0))
g_eff = gate_floor + (1 - gate_floor) * g
d_base = predictor_base(z0)
d_gate = predictor_gate(z0)

delta_pred = delta_scale * (d_base + g_eff * d_gate)
delta_tgt = z_cor - z0
```

## Loss
Weighted delta regression + cosine alignment:

```text
loss_pred = weighted_mse(delta_pred, delta_tgt)
loss_cos = 1 - cosine(delta_pred, delta_tgt)

loss_total =
    pred_weight * loss_pred
    + pred_cos_weight * loss_cos
    + optional regularizers
```

## Best observed hyperparameter region so far
Promising setup:
```text
delta_scale = 0.12
delta_l2_weight = 1e-5
pred_cos_weight = 0.3
pred_loss = weighted_mse
pred_weight_power = 0.5
pred_weight_clip = 10
freeze_encoder = 1
freeze_decoder = 1
```

---

# Main lessons learned

## 1. Gate-on-update is viable, but naive `g * dz` is too fragile
The original direct multiplicative residual collapsed too easily.

## 2. Pretraining helps, but does not solve everything
It stabilizes the latent target, but the objective still matters.

## 3. Split-update was the key architectural improvement
The always-on base update path prevented total branch death.

## 4. Loss design mattered as much as architecture
Dense plain MSE encouraged trivial near-zero residuals.
Target-aware weighting revived the branch.

## 5. Weighting needed softening
Raw target-magnitude weighting overemphasized rare components and caused overshoot.
Power-law + clipping fixed that.

## 6. Directional supervision mattered
Cosine loss materially improved calibration and target alignment.

---

# Current interpretation of the learned mechanism
The model now behaves roughly like this:

- `d_base` learns a broad corruption-response prior
- `d_gate` learns richer candidate update structure
- `g` selects/suppresses selective refinements
- weighted loss focuses learning on latent components that actually move
- cosine loss keeps the update pointed in the right direction

This is finally close to the original intended story:
- predictor proposes adjustments
- gate decides which ones are real
- final latent transition is residual and selective

---

# Open questions / next steps

## 1. Run stage 2 to completion in the current best regime
Need to see whether:
- `delta(ratio)` stabilizes near 1
- `delta(cos)` continues climbing
- gate remains selective

## 2. Possibly fine-tune encoder after stage-2 warmup
So far stage 2 has been evaluated mostly with frozen encoder/decoder.
Next clean step:
- warm up stage 2 in fixed latent
- then unfreeze encoder for co-adaptation

## 3. Evaluate whether exact delta target is fundamentally ambiguous
Even the current best runs have only moderate cosine alignment.
This may reflect a real limit:
- predicting exact corruption-induced latent delta from clean latent alone may be partially underdetermined

---

# Bottom-line summary
The project evolved like this:

1. **Direct gated residual (`g * dz`)**
   - collapsed

2. **Gate tweaks (binarization, floor)**
   - did not fix collapse

3. **Two-stage pretraining**
   - stabilized latent space, but collapse remained

4. **Split-update architecture (`d_base + g * d_gate`)**
   - predictor stayed alive

5. **Weighted delta loss**
   - broke collapse, but initially caused overshoot

6. **Soft/clipped weighting**
   - stabilized magnitude

7. **Cosine alignment loss**
   - improved calibration and directional match

## Current status
The best current formulation is:
- **two-stage**
- **split-update**
- **soft weighted delta loss**
- **cosine alignment loss**

And it is the first version that looks genuinely workable rather than pathological.

Next real question:

    What happens if you unfreeze the encoder after this stable warmup?

Because the whole representation-learning angle only becomes interesting once the encoder is allowed to adapt.

Can the encoder improve if we let it adapt after the residual/gate dynamics are already well-formed?

Answer is negative:

    When the same encoder is used both to define the corruption-induced latent target and to receive gradients from the predictive objective, stage-2 fine-tuning becomes unstable even at very low learning rates. Freezing the latent space is therefore important for stable residual-gated prediction under the current formulation.


Design the EMA/frozen-target encoder version as the next clean script


## Experiment Note: EMA/Frozen-Target Encoder Stage-2 Training

### Goal
After finding that direct encoder fine-tuning destabilized residual-gated latent prediction, we tested a cleaner alternative:

- keep the **split-update gated residual predictor**
- but define the corruption target in a **separate slowly moving target latent space**
- allow the online encoder to adapt against that target

The motivating idea was:

> maybe stage-2 encoder adaptation only fails because the same encoder both defines the target and chases it.

So the next question became:

> can we stabilize stage-2 learning by replacing the self-referential latent target with an EMA target encoder?

---

# 1. Motivation for the EMA target encoder

The previous conclusion was negative for naive encoder fine-tuning:

- if the same encoder defines
  ```text
  delta_tgt = z_cor - z0
  ```
  and also receives predictive gradients,
- then the target itself moves with the learner,
- and stage-2 training becomes unstable even at low LR.

So we introduced a **target encoder**:

- online encoder = trainable
- target encoder = no gradient
- target encoder updated by EMA from online encoder

This turns stage 2 into a teacher/student-style latent prediction setup.

---

# 2. EMA-target formulation

## Architecture
We kept the same split-update residual predictor:

```text
g = sigmoid(gate(z0_online))
g_eff = gate_floor + (1 - gate_floor) * g

d_base = predictor_base(z0_online)
d_gate = predictor_gate(z0_online)

delta_pred = delta_scale * (d_base + g_eff * d_gate)
z_cor_pred = z0_online + delta_pred
```

## Target definition
Instead of using the online encoder to define the target, we used a target encoder:

```text
z0_tgt = target_encoder(x)
z_cor_tgt = target_encoder(x_cor)
delta_tgt = z_cor_tgt - z0_tgt
```

with:
- no direct optimization of `target_encoder`
- EMA update after each step:
  ```text
  theta_tgt <- ema_decay * theta_tgt + (1 - ema_decay) * theta_online
  ```

---

# 3. First EMA-target attempt: still unstable

## Initial hope
We expected the EMA target encoder to solve the moving-target problem and make stage-2 encoder adaptation viable.

## What happened
It did **not** work by itself.

### Symptoms
In the first EMA runs:
- reconstruction exploded immediately
- predictor magnitude ran away
- `delta(ratio)` climbed well above 2
- branch norms kept growing
- gate remained selective, but that was not the bottleneck

Typical behavior:
- `recon_eval/test` jumped from stage-1 levels (~0.046) to very large values
- `delta(abs)` overshot `delta_tgt(abs)` severely
- `d_base(abs)` and `d_gate(abs)` grew rapidly

## Interpretation
EMA target removed one failure mode, but not the deeper one.

The online encoder still found a loophole:
- drift into a new representation
- let the predictor grow aggressively
- abandon decoder-compatible latents

Important lesson:

> **EMA target alone is not enough.**

A frozen decoder without stage-2 reconstruction pressure does not actually constrain the encoder strongly enough.

---

# 4. Added the missing stabilizers

To make the EMA-target setup genuinely constrained, we added several components together.

## 4.1 Frozen decoder + nonzero stage-2 reconstruction
We kept the decoder frozen **and** turned on a stage-2 reconstruction term:

```text
loss_recon = BCE(decoder(z0_online), x)
```

This mattered because without reconstruction pressure, a frozen decoder is only a passive observer.

With reconstruction on, the online encoder must remain decoder-compatible.

## 4.2 Stronger latent anchor
We added / increased:

```text
loss_anchor = MSE(z0_online, z0_tgt)
```

This keeps the online latent from drifting too far from the target latent frame.

## 4.3 Predicted-latent match
We also added:

```text
loss_pred_match = MSE(z_cor_pred, z_cor_tgt)
```

So the predicted corrupted latent is tied directly to the target corrupted latent, not just via delta regression.

## 4.4 Slower EMA and lower stage-2 LR
We found the following changes important:

- `ema_decay = 0.999`
- lower stage-2 LR (`1e-4`)

These reduced latent-frame churn and made adaptation smoother.

---

# 5. Stabilized EMA-target regime

With the combination:

- frozen decoder
- nonzero stage-2 reconstruction
- stronger latent anchor
- predicted-latent match
- slower EMA
- lower LR

the model became **stable**.

## What improved
The failure mode changed dramatically:

### Before
- reconstruction collapse
- latent drift
- amplitude explosion

### After
- reconstruction remained near stage-1 quality
- latent drift stayed small
- gate remained selective
- cosine improved steadily
- no catastrophic overshoot or collapse

This established the key positive result:

> **EMA-target stage-2 encoder adaptation can be stabilized, but only with additional geometric and reconstruction constraints.**

---

# 6. Weighted MSE becomes the next problem

At this point we revisited the prediction loss.

## What happened with weighted MSE
In the fixed-latent regime, weighted MSE had been crucial for breaking predictor collapse.

But in the stabilized EMA-target regime, weighted MSE behaved differently:

- predictor stayed alive
- direction improved
- but magnitude drifted upward
- `delta(ratio)` tended to climb toward ~2–2.5

Even with:
- stronger delta L2
- latent predicted-state match
- explicit magnitude loss

the weighted-MSE runs still showed a persistent bias toward oversized updates.

## Interpretation
This suggested a regime shift:

- in the earlier fragile fixed-latent setting, weighting was helping prevent zero-collapse
- in the new stabilized EMA-target setting, weighting was no longer solving the main problem
- instead, it appeared to be contributing to amplitude inflation

This was a major conceptual update.

---

# 7. Added explicit magnitude calibration loss

To attack the remaining scale problem directly, we added:

```text
mag_pred = mean(abs(delta_pred))
mag_tgt = mean(abs(delta_tgt))
loss_mag = MSE(mag_pred, mag_tgt)
```

and included:

```text
loss_total += pred_mag_weight * loss_mag
```

## Effect
This clearly influenced training:

- stronger magnitude weights reduced overshoot
- but with weighted MSE, the objectives began to fight each other
- `pred(mse)` could worsen while scale became more controlled

Conclusion:

> explicit magnitude supervision matters, but weighted delta regression still creates tension in the stabilized EMA-target setting.

---

# 8. Plain MSE becomes viable again

The crucial ablation was to replace weighted MSE with plain MSE while keeping the stabilized EMA-target setup.

## Setup
Plain MSE was tested with:
- EMA target encoder
- stage-2 reconstruction
- latent anchor
- predicted-latent match
- cosine loss
- explicit magnitude loss

## Result
This was the most important new finding.

### Plain MSE did **not** collapse
That was unexpected given earlier experience.

Instead:
- predictor stayed alive
- gate stayed selective
- cosine improved steadily
- reconstruction stayed stable
- latent drift stayed low
- update magnitude became much better calibrated

### Compared to weighted MSE
Plain MSE produced:
- much less overshoot
- no catastrophic drift
- slightly lower but still respectable cosine
- a conservative amplitude bias rather than an explosive one

This suggests:

> once stage-2 stabilization is strong enough, **plain MSE becomes viable again and may actually be preferable**.

---

# 9. Best current EMA-target regime

## Best observed configuration so far
Promising stable setup:

```text
pred_loss = mse
pred_cos_weight = 0.2
pred_mag_weight = 5.0
delta_l2_weight = 1e-4
ema_decay = 0.999
latent_anchor_weight = 0.3
latent_pred_match_weight = 0.3
recon_weight_stage2 = 0.1
stage2_lr = 1e-4
freeze_decoder_stage2 = 1
delta_scale = 0.12
```

## Behavior in the 40-epoch run
This run was stable over a long horizon.

### Main observations
- reconstruction stayed close to stage-1 quality
- latent drift remained low
- gate remained selective
- cosine rose steadily to about `0.32`
- `delta(ratio)` settled around `0.83–0.84`

So the model no longer:
- collapses to zero
- explodes in amplitude
- destroys the latent geometry

Instead it converges to a stable regime with:

- good directional structure
- selective gating
- mild **underprediction** of magnitude

That is a much cleaner failure mode than the previous overshoot regime.

---

# 10. Best-checkpoint behavior

We also modified the script to save the best stage-2 evaluation checkpoint automatically, using reconstruction evaluation as the selection criterion.

## Result
The best stage-2 checkpoint occurred around:

- **epoch 12**

with the lowest observed reconstruction evaluation.

This matters because later epochs:
- remained stable
- continued to improve some metrics like cosine
- but showed mild reconstruction drift

So using only the final epoch would no longer be optimal.

---

# 11. Updated interpretation of the mechanism

The EMA-target system now appears to behave like this:

- **target encoder** provides a slowly moving latent reference frame
- **online encoder** is allowed to adapt, but remains constrained by:
  - reconstruction
  - latent anchor
  - predicted-latent match
- **d_base** provides a persistent always-on residual path
- **d_gate** proposes richer selective residual structure
- **g** modulates selective refinements
- **cosine loss** helps directional alignment
- **magnitude loss** prevents residual scale from drifting too far
- **plain MSE** now works because the rest of the system is stable enough that regression no longer collapses

So the project’s core story still holds, but with an important refinement:

> the viability of the prediction loss depends on the stability of the latent-learning regime.

Weighted loss was important in the fragile fixed-latent rescue phase, but once the latent target is stabilized and encoder drift is controlled, plain MSE can become the better behaved objective.

---

# 12. Main lessons from the EMA-target continuation

## 1. EMA target is useful, but not sufficient by itself
It solves the immediate self-referential target problem, but does not prevent encoder drift on its own.

## 2. Frozen decoder only matters if reconstruction is active
Without stage-2 reconstruction pressure, the online encoder can still abandon the decoder-compatible latent manifold.

## 3. Stabilization required multiple constraints working together
The EMA-target regime only became viable after combining:
- reconstruction
- anchor
- predicted-latent match
- slower EMA
- lower LR

## 4. Weighted MSE is not universally optimal
It helped earlier, but in the stabilized EMA-target phase it biased the model toward amplitude overshoot.

## 5. Plain MSE becomes viable again once the regime is stabilized
This was the biggest conceptual surprise in the continuation.

## 6. The remaining error is now calibration, not collapse
The current best EMA-target regime is not pathological.
Its main limitation is a mild conservative amplitude bias.

---

# 13. Current status

## Best current EMA-target formulation
The best current EMA-target stage-2 model is:

- **EMA target encoder**
- **split-update residual predictor**
- **plain MSE prediction loss**
- **cosine alignment**
- **explicit magnitude supervision**
- **stage-2 reconstruction**
- **latent anchor**
- **predicted-latent match**

## Current qualitative verdict
This is the first EMA-target version that looks genuinely viable.

It does not yet perfectly match target amplitude, but it is:

- stable
- interpretable
- selective
- directionally meaningful
- and much better behaved than the earlier encoder-finetuning attempts

---

# Bottom-line summary

The EMA/frozen-target encoder continuation led to three main conclusions:

1. **EMA target encoder can stabilize stage-2 encoder adaptation, but only with additional reconstruction and latent-geometry constraints.**

2. **Weighted MSE, while crucial in the earlier fixed-latent rescue phase, becomes a source of amplitude overshoot in the stabilized EMA-target regime.**

3. **In the stabilized EMA-target setting, plain MSE with explicit magnitude control is currently the best-behaved objective, yielding a stable selective residual predictor with good cosine alignment and only mild underestimation of update magnitude.**

If you want, I can also turn this into a compact paper-style subsection with headings like:
- *EMA-Target Encoder Stabilization*
- *Loss Regime Shift: Weighted to Plain MSE*
- *Best Current Configuration*

Best test accuracy:  0.9856 (seed 111)