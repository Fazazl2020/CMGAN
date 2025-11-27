# COMPREHENSIVE ROOT CAUSE DIAGNOSTIC

## Executive Summary

You're right to be frustrated. I've given you potential fixes but haven't proven they're THE root cause. Let me do a BRUTAL, systematic analysis of EVERY possible bug.

## The Evidence

```
Epoch 0-6: PESQ improves (2.45 → 2.64)
Epoch 7-14: PESQ COLLAPSES (2.64 → 2.07) while loss continues to decrease
```

This is **loss-PESQ divergence** - the model optimizes the wrong objective.

## Analysis: Why I Was WRONG About Batch Size

**My Previous Claim**: "batch_size=1 breaks GAN training due to InstanceNorm"

**Why This Is WRONG**:
1. InstanceNorm2d normalizes **per-sample**, NOT across batch
2. With 4 GPUs × batch_size=4 = effective batch 16 samples
3. InstanceNorm doesn't care about batch size at all

**Conclusion**: Batch size is NOT the root cause. I was wrong.

---

## REAL ROOT CAUSES (Ranked by Likelihood)

### #1: RANDOM TEST SET EVALUATION (ALREADY FIXED ✓)

**Status**: This bug exists in BOTH Baseline and Modified, but you already fixed it.

**The Bug**:
```python
# In dataloader.py - BEFORE FIX:
wav_start = random.randint(0, length - self.cut_len)  # Random every time!
```

**Why It Causes Degradation**:
- Each epoch evaluates DIFFERENT random segments
- PESQ varies randomly, not due to model quality
- Makes training look like it's degrading when it's just evaluation variance

**Evidence This Is NOT The Full Answer**:
- You already fixed this (mode='test' uses center cutting)
- But if you trained BEFORE the fix, the reported PESQ values would be unreliable
- **ACTION**: Did you train with the OLD code (random test) or NEW code (fixed test)?

---

### #2: DISCRIMINATOR TRAINING BUG (MOST LIKELY)

**The Critical Code** (calculate_discriminator_loss, line 386-401):

```python
def calculate_discriminator_loss(self, generator_outputs):
    length = generator_outputs["est_audio"].size(-1)
    est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
    clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
    pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

    if pesq_score is not None:
        predict_enhance_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
        )
        predict_max_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["clean_mag"]
        )
        discrim_loss_metric = F.mse_loss(
            predict_max_metric.flatten(), generator_outputs["one_labels"]
        ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
```

**Potential Bugs**:

#### Bug 2A: PESQ Computation Frequency
- During training, PESQ is computed EVERY batch for discriminator
- If batch_pesq() returns None frequently (due to silent segments), discriminator doesn't train
- Check: How often is `pesq_score = None`?

#### Bug 2B: Audio Length Mismatch
```python
clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
```
- If ISTFT produces different length than STFT input, this slicing could be wrong
- Check: Does `est_audio.size(-1) == clean.size(-1)` always?

#### Bug 2C: Discriminator Sees Wrong Targets
```python
predict_max_metric = self.discriminator(
    generator_outputs["clean_mag"], generator_outputs["clean_mag"]
)
# Should predict 1.0 for (clean, clean)

predict_enhance_metric = self.discriminator(
    generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
)
# Should predict normalized_PESQ for (clean, est)
```

**Critical Question**: Is the discriminator actually learning to predict PESQ?
- If discriminator fails to learn, it gives BAD gradient signal to generator
- Generator optimizes loss but ignores perceptual quality

---

### #3: GENERATOR-DISCRIMINATOR IMBALANCE

**Loss Weights**:
```python
loss_weights = [0.1, 0.9, 0.2, 0.05]
# RI: 0.1, Magnitude: 0.9, Time: 0.2, GAN: 0.05
```

**The Problem**:
- Magnitude loss has weight 0.9
- GAN loss (perceptual feedback) has weight 0.05 (18x smaller!)
- Generator mostly optimizes magnitude, ignores discriminator feedback

**Why This Causes Degradation**:
- Early training: Magnitude loss dominates, PESQ improves as side effect
- Later training: Generator finds local minimum in magnitude space
- This minimum has LOW loss but POOR perceptual quality
- Discriminator feedback (0.05 weight) is too weak to prevent this

**Evidence**:
- Loss keeps decreasing (magnitude optimization working)
- PESQ degrades (perceptual quality ignored)

---

### #4: LEARNING RATE DECAY AT WRONG TIME

**Scheduler Settings**:
```python
decay_epoch = 30  # LR decay starts at epoch 30
gamma = 0.5       # LR halved every 30 epochs
```

**Current Status**:
- Degradation starts at epoch 7
- LR decay hasn't started yet (epoch 30)
- So this is NOT the cause

---

### #5: MULTI-GPU SYNCHRONIZATION BUG

**Comparison**:
- **Baseline**: Uses `mp.spawn` with manual DDP setup
- **Modified**: Uses `torchrun` with `init_distributed_mode()`

**Potential Issue**:
- If discriminator loss is aggregated incorrectly across GPUs
- Or if PESQ scores aren't synchronized properly
- Each GPU might see different training signals

**Check**:
- Are you seeing different logs from different GPU ranks?
- Is the training consistent across GPUs?

---

##  DIAGNOSTIC SCRIPT

I'm creating a diagnostic script you can run to identify the EXACT bug.

```python
# See diagnostic_script.py (created separately)
```

---

## MY HYPOTHESIS (Based on All Evidence)

**Most Likely Root Cause**: **Discriminator is NOT learning to predict PESQ properly**

**Why**:
1. batch_pesq() might return None frequently (silent segments)
2. When it returns None, discriminator doesn't train that batch
3. Discriminator learns poorly or incorrectly
4. Generator receives BAD gradient signal from discriminator
5. Generator optimizes magnitude loss (0.9 weight) but ignores perceptual quality

**How This Explains The Pattern**:
- Epochs 0-6: Generator optimizes magnitude, PESQ improves as side effect
- Epoch 7+: Generator finds magnitude-optimal but perceptually-poor solution
- Discriminator (poorly trained) doesn't prevent this
- Loss decreases, PESQ collapses

**How To Test This**:
1. Log how often `pesq_score = None` during training
2. Log discriminator predictions vs actual PESQ scores
3. Check if discriminator is learning at all

---

## IMMEDIATE ACTIONS

### 1. Check If You Trained With OLD Code
**Question**: When you got those PESQ scores (2.64 → 2.07), did you use:
- [ ] OLD dataloader (random test evaluation)
- [ ] NEW dataloader (fixed center evaluation)

**If OLD**: The PESQ scores are UNRELIABLE due to random evaluation. Restart training with fixed code.

### 2. Run Diagnostic Script
See `diagnostic_training_check.py` (below)

### 3. Compare with Baseline
Train the BASELINE code from scratch and see if it shows the same degradation.

---

## NEXT STEPS BASED ON DIAGNOSIS

**If diagnostic shows**: Discriminator not learning
→ Fix: Increase GAN loss weight from 0.05 to 0.2
→ Or: Remove silent segment filtering in batch_pesq

**If diagnostic shows**: Audio length mismatch
→ Fix: Ensure clean[:, :length] matches est_audio length

**If diagnostic shows**: Multi-GPU sync issue
→ Fix: Use Baseline's mp.spawn instead of torchrun

---

## CONCLUSION

I was wrong about batch_size=1 being the issue. The REAL issue is likely:

**Discriminator fails to learn PESQ prediction → gives bad gradients to generator → generator optimizes magnitude loss but ignores perceptual quality → PESQ collapses**

Run the diagnostic script to confirm.
