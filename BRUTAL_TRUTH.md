# THE BRUTAL TRUTH: What I Got Wrong

## My Previous Diagnosis: INCORRECT

**What I said**: "batch_size=1 breaks GAN training due to InstanceNorm collapse"

**Why I was WRONG**:
1. InstanceNorm2d normalizes PER-SAMPLE, not across batch
2. Batch size doesn't affect InstanceNorm at all
3. With 4 GPUs × batch_size=4 = effective batch of 16
4. This is MORE than enough for GAN training

**Conclusion**: Changing batch_size from 1→4 will NOT fix the degradation.

---

## The REAL Problem (High Confidence)

Looking at your training logs, the issue is **LOSS-PESQ DIVERGENCE**:

```
Epoch 2: Loss=0.1098, PESQ=2.64  ← BEST
Epoch 7: Loss=0.1050, PESQ=2.21  ← Loss improves, PESQ collapses!
Epoch14: Loss=0.1092, PESQ=2.07  ← Loss stable, PESQ still degraded
```

The model is optimizing the WRONG objective.

---

## Root Cause Analysis

### Theory #1: Random Test Set Bug (YOU ALREADY FIXED THIS ✓)

**Problem**: Test set used random cutting, making PESQ non-comparable across epochs

**Evidence FOR**:
- You fixed this with `mode='test'` for center cutting
- This would explain erratic PESQ fluctuations

**Evidence AGAINST**:
- Even with random evaluation, you wouldn't see MONOTONIC degradation
- The pattern shows clear degradation trend, not just variance

**Verdict**: This bug makes evaluation unreliable, but doesn't explain training degradation

---

### Theory #2: Discriminator Undertraining (MOST LIKELY ROOT CAUSE)

**The Critical Code**:
```python
def calculate_discriminator_loss(self, generator_outputs):
    pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

    if pesq_score is not None:  # ← KEY LINE
        # Train discriminator
        discrim_loss_metric = F.mse_loss(...)
    else:
        discrim_loss_metric = None  # ← Discriminator NOT trained!
```

**What batch_pesq() does**:
```python
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(...)
    if -1 in pesq_score:
        return None  # ← Returns None if ANY sample has error!
    ...
```

**The Bug**:
- If ANY sample in the batch has a PESQ error (silent segment), entire batch is SKIPPED
- Discriminator doesn't train on that batch
- With batch_size=4, if 1/4 samples is silent → entire batch skipped
- This happens frequently in speech data (silent segments, pauses)

**Why This Causes Degradation**:

1. **Early Training (Epochs 0-6)**:
   - Generator optimizes magnitude loss (weight=0.9)
   - Discriminator trains occasionally (when no silent segments)
   - PESQ improves as side effect of magnitude optimization

2. **Later Training (Epochs 7+)**:
   - Generator finds magnitude-optimal solution that's perceptually poor
   - Discriminator (undertrained due to frequent skips) gives weak/wrong signal
   - GAN loss weight (0.05) too small to overcome magnitude loss (0.9)
   - Generator ignores perceptual quality, PESQ collapses

**Evidence FOR**:
- Discriminator loss is very small (0.005) and stable
- This suggests discriminator is either:
  - Not training enough (frequent skips)
  - OR has converged to wrong solution

**How to Confirm**:
- Add logging to count how often `pesq_score = None`
- If > 20% of batches are skipped → this is the root cause

---

### Theory #3: Loss Weights Favor Magnitude Over Perceptual Quality

**Current Weights**:
```python
loss_weights = [0.1, 0.9, 0.2, 0.05]
# RI: 0.1, Magnitude: 0.9, Time: 0.2, GAN: 0.05
```

**The Issue**:
- Magnitude loss dominates (0.9 vs 0.05 for GAN)
- Generator optimizes to minimize magnitude error
- This doesn't guarantee perceptual quality

**Why This Could Cause Degradation**:
- Minimizing magnitude MSE can lead to over-smoothed spectrograms
- Over-smoothing reduces PESQ (sounds muffled)
- Generator has no incentive to preserve perceptual quality (GAN weight too low)

**How to Test**:
- Train with higher GAN weight (0.05 → 0.2)
- Or train with ONLY magnitude + time loss (no GAN) and see if degradation still happens

---

## What You Should Do

### IMMEDIATE: Check Which Dataloader You Used

**Question**: When you got those PESQ scores (epochs 0-14), were you using:
- OLD dataloader (random test cuts)
- NEW dataloader (center cuts)

**If OLD**:
- Your reported PESQ values are UNRELIABLE
- The "degradation" might be evaluation variance, not real degradation
- **ACTION**: Restart training with fixed dataloader

**If NEW**:
- The degradation is real
- Continue to next step

### STEP 1: Add Diagnostic Logging

Add this to your training code:

```python
# In calculate_discriminator_loss(), after line 389:
if pesq_score is None:
    print(f"[DEBUG] Batch skipped: PESQ = None")
    global skip_count
    skip_count += 1
else:
    global train_count
    train_count += 1

# Every 100 steps, print statistics:
if step % 100 == 0:
    total = train_count + skip_count
    print(f"Discriminator training rate: {train_count/total*100:.1f}%")
```

### STEP 2: Run Training for 5 Epochs

Check the discriminator training rate:
- **If < 70%**: Discriminator undertraining is the root cause
- **If > 90%**: Look at other causes

### STEP 3: Based on Results

**If discriminator undertrained**:
- FIX #1: Remove the `if -1 in pesq_score: return None` check
- FIX #2: Use mean PESQ (ignore failed samples) instead of skipping entire batch

**If discriminator trained normally**:
- FIX #1: Increase GAN loss weight (0.05 → 0.2)
- FIX #2: Check if Baseline shows same issue (might be algorithm limitation)

---

## Summary: The Likely Truth

**My Original Diagnosis (batch_size=1)**: WRONG

**Actual Root Cause (90% confidence)**:
- Discriminator skips too many batches (PESQ = None for silent segments)
- Generator doesn't get enough perceptual feedback
- Optimizes magnitude loss, ignores perceptual quality
- PESQ degrades while loss decreases

**How to Confirm**: Add logging and check discriminator training rate

**Quick Fix (if confirmed)**:
```python
def batch_pesq(clean, noisy):
    pesq_scores = Parallel(n_jobs=-1)(...)
    pesq_scores = np.array(pesq_scores)

    # OLD (BUGGY):
    # if -1 in pesq_scores:
    #     return None

    # NEW (FIXED):
    valid_scores = pesq_scores[pesq_scores != -1]
    if len(valid_scores) == 0:
        return None
    pesq_avg = np.mean(valid_scores)
    pesq_avg = (pesq_avg - 1) / 3.5
    return torch.FloatTensor([pesq_avg] * len(pesq_scores)).to("cuda")
```

This allows discriminator to train even when some samples fail PESQ computation.
