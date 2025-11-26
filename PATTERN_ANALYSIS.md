# CRITICAL PATTERN ANALYSIS - Why PESQ Degrades After Epoch 6-7

## The Consistent Pattern You're Seeing

```
ALL your models show the same pattern:
Epochs 0-2:  PESQ improves (2.45 → 2.64)
Epochs 3-6:  PESQ stable (~2.5-2.6)
Epochs 7+:   PESQ COLLAPSES (2.64 → 2.0-2.2) and STAYS LOW

Train Loss:  Keeps DECREASING (0.162 → 0.121)
Disc Loss:   Keeps DECREASING (0.0136 → 0.0048)
Test Loss:   Keeps DECREASING (0.116 → 0.102)
Test PESQ:   INCREASES then COLLAPSES
```

**This is NOT random variance. This is a REAL systematic problem.**

---

## Why This Matters

You said:
> "It's not only in this baseline, also in other models I modified and made my own models, all of them do same."

**This means the problem is NOT:**
- ❌ A bug in your specific code
- ❌ Dataset issue (pretrained checkpoints work)
- ❌ Random evaluation variance (pattern is too consistent)
- ❌ Model architecture (happens across different models)

**This means the problem IS:**
- ✅ Something in your TRAINING SETUP/ENVIRONMENT
- ✅ A systematic issue affecting all your training runs

---

## HYPOTHESIS: Discriminator Training Collapse

Looking at your discriminator loss:
```
Epoch 0:  Disc Loss = 0.0136
Epoch 2:  Disc Loss = 0.0064
Epoch 8:  Disc Loss = 0.0048
Epoch 16: Disc Loss = 0.0048
```

**The discriminator loss gets VERY small and STOPS changing.**

This suggests:
1. Discriminator stops training effectively after epoch 6-7
2. Generator continues optimizing magnitude loss (0.9 weight)
3. Without discriminator feedback, generator finds magnitude-optimal but PESQ-poor solution
4. PESQ collapses while loss improves

---

## ROOT CAUSE INVESTIGATION

### Critical Questions:

**1. What is your ACTUAL batch_size per GPU?**
```python
# In your server train.py config:
"batch_size": ???  # What value?
```

This matters because:
- batch_size=1: Discriminator trains on 1 sample at a time → very unstable
- batch_size=4: Discriminator trains on 4 samples → more stable

**2. How often does discriminator skip batches?**

Add this logging to your server code:

```python
# In calculate_discriminator_loss(), after pesq_score = discriminator.batch_pesq(...):
global disc_train_count, disc_skip_count
if pesq_score is None:
    disc_skip_count += 1
    print(f"[SKIP] Epoch {epoch}, discriminator skipped (PESQ=None)")
else:
    disc_train_count += 1

# Every 50 batches:
if (disc_train_count + disc_skip_count) % 50 == 0:
    total = disc_train_count + disc_skip_count
    print(f"[DISC STATS] Trained: {disc_train_count}/{total} ({disc_train_count/total*100:.1f}%)")
```

**Expected results:**
- If skip rate > 30%: Discriminator is undertrained → this is your problem
- If skip rate < 10%: Discriminator training frequency is OK

---

## DIAGNOSTIC TEST - Run This NOW

### Test 1: Check Discriminator Predictions

Add this to your code after epoch 8 (when PESQ has collapsed):

```python
# In test() function, after computing PESQ:
# Get discriminator predictions
with torch.no_grad():
    # Pick first batch
    batch = next(iter(self.test_ds))
    clean = batch[0].to(self.device)
    noisy = batch[1].to(self.device)

    gen_outputs = self.forward_generator_step(clean, noisy)

    # What does discriminator predict for (clean, clean)?
    pred_perfect = self.discriminator(
        gen_outputs["clean_mag"],
        gen_outputs["clean_mag"]
    )

    # What does discriminator predict for (clean, estimated)?
    pred_enhanced = self.discriminator(
        gen_outputs["clean_mag"],
        gen_outputs["est_mag"]
    )

    print(f"\n[DISC CHECK] Epoch {epoch}:")
    print(f"  Pred(clean, clean) should be ~1.0: {pred_perfect.mean().item():.4f}")
    print(f"  Pred(clean, enhanced): {pred_enhanced.mean().item():.4f}")
    print(f"  Actual PESQ normalized: {(test_pesq - 1) / 3.5:.4f}")
```

**What to look for:**
- If Pred(clean, clean) ≠ 1.0: Discriminator has collapsed
- If Pred(clean, enhanced) doesn't match actual PESQ: Discriminator is not learning PESQ

---

## MOST LIKELY ROOT CAUSE

Based on the pattern, my strongest hypothesis is:

### **Discriminator collapses after ~30-40k training steps**

**Why this happens:**
1. `batch_pesq()` returns None frequently (silent segments)
2. Discriminator doesn't train on those batches
3. Effective discriminator training rate: < 50% of batches
4. Discriminator learns poorly or overfits
5. Around epoch 6-7, discriminator output becomes meaningless
6. Generator ignores discriminator (GAN weight = 0.05 is too weak anyway)
7. Generator optimizes magnitude loss only
8. PESQ collapses

**This would affect ALL your models because:**
- They all use the same discriminator training code
- They all use the same batch_pesq with silent segment skipping
- They all use the same GAN loss weight (0.05)

---

## IMMEDIATE ACTION - Test My Hypothesis

### Quick Fix Test (30 minutes):

**Modify discriminator.py batch_pesq function:**

```python
def batch_pesq(clean, noisy):
    """Modified to NEVER skip batches"""
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)

    # NEW: Replace failed scores with mean of valid scores
    valid_scores = pesq_score[pesq_score != -1]

    if len(valid_scores) == 0:
        # All samples failed - use a default value
        pesq_normalized = 0.5  # Middle of range
    else:
        # Use mean of valid scores for ALL samples
        pesq_mean = np.mean(valid_scores)
        pesq_normalized = (pesq_mean - 1) / 3.5

    # Return same score for entire batch
    return torch.FloatTensor([pesq_normalized] * len(pesq_score)).to("cuda")
```

**Train for just 10 epochs with this change.**

**If PESQ doesn't collapse at epoch 7:**
→ CONFIRMED: Discriminator skipping is the root cause

**If PESQ still collapses:**
→ Different issue, need to investigate further

---

## Alternative Hypotheses (If Above Fails)

### Hypothesis 2: Learning Rate Too High

Your discriminator LR = 2 × 5e-4 = 1e-3

This might cause:
- Fast initial learning (epochs 0-6)
- Then discriminator overfits/oscillates
- Becomes unreliable after epoch 6

**Test:** Reduce discriminator LR to 5e-4 (same as generator)

### Hypothesis 3: Loss Weight Imbalance

GAN weight (0.05) is 18× smaller than magnitude weight (0.9)

This might cause:
- Early: Magnitude optimization improves PESQ (correlated)
- Later: Magnitude finds local minimum bad for PESQ
- GAN signal too weak to prevent this

**Test:** Increase GAN weight to 0.2 or 0.3

---

## What I Need From You

Before I can give you the DEFINITIVE fix:

**1. What is your actual batch_size?**
```python
# From your server train.py:
CONFIG = {
    "batch_size": ???,  # Tell me this value
    ...
}
```

**2. Run the discriminator skip rate diagnostic** (add the logging above, train 2 epochs)

**3. Run the 10-epoch test** with the modified batch_pesq that never skips

Report back:
- Batch size: ?
- Discriminator skip rate: ?%
- Does PESQ still collapse with modified batch_pesq: Yes/No

Then I can give you the exact fix.

---

## Why I'm Confident This is The Issue

The pattern you describe (consistent collapse at epoch 6-7 across ALL models) + discriminator loss going to near-zero + PESQ degrading while loss improves = **classic GAN training failure mode**.

The discriminator is either:
1. Not training enough (skipping batches)
2. Overfitting/collapsing
3. Being fooled by generator

All three point to the same fix: **Make discriminator training more robust.**
