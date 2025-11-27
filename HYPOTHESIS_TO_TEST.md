# HYPOTHESIS TO TEST - Discriminator Batch Skipping Fix

## The Problem

Analysis of training logs shows:
- **56% of batches** have discriminator loss near zero (< 0.001)
- Suggests discriminator is not training on many batches
- Pattern consistent: PESQ improves epochs 0-6, collapses epochs 7+
- Happens across ALL user's models

## The Hypothesis

**Original batch_pesq logic:**
```python
if -1 in pesq_score:  # If ANY sample fails
    return None        # Skip ENTIRE batch
```

When `None` is returned, discriminator doesn't train that batch.

**Theory:**
1. ~50% of batches are skipped (silent segments cause PESQ errors)
2. Discriminator gets insufficient training
3. After epoch 6-7, discriminator signal becomes unreliable
4. Generator optimizes magnitude loss (90% weight) without perceptual feedback
5. PESQ collapses while loss decreases

## The Fix

**Modified batch_pesq:**
```python
# Instead of skipping, use mean of valid PESQ scores
valid_scores = pesq_score[pesq_score != -1]
if len(valid_scores) == 0:
    pesq_normalized = 0.5  # Default if all fail
else:
    pesq_mean = np.mean(valid_scores)
    pesq_normalized = (pesq_mean - 1) / 3.5

return torch.FloatTensor([pesq_normalized] * len(pesq_score)).to("cuda")
```

**Expected Effect:**
- Discriminator trains on 100% of batches (instead of ~50%)
- More consistent perceptual feedback to generator
- PESQ should not collapse at epoch 6-7

## CRITICAL CAVEAT

**I am NOT 100% certain this is the root cause.**

**Confidence level: 70%**

Why uncertainty:
1. Original CMGAN has the same batch skipping logic - yet produces good results
2. Small disc_loss could mean discriminator is working well (accurate predictions)
3. Loss weight imbalance (GAN 5% vs Magnitude 90%) might be the real issue
4. Dataset or environment differences not accounted for

## Testing Protocol

1. **Apply this fix** to your server code
2. **Delete old checkpoints** (trained with old code)
3. **Train for 10 epochs**
4. **Check PESQ at epoch 7-10**

**If PESQ stops collapsing:**
✅ Hypothesis confirmed - this was the root cause
→ Continue training to 120 epochs

**If PESQ still collapses:**
❌ Hypothesis wrong - need to investigate other causes:
- Loss weight adjustment (increase GAN weight from 0.05 to 0.2)
- Learning rate schedule
- Data preprocessing differences
- Environment/hardware differences

## Files Modified

- `modfiied/src/models/discriminator.py` - batch_pesq function
- `Baseline/src/models/discriminator.py` - same fix for consistency

## Next Steps After Testing

**Report back:**
1. Does PESQ still collapse at epoch 7? (Yes/No)
2. What is PESQ at epoch 10?
3. Discriminator loss pattern (still near-zero on many batches?)

Then we'll know if this is the fix or if we need to look elsewhere.

---

**Bottom Line:** This is my best hypothesis based on log analysis, but it needs empirical testing to confirm. I'm giving you the fix to test, not claiming certainty.
