# THE HONEST TRUTH - I Was Wrong

## What I Found in the ORIGINAL CMGAN Code

I checked the original Baseline code (commit 01cdfeb, before my modifications):

### ORIGINAL Dataloader (Baseline/src/data/dataloader.py):
```python
class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):  # NO mode parameter
        self.cut_len = cut_len
        # ...

    def __getitem__(self, idx):
        # ...
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)  # ALWAYS random!
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

# Load data:
train_ds = DemandDataset(train_dir, cut_len)  # Random cutting
test_ds = DemandDataset(test_dir, cut_len)   # ALSO random cutting!
```

### ORIGINAL batch_pesq (Baseline/src/models/discriminator.py):
```python
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(...)
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:  # Skip entire batch if ANY sample fails
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")
```

---

## MY MISTAKE

**The ORIGINAL CMGAN code has BOTH "bugs" I claimed to fix:**
1. ❌ Random test evaluation (no mode parameter)
2. ❌ Skips batches when any PESQ fails

**Yet the original code produces PESQ ~3.41 results in the paper!**

**Conclusion**: These are NOT bugs preventing reproduction. They are the original design.

---

## Why Does the Original Work Despite These "Issues"?

### Reason #1: Final Evaluation Uses `evaluation.py`, Not `test()`

The paper results come from `evaluation.py`, which:
```python
def enhance_one_track(model, audio_path, ...):
    # Processes THE ENTIRE audio file (not random 2-second segments)
    # Uses overlapping windows if needed
    # Returns full enhanced audio
```

**Key Insight**:
- The `test()` function in `train.py` is just for monitoring during training
- Random test evaluation variance doesn't matter for final model selection
- Authors likely evaluated final checkpoints using `evaluation.py` on FULL audio files

### Reason #2: Random Test Variance Averages Out Over 120 Epochs

Your results are from epochs 0-14 only. If you train to epoch 120:
- Some epochs will randomly select "easy" segments (high PESQ)
- Some epochs will randomly select "hard" segments (low PESQ)
- Over 120 epochs, the variance averages out
- You can still see the overall trend (increasing/decreasing)

### Reason #3: Discriminator Skipping is Conservative Design

Skipping batches when PESQ fails might be intentional:
- Only train discriminator on reliable PESQ scores
- Prevents discriminator from learning on noisy/incorrect labels
- Better to skip than train on wrong signal

---

## What Your Training Logs Actually Mean

```
Epoch 2:  PESQ=2.64  ← Randomly selected easy segments
Epoch 7:  PESQ=2.21  ← Randomly selected hard segments
Epoch 14: PESQ=2.07  ← Different random segments
```

**These numbers DON'T prove your model is degrading!**

They prove: Random test evaluation has high variance.

---

## What You Should Actually Do

### Option 1: Train Exactly Like the Original (RECOMMENDED)

1. **Use the original code** (with random test evaluation)
2. **Train for FULL 120 epochs** (you only trained 14!)
3. **Ignore test() PESQ values** during training (they're noisy)
4. **Evaluate final model** using `evaluation.py` on full test files
5. **Compare final PESQ** to paper results

### Option 2: Use My "Improvements" (Better Monitoring, Not Required)

1. **Apply mode='test' fix** for stable test PESQ monitoring
2. **Apply batch_pesq fix** for more discriminator training
3. **Train for 120 epochs**
4. **Evaluate with evaluation.py**

**Both options should give ~3.41 PESQ if:**
- Dataset is correct
- Hyperparameters match (batch_size=4, lr=5e-4, etc.)
- Train for full 120 epochs

---

## Why I Was Confused

You showed me logs from epochs 0-14 with "degradation":
- I assumed this was real degradation
- I tried to find bugs preventing reproduction
- I didn't realize you're using random test evaluation (like the original)
- I didn't realize you only trained 14 epochs (not 120)

**The "degradation" is likely just random test variance, not a real problem.**

---

## CRITICAL QUESTION FOR YOU

**Did you train the Baseline code from scratch for 120 epochs?**

- If YES, and it also shows PESQ degradation → then there's a real issue
- If NO (you only used pretrained checkpoints) → you haven't verified baseline works

**Action**: Train the UNMODIFIED Baseline code for 120 epochs and check:
1. Does test() PESQ fluctuate randomly? (Expected: YES)
2. Does evaluation.py on final checkpoint give ~3.41 PESQ? (Expected: YES)

If Baseline training fails too → the issue is dataset/environment, not code.

---

## My Apologies

I gave you "fixes" without verifying the original code does the same things I called "bugs."

**The truth**:
- Your code matches the original design
- Random test evaluation is intentional (or at least standard practice)
- Discriminator skipping is in the original code
- You need to train for 120 epochs and use evaluation.py for final results

**My recommendations**:
- Are "improvements" for cleaner training monitoring
- But NOT required to reproduce paper results
- The original code (with its quirks) already works

---

## Bottom Line

**Question**: Why can't you reproduce CMGAN results?

**Answer**: You might already be reproducing them correctly! But:
1. You only trained 14 epochs (not 120)
2. You're judging based on test() PESQ (which has high variance)
3. You haven't evaluated the final model with evaluation.py

**Next Steps**:
1. Train your current code (or Baseline) for FULL 120 epochs
2. Pick best checkpoint based on test loss (not PESQ)
3. Run evaluation.py on that checkpoint
4. Compare to paper results

If that still fails, THEN we debug further.
