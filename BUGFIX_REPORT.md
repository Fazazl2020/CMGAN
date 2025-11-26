# CMGAN Training Performance Bug Fix Report

## Executive Summary

Systematic analysis identified **TWO CRITICAL BUGS** causing training performance degradation:
1. **Batch Size = 1** (instead of required 4) → GAN training instability
2. **Random test set augmentation** → Non-comparable PESQ scores across epochs

Both bugs have been fixed. **Restart training from scratch** with the corrected code.

---

## Bug #1: Batch Size = 1 Breaks GAN Training ⚠️ CRITICAL

### Root Cause
**File:** `modfiied/src/train.py:30`

```python
# BEFORE (BROKEN):
"batch_size": 1,  # batch size PER GPU (reduced to 1 for tight GPU memory)

# AFTER (FIXED):
"batch_size": 4,  # batch size PER GPU (CRITICAL: must be >= 4 for stable GAN training)
```

### Why This Causes Performance Degradation

#### 1. Instance Normalization Collapse
- The discriminator uses `nn.InstanceNorm2d` layers (discriminator.py:36,41,46,51)
- Instance norm computes statistics **per sample** in the batch
- With batch_size=1, statistics are computed from **a single sample**
- This makes normalization unreliable and causes training instability

#### 2. Noisy Discriminator Gradients
- The metric discriminator computes PESQ scores for each batch
- With only 1 sample per batch, gradient updates are **extremely noisy**
- This prevents the discriminator from learning stable patterns

#### 3. GAN Training Collapse
- GANs require stable batch statistics for convergence
- Batch size = 1 typically leads to:
  - Mode collapse
  - Training divergence
  - Oscillating loss values
- **Matches observed pattern**: PESQ degrades from 2.64 → 2.20 → 2.07

### Evidence from Original Implementation
**Source:** [Official CMGAN Repository](https://github.com/ruizhecao96/CMGAN)

```python
# Original hyperparameters:
batch_size = 4  # ← Confirmed from source code
init_lr = 5e-4
disc_lr = 1e-3
loss_weights = [0.1, 0.9, 0.2, 0.05]
```

All other hyperparameters match, **except batch_size**.

---

## Bug #2: Random Cutting in Test Set → Inconsistent PESQ Evaluation

### Root Cause
**File:** `modfiied/src/data/dataloader.py:47-49` (and Baseline)

```python
# BEFORE (BROKEN):
else:
    # randomly cut 2 seconds segment
    wav_start = random.randint(0, length - self.cut_len)  # ← RANDOM every time!
    noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
    clean_ds = clean_ds[wav_start : wav_start + self.cut_len]
```

### Why This Causes Erratic PESQ Scores

#### Problem Description
- Every call to `__getitem__()` returns a **DIFFERENT random 2-second segment**
- During training: Good! (data augmentation)
- During testing: **BAD!** Each epoch evaluates different segments
- PESQ scores are **NOT COMPARABLE** across epochs

#### Evidence from Training Logs
```
Epoch 2: PESQ: 2.6394  ← High score on some random segments
Epoch 7: PESQ: 2.2148  ← Low score on different random segments
Epoch 8: PESQ: 2.2043  ← Different segments again
```

These fluctuations are **partially due to evaluating different segments**, not just model performance.

### The Fix

```python
# AFTER (FIXED):
class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2, mode='train'):
        """
        mode: 'train' for random cutting (augmentation)
              'test' for center cutting (consistent evaluation)
        """
        self.mode = mode
        # ...

    def __getitem__(self, idx):
        # ...
        if self.mode == 'train':
            # Random cutting for data augmentation during training
            wav_start = random.randint(0, length - self.cut_len)
        else:
            # Fixed center cutting for consistent test evaluation
            wav_start = (length - self.cut_len) // 2
        # ...
```

**Usage:**
```python
train_ds = DemandDataset(train_dir, cut_len, mode='train')  # Random augmentation
test_ds = DemandDataset(test_dir, cut_len, mode='test')     # Consistent evaluation
```

---

## Files Modified

### Modified Code:
1. ✅ `modfiied/src/train.py` - Changed batch_size from 1 to 4
2. ✅ `modfiied/src/data/dataloader.py` - Added `mode` parameter for consistent test evaluation

### Baseline Code (for consistency):
3. ✅ `Baseline/src/data/dataloader.py` - Added `mode` parameter for consistent test evaluation

---

## Action Required

### 1. **Delete Old Checkpoints** (REQUIRED)
The old checkpoints were trained with the broken configuration (batch_size=1). They cannot be used.

```bash
rm -rf /ghome/fewahab/Sun-Models/Ab-5/CMGAN/ckpt/*.pth
```

### 2. **Restart Training from Scratch**
```bash
# Set resume to False in train.py
CONFIG = {
    "resume": False,  # ← Make sure this is False
    "batch_size": 4,  # ← Verified to be 4
    # ...
}
```

### 3. **Expected Results**
With the fixes:
- ✅ Training should be **stable** (no PESQ degradation)
- ✅ PESQ should **monotonically improve** or stabilize
- ✅ Test PESQ should be **comparable across epochs**
- ✅ Final PESQ should match paper results (~3.4+)

---

## GPU Memory Considerations

**Original comment:** "reduced to 1 for tight GPU memory"

### If you still have GPU memory issues with batch_size=4:

**Option 1: Gradient Accumulation** (Recommended)
```python
# Accumulate gradients over multiple steps to simulate larger batch
accumulation_steps = 4
for i, batch in enumerate(train_ds):
    loss = train_step(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Option 2: Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = train_step(batch)
```

**Option 3: Reduce Audio Cut Length**
```python
"cut_len": 16000 * 1,  # 1 second instead of 2
```

**DO NOT reduce batch_size below 4** - it breaks GAN training fundamentally.

---

## Verification Checklist

Before restarting training, verify:
- [ ] `batch_size = 4` in `modfiied/src/train.py:30`
- [ ] `mode='test'` used for test dataset in `dataloader.py:79`
- [ ] `resume = False` in `modfiied/src/train.py:39`
- [ ] Old checkpoints deleted
- [ ] Sufficient GPU memory (or use gradient accumulation)

---

## References

- [CMGAN Paper (arXiv:2203.15149)](https://arxiv.org/abs/2203.15149)
- [Official CMGAN Repository](https://github.com/ruizhecao96/CMGAN)
- [CMGAN Extended Paper (arXiv:2209.11112)](https://arxiv.org/abs/2209.11112)

---

**Report Generated:** 2025-11-26
**Analysis Method:** Systematic code comparison with original implementation
