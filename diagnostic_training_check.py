#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC SCRIPT

This script helps identify the root cause of PESQ degradation during training.

Usage:
    Modify this script to add logging statements to your training code,
    then run training and analyze the output.
"""

import sys
sys.path.insert(0, './modfiied/src')

# ADD THESE LOGGING STATEMENTS TO YOUR TRAINING CODE
# ===================================================

diagnostic_instructions = """
DIAGNOSTIC INSTRUCTIONS
=======================

Add these logging statements to modfiied/src/train.py:

1. In calculate_discriminator_loss(), AFTER line 389:
   -----------------------------------------------------
   # Log PESQ computation frequency
   if pesq_score is None:
       print(f"[DIAGNOSTIC] Epoch {epoch}, Step {step}: PESQ = None (silent segment)")
   else:
       print(f"[DIAGNOSTIC] Epoch {epoch}, Step {step}: PESQ scores = {pesq_score.cpu().numpy()}")

2. In calculate_discriminator_loss(), AFTER line 398:
   -----------------------------------------------------
   # Log discriminator predictions
   pred_max = predict_max_metric.flatten().detach().cpu().numpy()
   pred_enh = predict_enhance_metric.flatten().detach().cpu().numpy()
   print(f"[DIAGNOSTIC] Discriminator predictions:")
   print(f"  predict_max (should be ~1.0): {pred_max}")
   print(f"  predict_enhance (should match PESQ): {pred_enh}")
   if pesq_score is not None:
       print(f"  actual PESQ (normalized): {pesq_score.cpu().numpy()}")
       print(f"  prediction error: {np.abs(pred_enh - pesq_score.cpu().numpy())}")

3. In forward_generator_step(), AFTER line 343:
   ----------------------------------------------
   # Check audio lengths
   print(f"[DIAGNOSTIC] Audio lengths:")
   print(f"  clean.size(): {clean.size()}")
   print(f"  est_audio.size(): {est_audio.size()}")
   if clean.size(-1) != est_audio.size(-1):
       print(f"  WARNING: Length mismatch! clean={clean.size(-1)}, est={est_audio.size(-1)}")

4. In train_step(), AFTER line 436:
   ----------------------------------
   # Log discriminator training frequency
   global disc_train_count, disc_skip_count
   if discrim_loss_metric.item() > 0:
       disc_train_count += 1
   else:
       disc_skip_count += 1
   if step % 100 == 0:
       print(f"[DIAGNOSTIC] Discriminator training stats:")
       print(f"  Trained: {disc_train_count}, Skipped: {disc_skip_count}")
       print(f"  Train rate: {disc_train_count/(disc_train_count+disc_skip_count)*100:.1f}%")

5. At the top of train() method, add:
   ------------------------------------
   global disc_train_count, disc_skip_count
   disc_train_count = 0
   disc_skip_count = 0


WHAT TO LOOK FOR IN THE LOGS:
==============================

1. PESQ = None frequency
   ----------------------
   If you see "PESQ = None" more than 20% of the time:
   → Discriminator is NOT learning properly (not enough training signal)
   → FIX: Remove silent segment filtering or use different normalization

2. Discriminator predictions
   --------------------------
   Check if predict_max ≈ 1.0 and predict_enhance ≈ normalized_PESQ

   If predict_max is NOT close to 1.0:
   → Discriminator is not learning the "clean vs clean = perfect" signal
   → FIX: Increase discriminator learning rate or training iterations

   If prediction error is large (>0.2):
   → Discriminator cannot predict PESQ accurately
   → FIX: Discriminator architecture too weak, or needs more training

3. Audio length mismatch
   ----------------------
   If you see "WARNING: Length mismatch":
   → PESQ is computed on mismatched lengths → incorrect scores
   → FIX: Ensure ISTFT output matches STFT input length

4. Discriminator training rate
   ----------------------------
   If "Train rate" < 80%:
   → Discriminator skips too many batches (PESQ = None)
   → Generator doesn't get enough perceptual feedback
   → FIX: This is the likely root cause!


AFTER RUNNING DIAGNOSTICS:
==========================

1. Save the diagnostic output to a file:
   python train.py 2>&1 | tee diagnostic_output.txt

2. Check the patterns above

3. Report findings for targeted fix
"""

print(diagnostic_instructions)
print("\n" + "="*70)
print("QUICK DIAGNOSIS QUESTIONS:")
print("="*70)
print()
print("Answer these to narrow down the issue:")
print()
print("1. Did you train with the OLD dataloader (random test) or NEW (center cut)?")
print("   OLD: PESQ values are unreliable, restart training")
print("   NEW: Continue investigating")
print()
print("2. Does the BASELINE code show the same degradation when trained from scratch?")
print("   YES: Bug is in the model/algorithm itself, not your modifications")
print("   NO: Bug is in your modifications vs baseline")
print()
print("3. What's your effective batch size with 4 GPUs?")
print("   batch_size=4 per GPU × 4 GPUs = 16 samples per update")
print("   This should be fine for GAN training")
print()
print("4. Check your training logs for 'PESQ = None' messages")
print("   If > 20% of batches: Discriminator undertrained")
print()
