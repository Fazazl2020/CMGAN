import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
import torch
import torch.nn as nn
from utils import LearnableSigmoid


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy, device="cuda"):
    """
    Compute PESQ scores for a batch of audio samples.

    FIXED: Returns individual PESQ scores for each sample, not batch average.
    For failed samples, uses the mean of valid scores in the batch as estimate.

    This ensures discriminator learns to distinguish between samples with
    different quality levels, providing useful per-sample feedback to generator.

    Args:
        clean: List of clean audio samples (numpy arrays)
        noisy: List of noisy audio samples (numpy arrays)
        device: torch device to place the result tensor on (for DDP compatibility)

    Returns:
        Tensor of normalized PESQ scores, shape [batch_size]
    """
    pesq_scores = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_scores = np.array(pesq_scores)

    # Compute fallback value for failed samples
    valid_scores = pesq_scores[pesq_scores != -1]
    if len(valid_scores) > 0:
        fallback_pesq = np.mean(valid_scores)
    else:
        # All samples failed - use middle PESQ value
        fallback_pesq = 2.5  # middle of 1.0-4.5 range

    # Replace failed samples (-1) with fallback value
    pesq_scores[pesq_scores == -1] = fallback_pesq

    # Normalize PESQ scores (1.0-4.5 → 0.0-1.0)
    pesq_normalized = (pesq_scores - 1.0) / 3.5

    # Return individual scores for each sample
    return torch.FloatTensor(pesq_normalized).to(device)


class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)
