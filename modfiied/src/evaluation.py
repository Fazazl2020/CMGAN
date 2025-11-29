import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import torch   # make sure torch is imported


# ============== CONFIGURATION ==============
# Modify these values according to your setup
CONFIG = {
    "model_path": "/ghome/fewahab/Sun-Models/Ab-5/CMGAN/ckpt/best_model.pth",  # path to trained model checkpoint
    "test_dir": "/gdata/fewahab/data/VoicebanK-demand-16K/test",              # test dataset directory
    "save_tracks": True,                                                     # save enhanced audio tracks or not
    "save_dir": "/gdata/fewahab/Sun-Models/Ab-5/CMGAN/saved_tracks",         # directory to save enhanced tracks
}
# ===========================================


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    # ----- RMS normalization (same as original CMGAN) -----
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    window = torch.hamming_window(n_fft).cuda()

    # =======================================================
    # TRY ORIGINAL BEHAVIOR FIRST (100% baseline-compatible)
    # =======================================================
    try:
        # This block matches the original CMGAN evaluation logic
        if padded_len > cut_len:
            batch_size = int(np.ceil(padded_len / cut_len))
            while 100 % batch_size != 0:
                batch_size += 1
            noisy_proc = torch.reshape(noisy, (batch_size, -1))
        else:
            noisy_proc = noisy

        noisy_spec = torch.stft(
            noisy_proc, n_fft, hop, window=window, onesided=True
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=window,
            onesided=True,
        )

        # If we had multiple chunks, est_audio is (B, T); flatten like original
        est_audio = est_audio.reshape(-1)

    # =======================================================
    # FALLBACK ON OOM: SMALLER CHUNKS, SEQUENTIAL PROCESSING
    # (only used when original behavior cannot run at all)
    # =======================================================
    except torch.cuda.OutOfMemoryError:
        print(f"[WARNING] OOM on {name} with cut_len={cut_len/16000:.1f}s. "
              f"Falling back to smaller chunks.")
        torch.cuda.empty_cache()

        # Safer, smaller chunk length (e.g., 4 seconds)
        safe_cut_len = 16000 * 4
        padded_len_safe = noisy.size(-1)
        batch_size = int(np.ceil(padded_len_safe / safe_cut_len))
        while 100 % batch_size != 0:
            batch_size += 1

        noisy_chunks = torch.reshape(noisy, (batch_size, -1))
        est_audio_chunks = []

        for i in range(batch_size):
            chunk = noisy_chunks[i:i+1]  # [1, chunk_len]

            noisy_spec = torch.stft(
                chunk, n_fft, hop, window=window, onesided=True
            )
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio_chunk = torch.istft(
                est_spec_uncompress,
                n_fft,
                hop,
                window=window,
                onesided=True,
            )
            est_audio_chunks.append(est_audio_chunk)

        # Concatenate all chunks in time
        est_audio = torch.cat(est_audio_chunks, dim=-1)

    # ----- de-normalize and save (same as original) -----
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length

    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for idx, audio in enumerate(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        # Using ORIGINAL cut_len (16 seconds) - same as original CMGAN
        est_audio, length = enhance_one_track(
            model, noisy_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks
        )
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

        # Clear GPU cache every 10 files to prevent memory buildup
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"Processed {idx + 1}/{num} files...")

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
    )


if __name__ == "__main__":
    noisy_dir = os.path.join(CONFIG["test_dir"], "noisy")
    clean_dir = os.path.join(CONFIG["test_dir"], "clean")
    evaluation(CONFIG["model_path"], noisy_dir, clean_dir, CONFIG["save_tracks"], CONFIG["save_dir"])
