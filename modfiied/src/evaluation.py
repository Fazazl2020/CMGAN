import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf

# ============== CONFIGURATION ==============
# Modify these values according to your setup
CONFIG = {
    "model_path": "/ghome/fewahab/Sun-Models/Ab-5/CMGAN/best_ckpt",  # path to trained model checkpoint
    "test_dir": "/gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test",  # test dataset directory
    "save_tracks": True,  # save enhanced audio tracks or not
    "save_dir": "/gdata/fewahab/Sun-Models/Ab-5/CMGAN/saved_tracks",  # directory to save enhanced tracks
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

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    # MEMORY OPTIMIZATION: Process chunks sequentially instead of all at once
    # Original code processes all chunks simultaneously (high memory)
    # This version processes one chunk at a time (low memory, identical results)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy_chunks = torch.reshape(noisy, (batch_size, -1))

        # Process each chunk separately to save memory
        est_audio_chunks = []
        for i in range(batch_size):
            chunk = noisy_chunks[i:i+1]  # Single chunk [1, chunk_len]

            noisy_spec = torch.stft(
                chunk, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
            )
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio_chunk = torch.istft(
                est_spec_uncompress,
                n_fft,
                hop,
                window=torch.hamming_window(n_fft).cuda(),
                onesided=True,
            )
            est_audio_chunks.append(est_audio_chunk)

        # Concatenate all chunks
        est_audio = torch.cat(est_audio_chunks, dim=-1)
    else:
        # Short audio, process directly (same as original)
        noisy_spec = torch.stft(
            noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).cuda(),
            onesided=True,
        )

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
        # Sequential chunk processing avoids OOM while giving identical results
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
