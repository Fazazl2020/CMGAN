from models.generator import TSCNet
from models import discriminator
import os
import csv
from datetime import datetime
import numpy as np
from pesq import pesq

# ============== ENVIRONMENT SETUP FOR MULTI-GPU ==============
# These must be set BEFORE importing torch
# Note: PYTORCH_CUDA_ALLOC_CONF options like expandable_segments require PyTorch 2.0+
# Keeping only NCCL error handling for compatibility with older PyTorch versions
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# =============================================================

from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ============== CONFIGURATION ==============
# Modify these values according to your setup
CONFIG = {
    "epochs": 120,                      # number of epochs of training
    "batch_size": 1,                    # batch size PER GPU (reduced to 1 for tight GPU memory)
    "log_interval": 500,                # logging interval (steps)
    "decay_epoch": 30,                  # epoch from which to start lr decay
    "init_lr": 5e-4,                    # initial learning rate
    "cut_len": 16000 * 2,               # cut length, 2 seconds for denoise/dereverberation
    "data_dir": "/gdata/fewahab/data/Voicebank+demand/My_train_valid_test/",  # dataset directory
    "save_model_dir": "/ghome/fewahab/Sun-Models/Ab-5/CMGAN",  # directory to save model checkpoints
    "loss_weights": [0.1, 0.9, 0.2, 0.05],  # weights: RI components, magnitude, time loss, Metric Disc
    # ============== RESUME SETTINGS ==============
    "resume": False,                    # Set to True to resume training
    "resume_checkpoint": "/ghome/fewahab/Sun-Models/Ab-5/CMGAN/ckpt/checkpoint_epoch_0.pth",  # Full checkpoint path
    # =============================================
}
# ===========================================

logging.basicConfig(level=logging.INFO)


class TrainingLogger:
    """Logger to save training metrics to CSV files in logs subfolder."""

    def __init__(self, save_dir, resume=False):
        # Create logs subfolder
        self.logs_dir = os.path.join(save_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        # File paths
        self.step_log_path = os.path.join(self.logs_dir, "training_step_log.csv")
        self.epoch_log_path = os.path.join(self.logs_dir, "training_epoch_log.csv")

        # Initialize CSV files (append mode if resuming)
        mode = 'a' if resume and os.path.exists(self.step_log_path) else 'w'

        if mode == 'w':
            # Create step log with headers
            with open(self.step_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'step', 'gen_loss', 'disc_loss', 'lr_gen', 'lr_disc'])

            # Create epoch log with headers (includes PESQ)
            with open(self.epoch_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'train_gen_loss_avg', 'train_disc_loss_avg',
                               'test_gen_loss', 'test_disc_loss', 'test_pesq', 'lr_gen', 'lr_disc', 'is_best'])

    def log_step(self, epoch, step, gen_loss, disc_loss, lr_gen, lr_disc):
        """Log metrics for a single training step."""
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch, step, f"{gen_loss:.6f}", f"{disc_loss:.6f}",
                f"{lr_gen:.8f}", f"{lr_disc:.8f}"
            ])

    def log_epoch(self, epoch, train_gen_loss, train_disc_loss, test_gen_loss, test_disc_loss,
                  test_pesq, lr_gen, lr_disc, is_best=False):
        """Log metrics for a complete epoch (includes PESQ)."""
        with open(self.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch, f"{train_gen_loss:.6f}", f"{train_disc_loss:.6f}",
                f"{test_gen_loss:.6f}", f"{test_disc_loss:.6f}",
                f"{test_pesq:.4f}", f"{lr_gen:.8f}", f"{lr_disc:.8f}",
                "YES" if is_best else "NO"
            ])


def init_distributed_mode():
    """
    Initialize distributed training using environment variables set by torchrun.
    Similar to friend's utils.init_distributed_mode(args)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = True
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        distributed = True
    else:
        print("Not running in distributed mode")
        distributed = False
        rank = 0
        local_rank = 0
        world_size = 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        dist.barrier()

    return distributed, rank, local_rank, world_size


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


class Trainer:
    def __init__(self, train_ds, test_ds, local_rank: int, distributed: bool = True):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.local_rank = local_rank
        self.distributed = distributed
        self.start_epoch = 0  # Will be updated if resuming
        self.best_loss = float('inf')  # Track best model

        # Create checkpoint and logs directories
        self.ckpt_dir = os.path.join(CONFIG["save_model_dir"], "ckpt")
        if is_main_process():
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Initialize logger (only on main process)
        if is_main_process():
            self.logger = TrainingLogger(CONFIG["save_model_dir"], resume=CONFIG["resume"])
        else:
            self.logger = None

        # Create device
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create models on the correct GPU
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(self.device)

        if is_main_process():
            summary(
                self.model, [(1, 2, CONFIG["cut_len"] // self.hop + 1, int(self.n_fft / 2) + 1)]
            )

        self.discriminator = discriminator.Discriminator(ndf=16).to(self.device)

        if is_main_process():
            summary(
                self.discriminator,
                [
                    (1, 1, int(self.n_fft / 2) + 1, CONFIG["cut_len"] // self.hop + 1),
                    (1, 1, int(self.n_fft / 2) + 1, CONFIG["cut_len"] // self.hop + 1),
                ],
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG["init_lr"])
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * CONFIG["init_lr"]
        )

        # Create schedulers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=CONFIG["decay_epoch"], gamma=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=CONFIG["decay_epoch"], gamma=0.5
        )

        # Load checkpoint if resuming
        if CONFIG["resume"] and CONFIG["resume_checkpoint"]:
            self.load_checkpoint(CONFIG["resume_checkpoint"])

        # Wrap models with DDP if distributed (AFTER loading checkpoint)
        if distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
            self.discriminator = DDP(self.discriminator, device_ids=[local_rank], output_device=local_rank)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            if is_main_process():
                print(f"Checkpoint not found: {checkpoint_path}")
            return

        if is_main_process():
            print(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Load optimizer states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])

        # Load scheduler states
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        # Set start epoch (resume from next epoch)
        self.start_epoch = checkpoint['epoch'] + 1

        # Load best loss if available
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']

        if is_main_process():
            print(f"Resumed from epoch {checkpoint['epoch']}, will start from epoch {self.start_epoch}")
            print(f"Best loss so far: {self.best_loss:.6f}")

    def save_checkpoint(self, epoch, gen_loss, pesq_score):
        """Save checkpoint with all states for resuming. Returns True if best model."""
        if not is_main_process():
            return False

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Get model state dict (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        disc_state = self.discriminator.module.state_dict() if self.distributed else self.discriminator.state_dict()

        # Check if this is the best model (lowest loss)
        is_best = gen_loss < self.best_loss
        if is_best:
            self.best_loss = gen_loss

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'discriminator_state_dict': disc_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'gen_loss': gen_loss,
            'pesq_score': pesq_score,
            'best_loss': self.best_loss,
            'config': CONFIG,
        }

        # Save full checkpoint for resuming (in ckpt folder)
        checkpoint_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

        # Save model-only checkpoint (for evaluation, in ckpt folder)
        model_path = os.path.join(
            self.ckpt_dir,
            f"CMGAN_epoch_{epoch}_loss_{gen_loss:.4f}_pesq_{pesq_score:.3f}.pth"
        )
        torch.save(model_state, model_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(model_state, best_path)
            best_checkpoint_path = os.path.join(self.ckpt_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"*** New best model saved! Loss: {gen_loss:.6f}, PESQ: {pesq_score:.4f} ***")

        return is_best

    def forward_generator_step(self, clean, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            CONFIG["loss_weights"][0] * loss_ri
            + CONFIG["loss_weights"][1] * loss_mag
            + CONFIG["loss_weights"][2] * time_loss
            + CONFIG["loss_weights"][3] * gen_loss_GAN
        )

        return loss

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
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
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(CONFIG["batch_size"]).to(self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(CONFIG["batch_size"]).to(self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        """Run test/validation and compute PESQ score."""
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        pesq_scores = []

        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss

            # Calculate PESQ for this batch
            clean = batch[0].to(self.device)
            noisy = batch[1].to(self.device)
            with torch.no_grad():
                outputs = self.forward_generator_step(clean, noisy)
                est_audio = outputs["est_audio"]

            # Compute PESQ for each sample in batch
            length = est_audio.size(-1)
            for i in range(est_audio.size(0)):
                try:
                    est_np = est_audio[i].cpu().numpy()
                    clean_np = clean[i, :length].cpu().numpy()
                    # PESQ requires 16kHz sample rate
                    score = pesq(16000, clean_np, est_np, 'wb')
                    pesq_scores.append(score)
                except Exception:
                    pass  # Skip if PESQ calculation fails (silent segments)

        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        pesq_avg = np.mean(pesq_scores) if pesq_scores else 0.0

        template = "GPU: {}, Test Gen loss: {:.6f}, Disc loss: {:.6f}, PESQ: {:.4f}"
        logging.info(template.format(self.local_rank, gen_loss_avg, disc_loss_avg, pesq_avg))

        return gen_loss_avg, disc_loss_avg, pesq_avg

    def train(self):
        for epoch in range(self.start_epoch, CONFIG["epochs"]):
            # Set epoch for distributed sampler (important for proper shuffling!)
            if self.distributed and hasattr(self.train_ds, 'sampler') and self.train_ds.sampler is not None:
                self.train_ds.sampler.set_epoch(epoch)

            self.model.train()
            self.discriminator.train()

            # Track epoch losses
            epoch_gen_loss_total = 0.0
            epoch_disc_loss_total = 0.0
            num_steps = 0

            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                num_steps = step
                loss, disc_loss = self.train_step(batch)
                epoch_gen_loss_total += loss
                epoch_disc_loss_total += disc_loss

                # Log every log_interval steps
                if (step % CONFIG["log_interval"]) == 0:
                    template = "GPU: {}, Epoch {}, Step {}, loss: {:.6f}, disc_loss: {:.6f}"
                    logging.info(
                        template.format(self.local_rank, epoch, step, loss, disc_loss)
                    )
                    # Log to CSV (only on main process)
                    if self.logger is not None:
                        lr_gen = self.optimizer.param_groups[0]['lr']
                        lr_disc = self.optimizer_disc.param_groups[0]['lr']
                        self.logger.log_step(epoch, step, loss, disc_loss, lr_gen, lr_disc)

            # Calculate epoch averages
            train_gen_loss_avg = epoch_gen_loss_total / num_steps
            train_disc_loss_avg = epoch_disc_loss_total / num_steps

            # Run test (returns loss and PESQ)
            test_gen_loss, test_disc_loss, test_pesq = self.test()

            # Save checkpoint (includes all states for resuming)
            is_best = self.save_checkpoint(epoch, test_gen_loss, test_pesq)

            # Log epoch summary to CSV (only on main process)
            if self.logger is not None:
                lr_gen = self.optimizer.param_groups[0]['lr']
                lr_disc = self.optimizer_disc.param_groups[0]['lr']
                self.logger.log_epoch(epoch, train_gen_loss_avg, train_disc_loss_avg,
                                     test_gen_loss, test_disc_loss, test_pesq,
                                     lr_gen, lr_disc, is_best)

            # Print epoch summary
            if is_main_process():
                print(f"\n{'='*60}")
                print(f"Epoch {epoch} Summary:")
                print(f"  Train Loss: {train_gen_loss_avg:.6f}, Disc Loss: {train_disc_loss_avg:.6f}")
                print(f"  Test Loss: {test_gen_loss:.6f}, Test PESQ: {test_pesq:.4f}")
                print(f"  Best Model: {'YES' if is_best else 'NO'} (Best Loss: {self.best_loss:.6f})")
                print(f"{'='*60}\n")

            # Synchronize all processes before next epoch
            if dist.is_initialized():
                dist.barrier()

            self.scheduler_G.step()
            self.scheduler_D.step()


def main():
    """
    Main training function.
    Supports both torchrun and single GPU training.
    """
    # Initialize distributed mode (like friend's code)
    distributed, rank, local_rank, world_size = init_distributed_mode()

    if is_main_process():
        print("=" * 50)
        print("CMGAN Training")
        print("=" * 50)
        print("Configuration:", CONFIG)
        print(f"Distributed: {distributed}")
        print(f"World size (GPUs): {world_size}")
        print(f"Rank: {rank}, Local Rank: {local_rank}")
        if CONFIG["resume"]:
            print(f"Resuming from: {CONFIG['resume_checkpoint']}")
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print("Available GPUs:", available_gpus)
        print("=" * 50)

    # Load data
    train_ds, test_ds = dataloader.load_data(
        CONFIG["data_dir"], CONFIG["batch_size"], 2, CONFIG["cut_len"]
    )

    trainer = Trainer(train_ds, test_ds, local_rank, distributed)
    trainer.train()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
