from models.generator import TSCNet
from models import discriminator
import os

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

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ============== CONFIGURATION ==============
# Modify these values according to your setup
CONFIG = {
    "epochs": 120,                      # number of epochs of training
    "batch_size": 4,                    # batch size PER GPU (same as baseline)
    "log_interval": 500,                # logging interval
    "decay_epoch": 30,                  # epoch from which to start lr decay
    "init_lr": 5e-4,                    # initial learning rate
    "cut_len": 16000 * 2,               # cut length, 2 seconds for denoise/dereverberation
    "data_dir": "/gdata/fewahab/data/Voicebank+demand/My_train_valid_test/",  # dataset directory
    "save_model_dir": "/ghome/fewahab/Sun-Models/Ab-5/CMGAN",  # directory to save model checkpoints
    "loss_weights": [0.1, 0.9, 0.2, 0.05],  # weights: RI components, magnitude, time loss, Metric Disc
}
# ===========================================

logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    """
    Initialize distributed training (same as baseline).
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)  # Fixed: set device before init_process_group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.gpu_id = gpu_id

        # Create models on the correct GPU (Fixed: use .to(gpu_id) instead of .cuda())
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(gpu_id)

        if gpu_id == 0:
            summary(
                self.model, [(1, 2, CONFIG["cut_len"] // self.hop + 1, int(self.n_fft / 2) + 1)]
            )

        self.discriminator = discriminator.Discriminator(ndf=16).to(gpu_id)

        if gpu_id == 0:
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

        # Wrap models with DDP (same as baseline)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])

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
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
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
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
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
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(CONFIG["batch_size"]).to(self.gpu_id)

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

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(CONFIG["batch_size"]).to(self.gpu_id)

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
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=CONFIG["decay_epoch"], gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=CONFIG["decay_epoch"], gamma=0.5
        )

        for epoch in range(CONFIG["epochs"]):
            # Fixed: Set epoch for distributed sampler (important for proper shuffling!)
            if hasattr(self.train_ds, 'sampler') and self.train_ds.sampler is not None:
                self.train_ds.sampler.set_epoch(epoch)

            self.model.train()
            self.discriminator.train()

            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % CONFIG["log_interval"]) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )

            gen_loss = self.test()

            # Only GPU 0 saves checkpoints (same as baseline)
            if self.gpu_id == 0:
                if not os.path.exists(CONFIG["save_model_dir"]):
                    os.makedirs(CONFIG["save_model_dir"])

                path = os.path.join(
                    CONFIG["save_model_dir"],
                    "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
                )
                torch.save(self.model.module.state_dict(), path)

            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int):
    """
    Main training function (same structure as baseline).
    Args:
        rank: GPU id
        world_size: Total number of GPUs
    """
    ddp_setup(rank, world_size)

    if rank == 0:
        print("=" * 50)
        print("CMGAN Training")
        print("=" * 50)
        print("Configuration:", CONFIG)
        print(f"World size (GPUs): {world_size}")
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print("Available GPUs:", available_gpus)
        print("=" * 50)

    # Load data (same as baseline)
    train_ds, test_ds = dataloader.load_data(
        CONFIG["data_dir"], CONFIG["batch_size"], 2, CONFIG["cut_len"]
    )

    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    # Same as baseline: automatically use all available GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU(s). Starting training...")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
