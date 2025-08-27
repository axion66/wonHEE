import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image

from layers.transformer import Transformer, MODEL_CONFIG
from dataloader import loader
from layers.utils import get_device
from dataclasses import asdict
from pprint import pprint


class TrainAgent:
    def __init__(self):
        self.global_step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.debug = True
        self.main()

    def log_sequence_plot(self, src, output, mode, mode_type="train", mask_idx=None):
        idx = 0
        seq_original = src[idx].cpu().numpy().squeeze()
        seq_output = output[idx].detach().cpu().numpy().squeeze()

        plt.figure(figsize=(10, 4))

        if mode == "forecasting" and mask_idx is not None:  # show obs + predicted at masked indices
            mask_indices_np = mask_idx.cpu().numpy()
            observed_idx = np.setdiff1d(np.arange(len(seq_original)), mask_indices_np)
            plt.plot(observed_idx, seq_original[observed_idx], color='tab:green', label="Observed")
            plt.plot(mask_indices_np, seq_output[mask_indices_np], color='tab:red', label="Predicted")
        
        elif mode == "imputation" and mask_idx is not None:  # scatter plot
            mask_indices_np = mask_idx.cpu().numpy()
            line = seq_original.copy()
            line[mask_indices_np] = seq_output[mask_indices_np]
            observed_idx = np.setdiff1d(np.arange(len(line)), mask_indices_np)
            plt.scatter(observed_idx, line[observed_idx], color='tab:green', label="Observed")
            plt.scatter(mask_indices_np, line[mask_indices_np], color='tab:red', label="Imputed")
        
        else:  # other modes
            plt.plot(seq_output, label="Predicted", linestyle="--")

        plt.title(f"{mode} - Reconstructed ({mode_type})")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img = Image.open(buf)
        wandb.log({f"{mode}/{mode_type}_reconstructed": wandb.Image(img)}, step=self.global_step)

        plt.figure(figsize=(10, 4))
        plt.plot(seq_original, label="Ground Truth", color='tab:blue')
        plt.title(f"{mode} - Truth ({mode_type})")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img = Image.open(buf)
        wandb.log({f"{mode}/{mode_type}_truth": wandb.Image(img)}, step=self.global_step)


    def log_metrics(self, prefix, loss, stats=None, mode=None):
        wandb.log({f"{prefix}/loss": loss}, step=self.global_step)
        if mode:
            wandb.log({f"{prefix}/loss_{mode}": loss}, step=self.global_step)
        if stats:
            wandb.log({
                f"{prefix}/classification-correct": stats["correct"],
                f"{prefix}/classification-total": stats["total"],
                f"{prefix}/classification-accuracy": stats["correct"] / stats["total"],
            }, step=self.global_step)

    def validate_epoch(self, loader):
        self.model.eval()
        val_loss, count = 0.0, 0
        mode_losses = {m: [] for m in self.modes}
        global_correct, global_total = 0, 0  

        with torch.no_grad():
            for src, labels in tqdm(loader, desc="Validating"):
                src, labels = src.to(self.device), labels.to(self.device)
                if len(src.shape) == 2:
                    src = src.unsqueeze(-1)

                for mode in self.modes:
                    loss, stats, output, mask_idx = self.model.compute_loss(
                        src, labels, mode=mode, device=self.device
                    )
                    self.log_metrics("val", loss.item(), stats, mode)

                    if mode == "classification" and stats is not None:
                        global_correct += stats["correct"]
                        global_total += stats["total"]

                    if mode != "classification":
                        self.log_sequence_plot(src, output, mode, mode_type="val", mask_idx=mask_idx)

                    val_loss += loss.item()
                    mode_losses[mode].append(loss.item())
                count += 1

        avg_val_loss = val_loss / count if count > 0 else 0.0
        self.val_loss_history.append(avg_val_loss)
        wandb.log({f"val/loss": avg_val_loss}, step=self.global_step)

        for mode, losses in mode_losses.items():
            if losses:
                wandb.log({f"val/loss_{mode}": np.mean(losses)}, step=self.global_step)

        if global_total > 0:
            global_acc = global_correct / global_total
            wandb.log({
                "val/global_classification_accuracy": global_acc,
                "val/global_classification_correct": global_correct,
                "val/global_classification_total": global_total,
            }, step=self.global_step)

        self.model.train()

    def train_epoch(self):
        self.model.train()
        for batch_idx, (src, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            self.optimizer.zero_grad()
            src, labels = src.to(self.device), labels.to(self.device)
            if len(src.shape) == 2:
                src = src.unsqueeze(-1)

            src += torch.randn_like(src) * 0.05
            
            losses = []
            for mode in self.modes:
                loss, stats, output, mask_idx = self.model.compute_loss(src, labels, mode=mode, device=self.device)
                losses.append(loss)
                train_loss = loss.item()
                self.train_loss_history.append(train_loss)
                self.log_metrics("train", train_loss, stats, mode)
                if batch_idx % 40 == 0 and mode in ["imputation", "forecasting"]:
                    self.log_sequence_plot(src, output, mode, mode_type="train", mask_idx=mask_idx)

            total_loss = torch.stack(losses).sum()
            total_loss.backward()
            self.optimizer.step()
            self.global_step += 1

            if (batch_idx + 1) == 50:
                self.validate_epoch(self.test_loader)

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--model_dim', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_encoder', type=int, default=4)
        parser.add_argument('--seq_len', type=int, default=640)
        parser.add_argument('--features_path', type=str, default='dataset/MIT-BIH_features.npy')
        parser.add_argument('--labels_path', type=str, default='dataset/MIT-BIH_labels.npy')
        parser.add_argument('--balance_labels', action='store_true', default=True)
        args = parser.parse_args()

        wandb.init(
            project="ECG_submit", 
            name="MoE_final",
            mode="online",
            save_code=False,
        )
        
        self.device = get_device()
        self.modes = ["classification", "forecasting", "imputation"]

        self.train_loader, self.test_loader = loader(
            features_path=args.features_path,
            labels_path=args.labels_path,
            batch_size=args.batch_size,
            balance_labels=args.balance_labels
        )

        cfg = MODEL_CONFIG(
            input_dim=1,
            dim=args.model_dim,
            output_dim=1,
            n_head=args.num_heads,
            seq_len=args.seq_len,
            use_rotary=False,
            use_moe=True,
            n_expert=5,
            n_encoder_block=args.num_encoder,
            n_sub_head_block=2,
            n_class=len(np.unique([y.item() for _, y in self.train_loader.dataset]))
        )
        
        wandb.config.update(asdict(cfg))
        wandb.config.update(vars(args))
        pprint(asdict(cfg))
        self.model = Transformer(cfg).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            self.train_epoch()
            # self.validate_epoch(self.test_loader)
            torch.save(self.model, "checkpoint/best.pt")


if __name__ == "__main__":
    TrainAgent()
