from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

use_wandb = True

if use_wandb:
    import wandb

    wandb.init(project="cvad_hw1_cilrs")

use_amp = True
scaler = GradScaler(enabled=use_amp)

loss_weights = {
    "speed": 1.0,
    "throttle": 1.0,
    "brake": 1.0,
    "steer": 1.0,
}


# def validate(model, dataloader):
#     """Validate model performance on the validation dataset"""
#     # Your code here
#     val_losses = defaultdict(float)
#     model.eval()
#     prg = tqdm(dataloader, desc="Val")

#     for batch in prg:
#         rgb, measurements = batch
#         rgb = rgb.cuda()
#         speeds = measurements["speed"].cuda()

#         commands = measurements["command"].cuda()

#         throttle = measurements["throttle"].cuda()
#         brake = measurements["brake"].cuda()
#         steer = measurements["steer"].cuda()

#         with torch.no_grad():
#             pred_actions, pred_speeds = model(rgb, speeds, commands)
#             pred_throttle = pred_actions[:, 0].unsqueeze(-1)
#             pred_brake = pred_actions[:, 1].unsqueeze(-1)
#             pred_steer = pred_actions[:, 2].unsqueeze(-1)

#             # L1 loss
#             # Save losses separately so we can plot them later
#             loss_throttle = (
#                 torch.nn.functional.l1_loss(pred_throttle, throttle)
#                 * loss_weights["throttle"]
#             )
#             loss_brake = (
#                 torch.nn.functional.l1_loss(pred_brake, brake) * loss_weights["brake"]
#             )
#             loss_steer = (
#                 torch.nn.functional.l1_loss(pred_steer, steer) * loss_weights["steer"]
#             )
#             loss_speed = (
#                 torch.nn.functional.l1_loss(pred_speeds, speeds) * loss_weights["speed"]
#             )
#             loss = loss_throttle + loss_brake + loss_steer + loss_speed

#         val_losses["throttle"] += loss_throttle.item()
#         val_losses["brake"] += loss_brake.item()
#         val_losses["steer"] += loss_steer.item()
#         val_losses["speed"] += loss_speed.item()
#         val_losses["total"] += loss.item()

#         prg.set_postfix(loss=loss.item())

#     print(val_losses)

#     return {k: v / len(dataloader) for k, v in val_losses.items()}


def validate(model, dataloader):
    """Validate model on the val dataset for one epoch"""
    # Your code here
    val_losses = defaultdict(float)
    model.eval()
    prg = tqdm(dataloader, desc="Val")

    for batch in prg:
        rgb, measurements = batch
        rgb = rgb.cuda()
        speeds = measurements["speed"].cuda()

        commands = measurements["command"].cuda()

        throttle = measurements["throttle"].cuda()
        brake = measurements["brake"].cuda()
        steer = measurements["steer"].cuda()

        with torch.no_grad():
            with autocast(enabled=use_amp):
                pred_actions, pred_speeds = model(rgb, speeds, commands)

                pred_throttle = pred_actions[:, 0].unsqueeze(-1)
                pred_brake = pred_actions[:, 1].unsqueeze(-1)
                pred_steer = pred_actions[:, 2].unsqueeze(-1)

                # L1 loss
                # Save losses separately so we can plot them later
                loss_throttle = (
                    torch.nn.functional.l1_loss(pred_throttle, throttle)
                    * loss_weights["throttle"]
                )
                loss_brake = (
                    torch.nn.functional.l1_loss(pred_brake, brake)
                    * loss_weights["brake"]
                )
                loss_steer = (
                    torch.nn.functional.l1_loss(pred_steer, steer)
                    * loss_weights["steer"]
                )
                loss_speed = (
                    torch.nn.functional.l1_loss(pred_speeds, speeds)
                    * loss_weights["speed"]
                )
                loss = loss_throttle + loss_brake + loss_steer + loss_speed

        val_losses["throttle"] += loss_throttle.item()
        val_losses["brake"] += loss_brake.item()
        val_losses["steer"] += loss_steer.item()
        val_losses["speed"] += loss_speed.item()
        val_losses["total"] += loss.item()

        prg.set_postfix(loss=loss.item())

    # print(val_losses)

    return {k: v / len(dataloader) for k, v in val_losses.items()}


def train(model, dataloader, optimizer, sample=0.1):
    """Train model on the training dataset for one epoch"""
    # Your code here
    train_losses = defaultdict(float)
    model.train()
    # Progress bar for sample percentage of dataset
    prg = tqdm(dataloader, desc="Train", total=int(len(dataloader) * sample))
    max_cnt = int(len(dataloader) * sample)
    cur_cnt = 0

    for batch in prg:
        rgb, measurements = batch
        rgb = rgb.cuda()
        speeds = measurements["speed"].cuda()

        commands = measurements["command"].cuda()

        throttle = measurements["throttle"].cuda()
        brake = measurements["brake"].cuda()
        steer = measurements["steer"].cuda()

        with autocast(enabled=use_amp):
            pred_actions, pred_speeds = model(rgb, speeds, commands)

            pred_throttle = pred_actions[:, 0].unsqueeze(-1)
            pred_brake = pred_actions[:, 1].unsqueeze(-1)
            pred_steer = pred_actions[:, 2].unsqueeze(-1)

            # L1 loss
            # Save losses separately so we can plot them later
            loss_throttle = (
                torch.nn.functional.l1_loss(pred_throttle, throttle)
                * loss_weights["throttle"]
            )
            loss_brake = (
                torch.nn.functional.l1_loss(pred_brake, brake) * loss_weights["brake"]
            )
            loss_steer = (
                torch.nn.functional.l1_loss(pred_steer, steer) * loss_weights["steer"]
            )
            loss_speed = (
                torch.nn.functional.l1_loss(pred_speeds, speeds) * loss_weights["speed"]
            )
            loss = loss_throttle + loss_brake + loss_steer + loss_speed

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_losses["throttle"] += loss_throttle.item()
        train_losses["brake"] += loss_brake.item()
        train_losses["steer"] += loss_steer.item()
        train_losses["speed"] += loss_speed.item()
        train_losses["total"] += loss.item()

        prg.set_postfix(loss=loss.item())

        cur_cnt += 1
        if cur_cnt >= max_cnt:
            break

    # print(train_losses)

    return {k: v / len(dataloader) for k, v in train_losses.items()}


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # 5 plots, one for each loss term and one for the total loss
    fig, ax = plt.subplots(5, 1, figsize=(7, 30))
    for i, k in enumerate(train_loss[0]):
        print(k)
        key_name = k
        # Make first letter uppercase and add Loss at the end
        key_name = key_name[0].upper() + key_name[1:] + " Loss"

        ax[i].plot([x[k] for x in train_loss], label="Train")
        ax[i].plot([x[k] for x in val_loss], label="Val")
        ax[i].set_title(key_name)
        ax[i].legend()
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Loss")

    fig.savefig("cilrs_loss.png")


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/ssafadoust20/expert_data/train"
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    model = CILRS().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    train_dataset = ExpertDataset(train_root, train=True)
    val_dataset = ExpertDataset(val_root)
    # val_dataset = ExpertDataset(train_root, train=True)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 50
    batch_size = 256
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for i in trange(num_epochs, desc="Epoch"):
        train_losses.append(train(model, train_loader, optimizer))
        val_losses.append(validate(model, val_loader))

        if use_wandb:
            for k in train_losses[-1]:
                wandb.log({f"train/{k}": train_losses[-1][k]})
            for k in val_losses[-1]:
                wandb.log({f"val/{k}": val_losses[-1][k]})
        # If validation loss is lower than previous best, save model
        if val_losses[-1]["total"] < best_val_loss:
            best_val_loss = val_losses[-1]["total"]
            torch.save(model.state_dict(), save_path)
        plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
