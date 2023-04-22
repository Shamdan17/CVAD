import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

use_wandb = True

if use_wandb:
    import wandb

    wandb.init(project="cvad_hw1_pred")

use_amp = True
scaler = GradScaler(enabled=use_amp)


# def validate(model, dataloader):
#     """Validate model performance on the validation dataset"""
#     # Your code here
#     val_losses = []
#     model.eval()
#     prg = tqdm(dataloader, desc="Val")
#     all_preds = []
#     all_labels = []

#     for batch in prg:
#         rgb, measurements = batch
#         rgb = rgb.cuda()

#         commands = measurements["command"].cuda()

#         cnt_affordances = measurements["cnt_affordances"].cuda()
#         bin_affordances = measurements["bin_affordances"].cuda()

#         with autocast(enabled=use_amp):
#             with torch.no_grad():
#                 preds = model(rgb, commands)

#         cnt_affordances_pred = preds[:, :3]
#         bin_affordances_pred = preds[:, 3:]

#         all_preds.append(
#             (cnt_affordances_pred.cpu().detach(), bin_affordances_pred.cpu().detach())
#         )
#         all_labels.append(
#             (cnt_affordances.cpu().detach(), bin_affordances.cpu().detach())
#         )

#         # mse loss for continuous affordances, binary cross entropy for binary affordances
#         loss = torch.nn.functional.mse_loss(
#             cnt_affordances_pred, cnt_affordances
#         ) + torch.nn.functional.binary_cross_entropy_with_logits(
#             bin_affordances_pred, bin_affordances.float()
#         )

#         val_losses.append(loss.item())

#         prg.set_postfix(loss=loss.item())

#     return sum(val_losses) / len(val_losses), all_preds, all_labels


def validate(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    losses = []
    all_preds = []
    all_labels = []
    model.eval()
    prg = tqdm(dataloader, desc="Val")

    for batch in prg:
        rgb, measurements = batch
        rgb = rgb.cuda()
        commands = measurements["command"].cuda()

        cnt_affordances = measurements["cnt_affordances"].cuda()
        bin_affordances = measurements["bin_affordances"].cuda()

        with autocast(enabled=use_amp):
            with torch.no_grad():
                preds = model(rgb, commands)

            cnt_affordances_pred = preds[:, :3]
            bin_affordances_pred = preds[:, 3:]

            all_preds.append(
                (
                    cnt_affordances_pred.cpu().detach(),
                    bin_affordances_pred.cpu().detach(),
                )
            )
            all_labels.append(
                (cnt_affordances.cpu().detach(), bin_affordances.cpu().detach())
            )

            # mse loss for continuous affordances, binary cross entropy for binary affordances
            loss = torch.nn.functional.mse_loss(
                cnt_affordances_pred, cnt_affordances
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                bin_affordances_pred, bin_affordances.float()
            )

        losses.append(loss.item())

        prg.set_postfix(loss=loss.item())

    return sum(losses) / len(losses), all_preds, all_labels


def train(model, dataloader, optimizer, sample=0.1):
    """Train model on the training dataset for one epoch"""
    # Your code here
    losses = []
    all_preds = []
    all_labels = []
    model.train()
    # Progress bar for sample percentage of dataset
    prg = tqdm(dataloader, desc="Train", total=int(len(dataloader) * sample))
    max_cnt = int(len(dataloader) * sample)
    cur_cnt = 0

    for batch in prg:
        rgb, measurements = batch
        rgb = rgb.cuda()
        commands = measurements["command"].cuda()

        cnt_affordances = measurements["cnt_affordances"].cuda()
        bin_affordances = measurements["bin_affordances"].cuda()

        with autocast(enabled=use_amp):
            preds = model(rgb, commands)

            cnt_affordances_pred = preds[:, :3]
            bin_affordances_pred = preds[:, 3:]

            all_preds.append(
                (
                    cnt_affordances_pred.cpu().detach(),
                    bin_affordances_pred.cpu().detach(),
                )
            )
            all_labels.append(
                (cnt_affordances.cpu().detach(), bin_affordances.cpu().detach())
            )

            # mse loss for continuous affordances, binary cross entropy for binary affordances
            loss = torch.nn.functional.mse_loss(
                cnt_affordances_pred, cnt_affordances
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                bin_affordances_pred, bin_affordances.float()
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        losses.append(loss.item())

        prg.set_postfix(loss=loss.item())

        cur_cnt += 1
        if cur_cnt >= max_cnt:
            break

    return sum(losses) / len(losses), all_preds, all_labels


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    plt.figure(figsize=(10, 10))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("pred_loss.png")


def plot_prediction_metrics(metric_dicts):
    # 2x2 plot of the metrics
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, metric_name in enumerate(metric_dicts[0].keys()):
        metric = [x[metric_name] for x in metric_dicts]
        ax = axs[i // 2, i % 2]
        ax.plot(metric, label=metric_name)
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE" if "tl_state" in metric_name else "MSE")

    fig.savefig("pred_metrics.png")


affordance_index_map = {
    "lane_dist": 0,
    "route_angle": 1,
    "tl_dist": 2,
}


def calculate_metrics(all_preds, all_labels, prefix=""):
    """Calculate metrics for your predictions"""
    cont_preds = [x[0] for x in all_preds]
    bin_preds = [x[1] for x in all_preds]
    cont_labels = [x[0] for x in all_labels]
    bin_labels = [x[1] for x in all_labels]

    # Concatenate all predictions and labels
    cont_preds = torch.cat(cont_preds, dim=0)
    cont_labels = torch.cat(cont_labels, dim=0)
    bin_preds = torch.cat(bin_preds, dim=0)
    bin_labels = torch.cat(bin_labels, dim=0)

    metric_dict = {}

    for affordance_name, affordance_index in affordance_index_map.items():
        metric_dict[prefix + affordance_name] = torch.nn.functional.mse_loss(
            cont_preds[:, affordance_index], cont_labels[:, affordance_index]
        )

    metric_dict[
        prefix + "tl_state"
    ] = torch.nn.functional.binary_cross_entropy_with_logits(
        bin_preds[:, 0].float(), bin_labels[:, 0].float()
    )

    return metric_dict


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/ssafadoust20/expert_data/train"
    val_root = "/userfiles/ssafadoust20/expert_data/val"
    model = AffordancePredictor().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    train_dataset = ExpertDataset(train_root, train=True)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 50
    batch_size = 256
    save_path = "pred_model.ckpt"

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
    # val_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    # )

    train_losses = []
    val_losses = []
    val_metric_dicts = []
    for i in trange(num_epochs, desc="Epoch"):
        train_loss, train_preds, train_labels = train(model, train_loader, optimizer)
        train_losses.append(train_loss)
        val_loss, val_preds, val_labels = validate(model, val_loader)
        val_losses.append(val_loss)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **calculate_metrics(train_preds, train_labels, prefix="train/"),
                    **calculate_metrics(val_preds, val_labels, prefix="val/"),
                }
            )

        metric_dict = calculate_metrics(val_preds, val_labels)
        val_metric_dicts.append(metric_dict)

        plot_prediction_metrics(val_metric_dicts)
        plot_losses(train_losses, val_losses)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
