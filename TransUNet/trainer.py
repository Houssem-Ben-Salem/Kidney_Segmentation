import logging
import os
import random
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from datasets.dataset_kits19_list import KiTS19DatasetList
import wandb  # For logging metrics to W&B


def validate(val_loader, model, ce_loss, dice_loss, num_classes):
    """Runs validation and returns average validation loss and Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            # Calculate Dice score per class
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            batch_dice = 0.0
            for i in range(1, num_classes):  # Ignore background (class 0)
                batch_dice += ((2.0 * (outputs == i) * (label_batch == i)).sum()) / (
                    (outputs == i).sum() + (label_batch == i).sum() + 1e-5
                )
            batch_dice /= (num_classes - 1)

            total_loss += loss.item()
            total_dice += batch_dice.item()

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice


def trainer_kits19(args, model, snapshot_path):
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Load datasets
    train_dataset = KiTS19DatasetList(
        list_file=os.path.join(args.list_dir, "train.txt"),
        base_dir=args.root_path,
        slice_size=(args.img_size, args.img_size),
        augment=True
    )
    val_dataset = KiTS19DatasetList(
        list_file=os.path.join(args.list_dir, "val.txt"),
        base_dir=args.root_path,
        slice_size=(args.img_size, args.img_size),
        augment=False
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)

    logging.info(f"{len(train_loader)} iterations per epoch. {max_iterations} max iterations.")

    best_dice = 0.0  # Track the best validation Dice score
    best_model_path = None

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # Log training metrics
            wandb.log({"Train/Loss": loss.item(), "Train/Learning Rate": lr_, "Iteration": iter_num})
            logging.info(f"Iteration {iter_num}: Loss {loss.item():.4f}, Loss_CE {loss_ce.item():.4f}")

        # Run validation at the end of each epoch
        val_loss, val_dice = validate(val_loader, model, ce_loss, dice_loss, num_classes)
        wandb.log({"Val/Loss": val_loss, "Val/Dice": val_dice, "Epoch": epoch_num})
        logging.info(f"Epoch {epoch_num}: Val Loss {val_loss:.4f}, Val Dice {val_dice:.4f}")

        # Save the best model based on validation Dice score
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_path = os.path.join(snapshot_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at {best_model_path}")

    logging.info(f"Training completed. Best Dice: {best_dice:.4f}")
    wandb.save(best_model_path)
    return "Training Finished!"