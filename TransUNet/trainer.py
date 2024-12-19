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

            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            batch_dice = 0.0
            for i in range(1, num_classes):
                batch_dice += ((2.0 * (outputs == i) * (label_batch == i)).sum()) / (
                    (outputs == i).sum() + (label_batch == i).sum() + 1e-5
                )
            batch_dice /= (num_classes - 1)

            total_loss += loss.item()
            total_dice += batch_dice.item()

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice


def log_visualizations(image_batch, label_batch, outputs, attn_weights, iter_num):
    """Log segmentation outputs and attention maps to W&B."""
    if attn_weights is not None:
        for i, attn in enumerate(attn_weights[:2]):  # Log attention maps from the first two layers
            avg_attn = attn.mean(dim=1).cpu().numpy()  # Average over heads
            wandb.log({
                f"Attention Map/Layer {i}": wandb.Image(avg_attn, caption=f"Iteration {iter_num}")
            })

    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
    wandb.log({
        "Input Image": wandb.Image(image_batch[0].cpu().numpy(), caption=f"Iteration {iter_num}"),
        "Prediction": wandb.Image(predictions[0], caption=f"Iteration {iter_num}"),
        "Ground Truth": wandb.Image(label_batch[0].cpu().numpy(), caption=f"Iteration {iter_num}"),
    })


def trainer_kits19(args, model):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

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

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
    start_iter = 0
    best_dice = 0.0

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"Best model found at {best_model_path}. Resuming training...")
    elif os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_iter = checkpoint['iter_num']
        best_dice = checkpoint['best_dice']
        logging.info(f"Resuming from checkpoint at iteration {start_iter}. Best Dice so far: {best_dice:.4f}")
    else:
        logging.info("No checkpoint or best model found. Starting training from scratch.")

    iter_num = start_iter
    max_iterations = args.max_iterations
    save_interval = 2000

    model.train()
    logging.info(f"{len(train_loader)} iterations per epoch. {max_iterations} total iterations.")
    iterator = tqdm(range(start_iter, max_iterations), ncols=70)

    for _ in iterator:
        for sampled_batch in train_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs, attn_weights = model(image_batch, return_attn=True)
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

            wandb.log({"Train/Loss": loss.item(), "Train/LR": lr_, "Iteration": iter_num})
            logging.info(f"Iteration {iter_num}: Loss {loss.item():.4f}")

            if iter_num % 1000 == 0:
                log_visualizations(image_batch, label_batch, outputs, attn_weights, iter_num)

            if iter_num % save_interval == 0:
                val_loss, val_dice = validate(val_loader, model, ce_loss, dice_loss, num_classes)
                wandb.log({"Val/Loss": val_loss, "Val/Dice": val_dice, "Iteration": iter_num})
                logging.info(f"Validation: Iteration {iter_num}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

                checkpoint = {
                    'iter_num': iter_num,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_dice': best_dice
                }
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved latest checkpoint at {checkpoint_path}")

                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"New best model saved at {best_model_path} with Dice {best_dice:.4f}")

            if iter_num >= max_iterations:
                logging.info("Max iterations reached. Training complete.")
                iterator.close()
                break

    logging.info(f"Training finished. Best Dice score: {best_dice:.4f}")
    wandb.save(best_model_path)
    return "Training Finished!"