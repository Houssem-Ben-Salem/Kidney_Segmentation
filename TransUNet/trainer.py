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
import wandb

def get_scaled_learning_rate(base_lr, batch_size, reference_batch_size=24, scaling_method='linear'):
    """
    Scale learning rate based on batch size.
    
    Args:
        base_lr: Base learning rate for reference batch size
        batch_size: Current batch size
        reference_batch_size: Reference batch size that the base_lr was tuned for
        scaling_method: 'linear' or 'sqrt' scaling
        
    Returns:
        Scaled learning rate
    """
    if scaling_method == 'linear':
        # Linear scaling: lr ∝ batch_size
        return base_lr * (batch_size / reference_batch_size)
    elif scaling_method == 'sqrt':
        # Square root scaling: lr ∝ √(batch_size)
        return base_lr * (batch_size / reference_batch_size) ** 0.5
    else:
        return base_lr
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # Pre-define Sobel kernels
        self.sobel_x = None
        self.sobel_y = None
    
    def forward(self, pred, target, softmax=True):
        # Initialize or move sobel filters to the correct device
        if self.sobel_x is None or self.sobel_x.device != pred.device:
            self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
            self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                      dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        if softmax:
            pred_softmax = torch.softmax(pred, dim=1)
        else:
            pred_softmax = pred
        
        # Safety check for class index bounds
        num_classes = pred_softmax.size(1)
        if num_classes <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Create one-hot encoding of target
        target_one_hot = torch.zeros_like(pred_softmax)
        for b in range(target.size(0)):
            target_classes = target[b].unique()
            for c in target_classes:
                if c < num_classes:  # Ensure class index is valid
                    target_one_hot[b, c] = (target[b] == c).float()
        
        total_loss = 0.0
        valid_classes = 0
        
        # Process only non-background classes
        for class_idx in range(1, num_classes):
            # Skip if class doesn't exist in this batch
            if not torch.any(target_one_hot[:, class_idx] > 0):
                continue
                
            try:
                # Prediction edges
                edge_x_pred = torch.nn.functional.conv2d(
                    pred_softmax[:, class_idx:class_idx+1], self.sobel_x, padding=1
                )
                edge_y_pred = torch.nn.functional.conv2d(
                    pred_softmax[:, class_idx:class_idx+1], self.sobel_y, padding=1
                )
                
                # Add epsilon inside the square root to prevent NaN gradients
                edge_pred = torch.sqrt(edge_x_pred**2 + edge_y_pred**2 + 1e-8)
                
                if torch.max(edge_pred) > 0:  # Only normalize if non-zero
                    edge_pred = edge_pred / (torch.max(edge_pred) + 1e-8)
                
                # Target edges with normalization
                edge_x_target = torch.nn.functional.conv2d(
                    target_one_hot[:, class_idx:class_idx+1], self.sobel_x, padding=1
                )
                edge_y_target = torch.nn.functional.conv2d(
                    target_one_hot[:, class_idx:class_idx+1], self.sobel_y, padding=1
                )
                
                # Also add epsilon here for consistency
                edge_target = torch.sqrt(edge_x_target**2 + edge_y_target**2 + 1e-8)
                
                if torch.max(edge_target) > 0:  # Only normalize if non-zero
                    edge_target = edge_target / (torch.max(edge_target) + 1e-8)
                
                # Explicitly ensure values are in [0,1] range
                edge_pred = torch.clamp(edge_pred, 0.0, 1.0)
                edge_target = torch.clamp(edge_target, 0.0, 1.0)
                
                # Use MSE loss instead of BCE for safer computation
                class_loss = torch.nn.functional.mse_loss(edge_pred, edge_target)
                
                total_loss += class_loss
                valid_classes += 1
                
            except Exception as e:
                print(f"Error processing class {class_idx}: {e}")
                continue
        
        # Return average loss or zero if no valid classes
        if valid_classes > 0:
            return total_loss / valid_classes
        else:
            # Create a zero tensor with gradient requirements
            return torch.sum(pred_softmax[:, 0] * 0)
    
def validate(val_loader, model, ce_loss, dice_loss, boundary_loss, focal_loss, num_classes):
    """Runs validation and returns detailed metrics per class."""
    model.eval()
    metrics = {
        'total_loss': 0.0,
        'boundary_loss': 0.0,
        'dice_kidney': 0.0,
        'dice_tumor': 0.0
    }
    num_batches = len(val_loader)

    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_focal = focal_loss(outputs, label_batch.long())
            loss_boundary = boundary_loss(outputs, label_batch)
            loss = 0.25 * loss_ce + 0.4 * loss_dice + 0.15 * loss_focal + 0.2 * loss_boundary

            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Calculate Dice per class
            for class_idx, class_name in [(1, 'kidney'), (2, 'tumor')]:
                dice = ((2.0 * (outputs == class_idx) * (label_batch == class_idx)).sum()) / (
                    (outputs == class_idx).sum() + (label_batch == class_idx).sum() + 1e-5
                )
                metrics[f'dice_{class_name}'] += dice.item()
            
            metrics['total_loss'] += loss.item()
            metrics['boundary_loss'] += loss_boundary.item()

    # Average the metrics
    for key in metrics:
        metrics[key] /= num_batches
        
    return metrics

def trainer_kits19(args, model):
    # Set default save_interval if not provided in args
    if not hasattr(args, 'save_interval'):
        args.save_interval = 2000  # Default value, saves every 2000 iterations
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    reference_batch_size = 32  # Typical batch size you usually use

    # Calculate scaled learning rate
    scaled_lr = get_scaled_learning_rate(
        base_lr, 
        batch_size, 
        reference_batch_size=reference_batch_size,
        scaling_method='sqrt'  # 'sqrt' is often more stable
    )
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

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # Initialize losses with class weights
    class_weights = torch.tensor([0.01, 1.45, 6.25]).cuda()  # Based on pixel distribution
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss(gamma=2)
    boundary_loss = BoundaryLoss()
    # Initialize optimizer and scheduler with adjusted learning rate for AdamW
    # Adjust for AdamW
    adjusted_lr = scaled_lr * 0.1
    logging.info(f"Base LR: {base_lr}, Scaled LR: {scaled_lr}, Adjusted LR: {adjusted_lr}")

    optimizer = optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=0.0001)
    
    # Cosine scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,  # First restart cycle length
        T_mult=2,   # Multiply cycle length by 2 after each restart
        eta_min=adjusted_lr * 0.01  # Minimum learning rate
    )

    # Load checkpoints if they exist
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
    start_iter = 0
    best_dice = 0.0
    best_tumor_dice = 0.0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_iter = checkpoint['iter_num']
        best_dice = checkpoint['best_dice']
        best_tumor_dice = checkpoint.get('best_tumor_dice', 0.0)
        logging.info(f"Resuming from iteration {start_iter}. Best Dice: {best_dice:.4f}")

    model.train()
    iter_num = start_iter
    max_iterations = args.max_iterations
    
    # Calculate total steps for tqdm
    total_steps = max_iterations - start_iter
    pbar = tqdm(total=total_steps, initial=0, desc='Training Progress')

    while iter_num < max_iterations:
        for sampled_batch in train_loader:
            if iter_num >= max_iterations:
                break
                
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass
            outputs = model(image_batch)
            if isinstance(outputs, tuple):
                outputs, attn_weights = outputs
            else:
                attn_weights = None
            
            # Combined loss with weights
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_focal = focal_loss(outputs, label_batch.long())
            loss_boundary = boundary_loss(outputs, label_batch)
            # Adjusted weights with boundary loss
            loss = 0.25 * loss_ce + 0.4 * loss_dice + 0.15 * loss_focal + 0.2 * loss_boundary


            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            iter_num += 1
            pbar.update(1)
            current_lr = scheduler.get_last_lr()[0]

            # Update progress bar
            pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'boundary': f'{loss_boundary.item():.4f}',  # Add this to monitor boundary loss
            'lr': f'{current_lr:.6f}'
             })

            # Logging
            # Add boundary loss to logging
            wandb.log({
                "Train/Loss": loss.item(),
                "Train/CE_Loss": loss_ce.item(),
                "Train/Dice_Loss": loss_dice.item(),
                "Train/Focal_Loss": loss_focal.item(),
                "Train/Boundary_Loss": loss_boundary.item(),
                "Train/LR": current_lr,
                "Iteration": iter_num
            })

            # Validation and checkpointing
            if iter_num % args.save_interval == 0:
                metrics = validate(val_loader, model, ce_loss, dice_loss, boundary_loss, focal_loss, num_classes)
                
                wandb.log({
                "Val/Loss": metrics['total_loss'],
                "Val/Boundary_Loss": metrics['boundary_loss'],  # Add this line
                "Val/Dice_Kidney": metrics['dice_kidney'],
                "Val/Dice_Tumor": metrics['dice_tumor'],
                "Iteration": iter_num
                })

                avg_dice = (metrics['dice_kidney'] + metrics['dice_tumor']) / 2
                
                # Save checkpoint
                checkpoint = {
                    'iter_num': iter_num,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'best_tumor_dice': best_tumor_dice
                }
                torch.save(checkpoint, checkpoint_path)

                # Save best models
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"New best model saved with Dice {best_dice:.4f}")

                if metrics['dice_tumor'] > best_tumor_dice:
                    best_tumor_dice = metrics['dice_tumor']
                    torch.save(model.state_dict(), 
                             os.path.join(args.checkpoint_dir, "best_tumor_model.pth"))
                    logging.info(f"New best tumor model saved with Dice {best_tumor_dice:.4f}")

    pbar.close()
    logging.info(f"Training finished. Best overall Dice: {best_dice:.4f}")
    logging.info(f"Best tumor Dice: {best_tumor_dice:.4f}")
    return "Training Finished!"