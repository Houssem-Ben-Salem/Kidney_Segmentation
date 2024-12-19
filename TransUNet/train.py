import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_kits19
from datasets.dataset_kits19_list import KiTS19DatasetList
import wandb  # Import W&B

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='kits19/data', help='root dir for KiTS19 data')
parser.add_argument('--list_dir', type=str,
                    default='./lists_kits19', help='directory containing train/val/test lists')
parser.add_argument('--dataset', type=str,
                    default='KiTS19', help='Dataset name, e.g., Synapse or KiTS19')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channels for segmentation (background, kidney, tumor)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per GPU')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether to use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='number of skip connections to use')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one ViT model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='ViT patch size, default is 16')
parser.add_argument('--checkpoint_dir', type=str,
                    default='./checkpoints', help='directory to save/load model checkpoints')
parser.add_argument('--use_attention', type=int, default=0, 
                    help='Use Attention Gates in the decoder (1 for Yes, 0 for No)')
args = parser.parse_args()

if __name__ == "__main__":
    # Set deterministic training behavior
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set experiment name for W&B and logging
    args.exp = 'TU_' + args.dataset + '_' + str(args.img_size)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize W&B for logging
    wandb.init(project="KiTS19-Segmentation", name=args.exp, config=vars(args))

    # Load model configuration
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_attention = bool(args.use_attention)  # Pass the attention gate toggle

    if 'R50' in args.vit_name:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # Initialize the model and load pretrained weights
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # Start training using the updated trainer
    trainer_kits19(args, net)