import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from matplotlib.colors import Normalize

def visualize_transformer_attention(attention_weights, img, save_path=None, layer_idx=-1, head_idx=None):
    """
    Visualizes transformer self-attention weights overlaid on the original image.
    
    Args:
        attention_weights: List of attention weights from each transformer layer [B, H, N, N]
        img: Original image tensor [B, C, H, W]
        save_path: Path to save the visualization
        layer_idx: Index of layer to visualize (-1 for last layer)
        head_idx: Index of attention head to visualize (None for average across heads)
    """
    # Select layer
    attn = attention_weights[layer_idx]  # [B, H, N, N]
    
    # Convert image to numpy and normalize
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # Convert from [B, C, H, W] to [B, H, W, C]
        img = np.transpose(img, (0, 2, 3, 1))
        
        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0
    
    # Get batch size and image dimensions
    B, h, w, c = img.shape
    
    # Create figure
    fig, axes = plt.subplots(B, 1, figsize=(10, 5 * B))
    if B == 1:
        axes = [axes]
    
    for b in range(B):
        # Get attention for this batch item
        batch_attn = attn[b]  # [H, N, N]
        
        # Average across heads or select specific head
        if head_idx is not None:
            batch_attn = batch_attn[head_idx]  # [N, N]
        else:
            batch_attn = batch_attn.mean(0)  # [N, N]
        
        # Reshape to spatial dimensions [H, W, H, W]
        n = int(np.sqrt(batch_attn.shape[0]))
        batch_attn = batch_attn.reshape(n, n, n, n)
        
        # Average attention across queries to get attention per grid cell
        # Here we average across the second dimension to get [H, W]
        spatial_attn = batch_attn.mean(axis=(2, 3))
        
        # Resize to image dimensions (for better visualization)
        spatial_attn = cv2.resize(spatial_attn, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Plot
        ax = axes[b]
        im = ax.imshow(img[b])
        ax.imshow(spatial_attn, alpha=0.5, cmap='jet')
        ax.set_title(f"Transformer Attention (Layer {layer_idx}, {'Avg Heads' if head_idx is None else f'Head {head_idx}'})")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def visualize_decoder_attention(decoder_attention_maps, img, save_path=None):
    """
    Visualizes attention gate maps from the decoder overlaid on the original image.
    
    Args:
        decoder_attention_maps: List of attention maps from each decoder block
        img: Original image tensor [B, C, H, W]
        save_path: Path to save the visualization
    """
    # Get number of decoder attention maps
    num_maps = len(decoder_attention_maps)
    
    # Convert image to numpy and normalize
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # Convert from [B, C, H, W] to [B, H, W, C]
        img = np.transpose(img, (0, 2, 3, 1))
        
        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0
    
    # Get batch size
    B = img.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(B, num_maps, figsize=(4 * num_maps, 4 * B))
    if B == 1 and num_maps == 1:
        axes = np.array([[axes]])
    elif B == 1:
        axes = np.array([axes])
    elif num_maps == 1:
        axes = np.array([[ax] for ax in axes])
    
    for b in range(B):
        for i, attn_map in enumerate(decoder_attention_maps):
            if attn_map is None:
                continue
                
            # Get attention for this batch item
            attn = attn_map[b, 0]  # Assuming attention map shape is [B, 1, H, W]
            
            # Convert to numpy if it's a tensor
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu().numpy()
            
            # Resize to match image dimensions
            attn = cv2.resize(attn, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_LINEAR)
            
            # Plot
            ax = axes[b, i]
            ax.imshow(img[b])
            ax.imshow(attn, alpha=0.5, cmap='jet')
            ax.set_title(f"Decoder Attention Block {i+1}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_all_attention_maps(img, transformer_attn, decoder_attn, save_dir=None):
    """
    Visualizes both transformer and decoder attention maps.
    
    Args:
        img: Original image tensor [B, C, H, W]
        transformer_attn: List of attention weights from transformer
        decoder_attn: List of attention maps from decoder
        save_dir: Directory to save visualizations
    """
    # Visualize transformer attention for different layers and heads
    for layer_idx in range(len(transformer_attn)):
        # Average across all heads
        visualize_transformer_attention(
            transformer_attn, img, 
            save_path=f"{save_dir}/transformer_layer{layer_idx}_avg.png" if save_dir else None,
            layer_idx=layer_idx
        )
        
        # Visualize each head separately (for the last layer only)
        if layer_idx == len(transformer_attn) - 1:
            num_heads = transformer_attn[layer_idx].shape[1]  # Get number of heads
            for head_idx in range(num_heads):
                visualize_transformer_attention(
                    transformer_attn, img,
                    save_path=f"{save_dir}/transformer_layer{layer_idx}_head{head_idx}.png" if save_dir else None,
                    layer_idx=layer_idx, head_idx=head_idx
                )
    
    # Visualize decoder attention maps
    visualize_decoder_attention(
        decoder_attn, img,
        save_path=f"{save_dir}/decoder_attention.png" if save_dir else None
    )