import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        """
        Attention Gate for focusing on important features from the skip connection.
        
        Args:
            in_channels (int): Number of channels in the skip connection feature map.
            gating_channels (int): Number of channels in the decoder feature map.
            inter_channels (int): Number of intermediate channels for the attention mechanism.
        """
        super(AttentionGate, self).__init__()
        # Linear transformations for input (skip connection) and gating signal
        self.W_skip = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        # Attention map computation
        self.psi = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Activation
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, gating, return_attention_map=False):
        """
        Forward pass of the Attention Gate.
        
        Args:
            x (torch.Tensor): Skip connection feature map (from encoder).
            gating (torch.Tensor): Decoder feature map (lower-level features).
            return_attention_map (bool): Whether to return the attention map.
        
        Returns:
            torch.Tensor or tuple: Skip connection features weighted by the attention map,
                                   and optionally the attention map itself.
        """
        g1 = self.W_gating(gating)
        x1 = self.W_skip(x)
        psi = self.relu(g1 + x1)  # Combine gating and skip
        attention = self.psi(psi)  # Generate attention map
        
        # Apply attention to skip connection
        attended_x = x * attention
        
        if return_attention_map:
            return attended_x, attention
        return attended_x

class EnhancedAttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        """
        Enhanced Attention Gate with both spatial and channel attention.
        
        Args:
            in_channels (int): Number of channels in the skip connection feature map.
            gating_channels (int): Number of channels in the decoder feature map.
            inter_channels (int): Number of intermediate channels for the attention mechanism.
        """
        super(EnhancedAttentionGate, self).__init__()
        
        # Spatial attention components (from your original implementation)
        self.W_skip = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # Channel attention component (new)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=True),  # Dimension reduction
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=True),  # Dimension restoration
            nn.Sigmoid()  # Scale each channel
        )
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, gating):
        """
        Forward pass of the Enhanced Attention Gate.
        
        Args:
            x (torch.Tensor): Skip connection feature map (from encoder).
            gating (torch.Tensor): Decoder feature map (lower-level features).
        
        Returns:
            torch.Tensor: Skip connection features weighted by both spatial and channel attention.
        """
        # Spatial attention (from your original implementation)
        g1 = self.W_gating(gating)
        x1 = self.W_skip(x)
        psi = self.relu(g1 + x1)  # Combine gating and skip
        spatial_attn = self.psi(psi)  # Generate spatial attention map
        
        # Channel attention (new)
        channel_attn = self.channel_gate(x)  # Generate channel attention map
        
        # Combine both attention mechanisms
        return x * spatial_attn * channel_attn
    
class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_attention=True, use_batchnorm=True):
        super(DecoderBlockWithAttention, self).__init__()
        self.use_attention = use_attention

        # Attention gate for skip connection
        self.attention_gate = (
            AttentionGate(skip_channels, in_channels, inter_channels=skip_channels // 2)
            if self.use_attention and skip_channels > 0
            else None
        )

        # Compute the adjusted input channels for conv1
        adjusted_in_channels = in_channels + skip_channels

        # Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(adjusted_in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=False),
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None, return_attn=False):
        # Upsample the input
        x = self.up(x)

        # Align spatial size of x to skip, if needed
        if skip is not None and x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

        attention_map = None
        
        # Apply attention gate to skip connection, if enabled
        if skip is not None:
            if self.use_attention and self.attention_gate is not None:
                # Modify AttentionGate to return both attended features and attention map
                skip, attention_map = self.forward_attention(skip, x)
            x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension

        x = self.conv1(x)
        x = self.conv2(x)
        
        if return_attn:
            return x, attention_map
        return x
        
    def forward_attention(self, skip, x):
        """
        Forward the skip connection through the attention gate and return both
        the attended features and the attention map.
        """
        if not hasattr(self, 'attention_gate') or self.attention_gate is None:
            return skip, None
            
        # Get the input for attention computation
        g1 = self.attention_gate.W_gating(x)
        x1 = self.attention_gate.W_skip(skip)
        psi = self.attention_gate.relu(g1 + x1)
        
        # Generate attention map
        attention_map = self.attention_gate.psi(psi)
        
        # Apply attention to skip connection
        attended_skip = skip * attention_map
        
        return attended_skip, attention_map