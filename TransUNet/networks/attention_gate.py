import torch
import torch.nn as nn

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
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gating):
        """
        Forward pass of the Attention Gate.
        
        Args:
            x (torch.Tensor): Skip connection feature map (from encoder).
            gating (torch.Tensor): Decoder feature map (lower-level features).
        
        Returns:
            torch.Tensor: Skip connection features weighted by the attention map.
        """
        g1 = self.W_gating(gating)
        x1 = self.W_skip(x)
        psi = self.relu(g1 + x1)  # Combine gating and skip
        attention = self.psi(psi)  # Generate attention map
        return x * attention  # Apply attention to skip connection


class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_attention=True, use_batchnorm=True):
        """
        Decoder block with optional Attention Gate.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            skip_channels (int): Number of channels in the skip connection.
            use_attention (bool): Whether to use Attention Gate in the decoder block.
            use_batchnorm (bool): Whether to use BatchNorm in convolution layers.
        """
        super(DecoderBlockWithAttention, self).__init__()
        self.use_attention = use_attention
        if self.use_attention and skip_channels > 0:
            self.attention_gate = AttentionGate(skip_channels, in_channels, inter_channels=skip_channels // 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + (skip_channels if not use_attention else 0), out_channels,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        """
        Forward pass for the decoder block.

        Args:
            x (torch.Tensor): Input feature map from the decoder.
            skip (torch.Tensor, optional): Skip connection feature map. Defaults to None.

        Returns:
            torch.Tensor: Output feature map after upsampling and attention application.
        """
        x = self.up(x)
        if skip is not None:
            if self.use_attention:
                skip = self.attention_gate(skip, x)  # Apply Attention Gate to skip connection
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x