"""
model.py
========
U-Net with ResNet-18 encoder for crater segmentation.

Author: Miriam Garcia Sollo
Date:   June 2026

Architecture (Ronneberger et al. 2015, encoder from He et al. 2016):

    Input (1, 128, 128)
        |
    [Encoder - ResNet-18 blocks]
        stem     -> (64,  64, 64)
        block1   -> (64,  64, 64)
        block2   -> (128, 32, 32)
        block3   -> (256, 16, 16)
        block4   -> (512,  8,  8)   <- bottleneck
        |
    [Decoder - 4 upsampling stages with skip connections]
        up4 + skip(block3) -> (256, 16, 16)
        up3 + skip(block2) -> (128, 32, 32)
        up2 + skip(block1) -> (64,  64, 64)
        up1 + skip(stem)   -> (32, 128, 128)
        |
    [Head]
        conv 1x1 -> (1, 128, 128) logits
        sigmoid  -> binary mask

Total parameters: ~14M with ResNet-18 encoder (pretrained or NFW-pretrained).
Lightweight variant (scratch): ~1.2M with custom 4-block encoder.

The model supports three initialisation modes:
    'scratch'   - random initialisation (Condition A baseline)
    'imagenet'  - ResNet-18 encoder pretrained on ImageNet (Condition B)
    'nfw'       - encoder weights loaded from NFW pretraining checkpoint (Condition C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two successive Conv-BN-ReLU blocks, the basic U-Net decoder unit."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + concatenate skip connection + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if skip and x sizes differ by 1 pixel (odd input sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Lightweight encoder (scratch baseline)
# ─────────────────────────────────────────────

class LightEncoder(nn.Module):
    """4-block CNN encoder for the scratch baseline (Condition A).

    Produces the same 5 feature maps as the ResNet-18 encoder
    so the decoder is reusable across all three conditions.

    Channels: 1 -> 32 -> 64 -> 128 -> 256 -> 512
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.stem   = DoubleConv(in_channels, 32)
        self.pool   = nn.MaxPool2d(2)
        self.block1 = DoubleConv(32, 64)
        self.block2 = DoubleConv(64, 128)
        self.block3 = DoubleConv(128, 256)
        self.block4 = DoubleConv(256, 512)

        # Channel counts for skip connections
        self.skip_channels = [32, 64, 128, 256]
        self.bottleneck_channels = 512

    def forward(self, x):
        s0 = self.stem(x)                        # (32,  128, 128)
        s1 = self.block1(self.pool(s0))          # (64,   64,  64)
        s2 = self.block2(self.pool(s1))          # (128,  32,  32)
        s3 = self.block3(self.pool(s2))          # (256,  16,  16)
        s4 = self.block4(self.pool(s3))          # (512,   8,   8)
        return s0, s1, s2, s3, s4


# ─────────────────────────────────────────────
# ResNet-18 encoder wrapper
# ─────────────────────────────────────────────

class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder adapted for single-channel DEM input.

    The first conv layer is modified from 3-channel to 1-channel input.
    When mode='imagenet', pretrained weights are loaded and the first
    conv weights are averaged across the RGB channels.

    Skip channels: [64, 64, 128, 256], bottleneck: 512
    (same interface as LightEncoder)
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        try:
            import torchvision.models as tvm
            resnet = tvm.resnet18(
                weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            import torchvision.models as tvm
            resnet = tvm.resnet18(pretrained=pretrained)

        # Adapt first conv: 3 channels -> 1 channel
        orig_conv = resnet.conv1
        new_conv  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average RGB weights to initialise single-channel conv
            with torch.no_grad():
                new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        # Extract layers for skip connections
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.block1 = resnet.layer1    # (64,  64, 64)  after pool
        self.block2 = resnet.layer2    # (128, 32, 32)
        self.block3 = resnet.layer3    # (256, 16, 16)
        self.block4 = resnet.layer4    # (512,  8,  8)

        self.skip_channels       = [64, 64, 128, 256]
        self.bottleneck_channels = 512

    def forward(self, x):
        s0 = self.stem(x)           # (64,  64, 64)
        p  = self.pool(s0)          # (64,  32, 32)  (only used as input to block1)
        s1 = self.block1(p)         # (64,  32, 32)
        s2 = self.block2(s1)        # (128, 16, 16)
        s3 = self.block3(s2)        # (256,  8,  8)
        s4 = self.block4(s3)        # (512,  4,  4)
        return s0, s1, s2, s3, s4


# ─────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────

class DarkNavUNet(nn.Module):
    """U-Net for crater segmentation with interchangeable encoder.

    Parameters
    ----------
    mode : str
        'scratch'  - LightEncoder with random init (Condition A)
        'imagenet' - ResNet-18 encoder, ImageNet pretrained (Condition B)
        'nfw'      - ResNet-18 encoder, loaded from NFW checkpoint (Condition C)
    nfw_checkpoint : str or Path, optional
        Path to .pth checkpoint from NFW pretraining. Required when mode='nfw'.

    Usage
    -----
        model = DarkNavUNet(mode='scratch')
        logits = model(x)           # (B, 1, H, W), raw logits
        probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        mode: str = 'scratch',
        nfw_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        assert mode in ('scratch', 'imagenet', 'nfw'), \
            f"mode must be 'scratch', 'imagenet', or 'nfw', got '{mode}'"

        # Build encoder
        if mode == 'scratch':
            self.encoder = LightEncoder(in_channels=1)
        elif mode == 'imagenet':
            self.encoder = ResNet18Encoder(pretrained=True)
        else:  # nfw
            self.encoder = ResNet18Encoder(pretrained=False)

        sc = self.encoder.skip_channels        # [s0, s1, s2, s3]
        bc = self.encoder.bottleneck_channels  # 512

        # Decoder: 4 upsampling stages
        self.up4 = UpBlock(bc,     sc[3], 256)
        self.up3 = UpBlock(256,    sc[2], 128)
        self.up2 = UpBlock(128,    sc[1], 64)
        self.up1 = UpBlock(64,     sc[0], 32)

        # Final upsample to input resolution + classification head
        self.final_up   = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.head       = nn.Conv2d(32, 1, kernel_size=1)

        # Load NFW checkpoint if requested
        if mode == 'nfw':
            if nfw_checkpoint is None:
                raise ValueError(
                    "nfw_checkpoint path is required when mode='nfw'. "
                    "Run the NFW pretraining phase first."
                )
            self._load_nfw_checkpoint(nfw_checkpoint)

        self.mode = mode

    def _load_nfw_checkpoint(self, checkpoint_path: str) -> None:
        """Load encoder weights from an NFW pretraining checkpoint.

        The checkpoint may contain full model state_dict or encoder-only weights.
        Only encoder weights are transferred; decoder is randomly initialised.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"NFW checkpoint not found: {path}")

        ckpt = torch.load(path, map_location='cpu')

        # Handle both full model checkpoints and encoder-only checkpoints
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'encoder_state_dict' in ckpt:
            state = ckpt['encoder_state_dict']
        else:
            state = ckpt

        # Filter to encoder keys only
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in state.items()
            if k.startswith('encoder.')
        }

        if len(encoder_state) == 0:
            # Checkpoint has flat keys (encoder trained standalone)
            encoder_state = state

        missing, unexpected = self.encoder.load_state_dict(
            encoder_state, strict=False
        )
        print(f"NFW checkpoint loaded from {path.name}")
        if missing:
            print(f"  Missing keys  : {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, H, W), float32, values in [-1, 1]

        Returns
        -------
        logits : torch.Tensor, shape (B, 1, H, W)
                 Raw logits. Apply torch.sigmoid for probabilities.
                 Use BCEWithLogitsLoss during training (numerically stable).
        """
        s0, s1, s2, s3, s4 = self.encoder(x)

        d = self.up4(s4, s3)    # (256, 16, 16)
        d = self.up3(d,  s2)    # (128, 32, 32)
        d = self.up2(d,  s1)    # (64,  64, 64)
        d = self.up1(d,  s0)    # (32, 128, 128)  for ResNet, s0 is (64, 64)

        d = self.final_up(d)    # (32, 256, 256) or (32, 128, 128)

        # Ensure output matches input spatial size
        if d.shape[2:] != x.shape[2:]:
            d = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.head(d)     # (B, 1, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Dimension verification utility
# ─────────────────────────────────────────────

def verify_model_dimensions(mode: str = 'scratch', verbose: bool = True) -> dict:
    """Instantiate the model and run a test forward pass to verify shapes.

    Parameters
    ----------
    mode : 'scratch', 'imagenet', or 'nfw' (nfw requires a checkpoint)
    verbose : print shape at each stage

    Returns
    -------
    info : dict with parameter count, input/output shapes, RAM estimate
    """
    if mode == 'nfw':
        print("Skipping nfw mode in verification (requires checkpoint).")
        mode = 'scratch'

    model = DarkNavUNet(mode=mode)
    model.eval()

    x = torch.zeros(1, 1, 128, 128)

    if verbose:
        print(f"Model mode     : {mode}")
        print(f"Parameters     : {model.count_parameters():,}")
        print(f"Input shape    : {tuple(x.shape)}")

    with torch.no_grad():
        # Trace encoder
        s0, s1, s2, s3, s4 = model.encoder(x)
        if verbose:
            print(f"Encoder output shapes:")
            print(f"  s0 (stem)   : {tuple(s0.shape)}")
            print(f"  s1 (block1) : {tuple(s1.shape)}")
            print(f"  s2 (block2) : {tuple(s2.shape)}")
            print(f"  s3 (block3) : {tuple(s3.shape)}")
            print(f"  s4 (bottleneck): {tuple(s4.shape)}")

        logits = model(x)
        probs  = torch.sigmoid(logits)

        if verbose:
            print(f"Output logits  : {tuple(logits.shape)}")
            print(f"Output probs   : {tuple(probs.shape)}")
            print(f"Logits range   : [{logits.min():.3f}, {logits.max():.3f}]")

    assert logits.shape == (1, 1, 128, 128), \
        f"Output shape mismatch: expected (1,1,128,128), got {tuple(logits.shape)}"

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ram_mb = param_bytes / 1e6

    if verbose:
        print(f"Model RAM      : {ram_mb:.1f} MB")
        print("Shape verification PASSED.")

    return {
        'mode':       mode,
        'parameters': model.count_parameters(),
        'output_shape': tuple(logits.shape),
        'ram_mb':     ram_mb,
    }


if __name__ == '__main__':
    print("=== DarkNavUNet dimension verification ===\n")
    for m in ['scratch', 'imagenet']:
        print(f"--- mode: {m} ---")
        info = verify_model_dimensions(mode=m, verbose=True)
        print()
