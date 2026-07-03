"""MViT-V2-S backbone for the agaldran challenge model (ODV-214).

Vendored from the ODELIA challenge submission and trimmed to the single
architecture this subunit serves (``mvit_v2_s``). Kept faithful to the original
construction so the trained state_dict layout is preserved.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s

from ..base import BasicClassifier, ModelWrapper


def load_mvit_v2_s(
    pretrained_path: str | None = None,
    num_classes: int = 3,
    in_ch: int = 3,  # Pre, Sub1, T2
    target_frames: int = 16,
    target_hw: int = 224,
) -> nn.Module:
    """
    Offline-safe MViT-V2-S wrapper with
      • channel-agnostic stem (in_ch = 1, 3, …)
      • depth-wise stride-4 blur Conv3d to compress 64→target_frames
      • spatial up-sample to target_hw
      • fresh Linear head (num_classes)
    """

    # -------- 1) backbone skeleton ---------------------------------------
    core = mvit_v2_s(weights=None)  # never auto-downloads

    # -------- 2) optional checkpoint -------------------------------------
    if pretrained_path:
        ckpt = (
            pretrained_path
            if os.path.isfile(pretrained_path)
            else os.path.join(
                pretrained_path,
                next(
                    f
                    for f in os.listdir(pretrained_path)
                    if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))
                ),
            )
        )
        print("🚀 loading MViT-V2 weights from", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        core.load_state_dict(state, strict=False)
    else:
        print("⚙️  model initialised without pretrained weights")

    # -------- 3) patch conv_proj for in_ch --------------------------------
    proj = core.conv_proj  # (96,3,3,7,7)
    if in_ch != proj.in_channels:
        w = proj.weight  # rgb weights
        if in_ch == 1:  # RGB → mono
            new_w = w.mean(1, keepdim=True)
        elif in_ch > 3:  # replicate & scale
            reps = (in_ch + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:  # 2-channel
            new_w = w[:, :in_ch]

        new_proj = nn.Conv3d(
            in_ch,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False,
        )
        new_proj.weight = nn.Parameter(new_w.clone())
        core.conv_proj = new_proj
        print(f"✓ patched conv_proj → {in_ch}-channel")

    # -------- 4) replace classifier head ----------------------------------
    emb_dim = core.head[1].in_features
    core.head[1] = nn.Linear(emb_dim, num_classes)

    # -------- 5) wrapper with temporal blur & upsample --------------------
    class Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            stride_t = math.ceil(64 / target_frames)
            self.reduce = nn.Conv3d(
                in_ch,
                in_ch,
                kernel_size=(5, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(2, 0, 0),
                groups=in_ch,
                bias=False,
            )
            # blur kernel [1 2 4 2 1] / 10 replicated per channel
            with torch.no_grad():
                k = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32) / 10.0
                k = k.view(1, 1, 5, 1, 1).repeat(in_ch, 1, 1, 1, 1)
                self.reduce.weight.copy_(k)
            self.core = backbone

        def forward(self, x):  # B×C×64×128×128
            x = self.reduce(x)  # B×C×16×128×128
            x = F.interpolate(
                x,
                size=(target_frames, target_hw, target_hw),
                mode="trilinear",
                align_corners=False,
            )
            return self.core(x)  # logits

    print(f"✓ temporal blur 64→{target_frames} & spatial ↑ → {target_hw}²")
    return Wrapper(core)


VIDEO_BACKBONES = {
    "mvit_v2_s": load_mvit_v2_s,
}


def set_global_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------- Factory -----------------------
def model_factory(
    arch: str,
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 3,
    freeze_backbone: bool = False,
    seed: int | None = 42,
    **classifier_kwargs,
) -> BasicClassifier:  # TODO adaption: previous -> nn.Module:
    """
    Builds and returns a video classification model for 3D medical volumes.

    Args:
        arch: Key of the model in the registry.
        pretrained_path: Local path to pretrained weights.
        num_classes: Number of output classes.
        freeze: If True, freezes all model parameters.
        seed: Random seed for reproducibility.
    """
    set_global_seed(seed)
    arch = arch.lower()
    if arch not in VIDEO_BACKBONES:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(VIDEO_BACKBONES.keys())}"
        )

    model = VIDEO_BACKBONES[arch](
        pretrained_path=pretrained_path, num_classes=num_classes, in_ch=in_ch
    )

    def freeze_backbone_only(model: nn.Module) -> None:
        # 1) freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # 2) un-freeze the *last* nn.Linear encountered
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    p.requires_grad = True
                break

    if freeze_backbone:
        freeze_backbone_only(model)

    # if freeze_backbone:
    #     for name, param in model.named_parameters():
    #         if not name.startswith("classifier"):
    #             param.requires_grad = False

    # TODO adaption: added:
    return ModelWrapper(backbone=model, in_ch=in_ch, num_classes=num_classes, **classifier_kwargs)
