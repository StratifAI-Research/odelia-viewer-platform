"""Multi-slice transformer (MST) classifier with a DINOv2 backbone."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from x_transformers import Encoder

from .base import BasicClassifier


class _TransformerEncoder(Encoder):
    """x-transformers Encoder with MediSwarm's mask conventions."""

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(
            x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask
        )


class _MST(nn.Module):
    """Per-slice DINOv2 features fused across slices for volume classification."""

    def __init__(
        self,
        out_ch: int = 1,
        backbone_type: str = "dinov2",
        model_size: str | None = None,
        slice_fusion_type: str = "transformer",
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type != "dinov2":
            raise ValueError(f"Unknown backbone_type: {backbone_type!r}")
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{model_size}14")
        self.backbone.mask_token = None
        emb_ch = self.backbone.num_features
        self.emb_ch = emb_ch

        if slice_fusion_type == "transformer":
            self.slice_fusion = _TransformerEncoder(
                dim=emb_ch,
                heads=12 if emb_ch % 12 == 0 else 8,
                ff_mult=1,
                attn_dropout=0.0,
                pre_norm=True,
                depth=1,
                attn_flash=True,
                ff_no_bias=True,
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type in ("average", "none"):
            self.slice_fusion = None
        else:
            raise ValueError(f"Unknown slice_fusion_type: {slice_fusion_type!r}")

        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = rearrange(x, "b c d h w -> (b c d) h w")
        x = x[:, None].repeat(1, 3, 1, 1)  # gray -> RGB

        x = self.backbone(x)  # (B * D, E)
        x = rearrange(x, "(b d) e -> b d e", b=b)

        if self.slice_fusion_type == "none":
            return x
        if self.slice_fusion_type == "transformer":
            x = torch.cat([x, self.cls_token.repeat(b, 1, 1)], dim=1)
            x = self.slice_fusion(x)
        elif self.slice_fusion_type == "average":
            x = x.mean(dim=1, keepdim=True)

        return self.linear(x[:, -1])


class MST(BasicClassifier):
    """MST classifier: DINOv2 per-slice encoder + transformer slice fusion."""

    def __init__(
        self,
        n_input_channels: int,
        num_classes: int,
        spatial_dims: int,
        backbone_type: str = "dinov2",
        model_size: str = "s",
        slice_fusion_type: str = "transformer",
        loss_kwargs: dict | None = None,
    ) -> None:
        super().__init__(n_input_channels, num_classes, spatial_dims, loss_kwargs=loss_kwargs)
        self.mst = _MST(
            out_ch=num_classes,
            backbone_type=backbone_type,
            model_size=model_size,
            slice_fusion_type=slice_fusion_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mst(x)
