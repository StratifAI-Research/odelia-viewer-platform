"""Inference base for the ODELIA built-in classifiers.

Trained ODELIA checkpoints come from MediSwarm's Lightning classifiers, but
serving only needs the module structure and a forward pass. This base keeps
exactly the ``state_dict``-affecting pieces — the backbone that subclasses
register (``self.model`` / ``self.mst``) and an optional ``_class_weight``
buffer so checkpoints trained with class weights load strictly (ODV-216) — and
drops the training / optimizer / metric machinery.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicClassifier(nn.Module):
    """Base for the built-in classifiers.

    Subclasses register their network and implement :meth:`forward`. When
    ``loss_kwargs`` carries a ``weight``, it is stored as the ``_class_weight``
    buffer so it matches state_dicts trained with class weights.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int,
        loss_kwargs: dict | None = None,
        **_kwargs: object,
    ) -> None:
        # ``**_kwargs`` swallows the training-only args (optimizer/scheduler/metric
        # kwargs) that the ported models still pass; serving ignores them.
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        weight = (loss_kwargs or {}).get("weight")
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("_class_weight", weight)
        else:
            self._class_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ModelWrapper(BasicClassifier):
    """Wrap an arbitrary backbone as a classifier.

    The backbone is registered as ``self.backbone`` (→ ``backbone.*`` state_dict
    keys), matching the challenge models trained via this wrapper.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_ch: int,
        num_classes: int,
        spatial_dims: int = 3,
        loss_kwargs: dict | None = None,
    ) -> None:
        super().__init__(in_ch, num_classes, spatial_dims, loss_kwargs=loss_kwargs)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
