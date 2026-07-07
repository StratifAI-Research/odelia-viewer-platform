"""MediSwarm inference transforms, ported (numerics preserved, types added).

Only ``ZNormalization`` is genuinely custom: it clips to a percentile band
before z-normalizing (masked). At inference all augmentation is off and
``CropOrPad``'s random-center branch is unused, so stock ``tio.CropOrPad``
suffices elsewhere and is not re-ported.
"""

from __future__ import annotations

import torch
import torchio as tio
from torchio import Subject
from torchio.transforms.transform import TypeMaskingMethod


def parse_per_channel(
    per_channel: bool | list[tuple[int, ...]], channels: int
) -> list[tuple[int, ...]]:
    """Expand a bool into per-group index tuples, or pass an explicit list through."""
    if isinstance(per_channel, bool):
        if per_channel:
            return [(ch,) for ch in range(channels)]
        return [tuple(ch for ch in range(channels))]
    return per_channel


class ZNormalization(tio.ZNormalization):
    """Z-normalization with per-channel/per-slice grouping and percentile clipping."""

    def __init__(
        self,
        percentiles: tuple[float, float] = (0.0, 100.0),
        per_channel: bool | list[tuple[int, ...]] = True,
        per_slice: bool | list[tuple[int, ...]] = False,
        masking_method: TypeMaskingMethod = None,
        **kwargs: object,
    ) -> None:
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel
        self.per_slice = per_slice

    def apply_normalization(
        self, subject: Subject, image_name: str, mask: torch.Tensor
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        per_slice = parse_per_channel(self.per_slice, image.shape[-1])

        image.set_data(
            torch.cat(
                [
                    torch.cat(
                        [
                            self._znorm(
                                image.data[chs,][:, :, :, sl,],
                                mask[chs,][:, :, :, sl,],
                                image_name,
                                image.path,
                            )
                            for sl in per_slice
                        ],
                        dim=-1,
                    )
                    for chs in per_channel
                ]
            )
        )

    def _znorm(
        self,
        image_data: torch.Tensor,
        mask: torch.Tensor,
        image_name: str,
        image_path: object,
    ) -> torch.Tensor:
        cutoff = torch.quantile(
            image_data.masked_select(mask).float(),
            torch.tensor(self.percentiles) / 100.0,
        )
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        standardized = self.znorm(image_data, mask)
        if standardized is None:
            raise RuntimeError(
                f'Standard deviation is 0 for masked values in image "{image_name}" ({image_path})'
            )
        return standardized
