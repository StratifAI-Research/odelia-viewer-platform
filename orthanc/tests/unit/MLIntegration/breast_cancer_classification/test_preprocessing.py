"""Tests for breast-cancer-classification/preprocessing.py.

preprocessing.py imports torchio and torch at module level and subclasses
tio.ZNormalization and tio.CropOrPad at class-definition time.
Both torchio_stub and torch_stub fixtures (from the parent conftest) must be
active before importing the module.

Uncovered paths (intentional):
- ZNormalization.apply_normalization / _znorm: require real torch tensor
  arithmetic (quantile, clamp, znorm).  Fully testing these needs real torchio.
- preprocess_for_side: calls torch.cat on real TorchIO Image objects and moves
  tensor to device — exercising that would require real torch.  The function
  signature and argument passing are validated indirectly via module importability.
"""
import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _evict_preprocessing() -> Iterator[None]:
    """Ensure preprocessing is re-imported fresh with stubs active."""
    sys.modules.pop('preprocessing', None)
    yield
    sys.modules.pop('preprocessing', None)


def test_preprocessing_imports_with_stubs(torch_stub, torchio_stub):
    pass  # should not raise


def test_parse_per_channel_per_channel_true(torch_stub, torchio_stub):
    import preprocessing
    result = preprocessing.parse_per_channel(True, 3)
    assert result == [(0,), (1,), (2,)]


def test_parse_per_channel_per_channel_false(torch_stub, torchio_stub):
    import preprocessing
    result = preprocessing.parse_per_channel(False, 3)
    assert result == [(0, 1, 2)]


def test_parse_per_channel_single_channel(torch_stub, torchio_stub):
    import preprocessing
    result = preprocessing.parse_per_channel(True, 1)
    assert result == [(0,)]


def test_image_to_tensor_class_exists(torch_stub, torchio_stub):
    import preprocessing
    assert hasattr(preprocessing, 'ImageToTensor')
    assert callable(preprocessing.ImageToTensor)


def test_znormalization_class_exists(torch_stub, torchio_stub):
    import preprocessing
    assert hasattr(preprocessing, 'ZNormalization')


def test_random_crop_or_pad_class_exists(torch_stub, torchio_stub):
    import preprocessing
    assert hasattr(preprocessing, 'RandomCropOrPad')


def test_get_preprocessing_pipeline_returns_compose(torch_stub, torchio_stub):
    """get_preprocessing_pipeline should return a tio.Compose instance."""
    import preprocessing
    import torchio as tio
    pipeline = preprocessing.get_preprocessing_pipeline()
    assert isinstance(pipeline, tio.Compose)


def test_get_preprocessing_pipeline_has_three_transforms(torch_stub, torchio_stub):
    """Pipeline should have exactly 3 transforms: CropOrPad, ZNormalization, ImageToTensor."""
    import preprocessing
    pipeline = preprocessing.get_preprocessing_pipeline()
    assert len(pipeline.transforms) == 3


def test_get_preprocessing_pipeline_first_is_crop_or_pad(torch_stub, torchio_stub):
    import preprocessing
    pipeline = preprocessing.get_preprocessing_pipeline()
    assert isinstance(pipeline.transforms[0], preprocessing.RandomCropOrPad)


def test_get_preprocessing_pipeline_second_is_znormalization(torch_stub, torchio_stub):
    import preprocessing
    pipeline = preprocessing.get_preprocessing_pipeline()
    assert isinstance(pipeline.transforms[1], preprocessing.ZNormalization)


def test_get_preprocessing_pipeline_third_is_image_to_tensor(torch_stub, torchio_stub):
    import preprocessing
    pipeline = preprocessing.get_preprocessing_pipeline()
    assert isinstance(pipeline.transforms[2], preprocessing.ImageToTensor)


def test_znormalization_percentiles_stored(torch_stub, torchio_stub):
    import preprocessing
    zn = preprocessing.ZNormalization(percentiles=(0.5, 99.5), per_channel=True)
    assert zn.percentiles == (0.5, 99.5)
    assert zn.per_channel is True


def test_preprocess_for_side_function_exists(torch_stub, torchio_stub):
    import preprocessing
    assert callable(preprocessing.preprocess_for_side)


# ---------------------------------------------------------------------------
# B1, B2, B3 — REAL torch + torchio (no _stub fixtures used in these tests)
# ---------------------------------------------------------------------------

def test_znormalization_raises_on_constant_data_zero_std():
    """When masked data has zero variance, _znorm raises RuntimeError (std=0)."""
    import torch
    import preprocessing
    z = preprocessing.ZNormalization(
        per_channel=False, percentiles=(0, 100), masking_method=lambda x: x > 0,
    )
    data = torch.ones(2, 2, 2) * 5.0  # constant masked region
    mask = torch.ones(2, 2, 2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="Standard deviation is 0"):
        z._znorm(data, mask, "img", "/path")


def test_znormalization_clips_outlier_before_znorm():
    """Compare clipped vs unclipped: percentile clipping must reduce outlier z-score by >100x."""
    import torch
    import preprocessing
    z_clip = preprocessing.ZNormalization(
        per_channel=False, percentiles=(0.5, 99.5), masking_method=lambda x: x > 0,
    )
    z_noclip = preprocessing.ZNormalization(
        per_channel=False, percentiles=(0, 100), masking_method=lambda x: x > 0,
    )
    # Outlier 1e9 vs bulk in [1, 199] — without clip the outlier dominates the z-score.
    # 2000 bulk values + 1 outlier so the 99.5 percentile lands firmly in the bulk,
    # not in the interpolation zone between the last bulk value and the outlier.
    data_clip = torch.arange(1, 2000, dtype=torch.float32).reshape(1, 1999, 1).clone()
    data_clip[0, 0, 0] = 1e9
    data_noclip = data_clip.clone()
    mask = torch.ones_like(data_clip, dtype=torch.bool)

    out_clip = z_clip._znorm(data_clip, mask, "img", "/path")
    out_noclip = z_noclip._znorm(data_noclip, mask, "img", "/path")

    assert torch.isfinite(out_clip).all()
    assert torch.isfinite(out_noclip).all()
    # Clipped: outlier is brought into the bulk so max-z stays small (<5).
    # Unclipped: outlier widens std but its own z-score still grows beyond 10.
    assert out_clip.abs().max().item() < 5.0
    assert out_noclip.abs().max().item() > 10.0


def test_get_preprocessing_pipeline_crop_or_pad_target_shape_is_256_256_32():
    """First pipeline transform pins (256, 256, 32) target shape — guards against silent dim regressions."""
    import preprocessing
    pipeline = preprocessing.get_preprocessing_pipeline()
    first = pipeline.transforms[0]
    assert isinstance(first, preprocessing.RandomCropOrPad)
    assert first.target_shape == (256, 256, 32)


def test_preprocess_for_side_returns_5d_cpu_tensor():
    """preprocess_for_side concatenates pre+post along dim=0, applies pipeline, adds batch dim -> 5D."""
    import torch
    import torchio as tio
    import preprocessing
    # ZNormalization uses mask = lambda x: x > 0 — use positive data so the mask is non-empty.
    torch.manual_seed(0)
    pre = tio.ScalarImage(tensor=torch.rand(1, 64, 64, 16) + 0.1)
    post = tio.ScalarImage(tensor=torch.rand(1, 64, 64, 16) + 0.1)
    pipeline = preprocessing.get_preprocessing_pipeline()
    result = preprocessing.preprocess_for_side(pre, post, pipeline, "cpu")
    assert result.dim() == 5
    assert result.shape[0] == 1  # batch dim added via [None]
    assert result.device.type == "cpu"
