"""ODV-217 unit tests for the MediSwarm preprocessing package.

Under tests/unit/MLIntegration/odelia_classification/ so the autouse
_force_odelia_path fixture resolves imports to odelia-classification. Uses real
torch/torchio (like the ODV-214 build+forward test).
"""

import torch


class TestVolumeView:
    def test_holds_label_and_tensor(self):
        from preprocessing.types import VolumeView

        t = torch.zeros(1, 1, 32, 224, 224)
        view = VolumeView(label="left", tensor=t)
        assert view.label == "left"
        assert view.tensor.shape == (1, 1, 32, 224, 224)

    def test_is_frozen(self):
        import dataclasses

        import pytest

        from preprocessing.types import VolumeView

        view = VolumeView(label="right", tensor=torch.zeros(1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.label = "left"


class TestZNormalization:
    def _subject(self, tensor):
        import torchio as tio

        return tio.Subject(img=tio.ScalarImage(tensor=tensor))

    def test_clips_outlier_to_percentile_then_standardizes(self):
        import torchio as tio

        from preprocessing.transforms import ZNormalization

        # A ramp 1..1000 plus a huge outlier; min/max are excluded by the mask.
        data = torch.arange(1, 1001, dtype=torch.float32).reshape(1, 10, 10, 10).clone()
        data[0, 0, 0, 0] = 100000.0  # outlier -> must be clipped, not survive
        subject = self._subject(data)

        transform = ZNormalization(
            percentiles=(0.5, 99.5),
            per_channel=True,
            per_slice=False,
            masking_method=lambda x: (x > x.min()) & (x < x.max()),
        )
        out = transform(subject).img.data

        # Outlier was clipped to <= the 99.5th percentile (nowhere near 100000).
        assert out.max().item() < 10.0
        # Masked region is standardized: ~0 mean, ~1 std.
        mask = (data > data.min()) & (data < data.max())
        vals = out.masked_select(mask)
        assert abs(vals.mean().item()) < 0.1
        assert abs(vals.std().item() - 1.0) < 0.1

    def test_parse_per_channel_bool(self):
        from preprocessing.transforms import parse_per_channel

        assert parse_per_channel(True, 3) == [(0,), (1,), (2,)]
        assert parse_per_channel(False, 3) == [(0, 1, 2)]


class TestHelpers:
    def test_image_to_tensor_swaps_axes(self):
        import torchio as tio

        from preprocessing.transforms import image_to_tensor

        # data is [C, W, H, D]; swapaxes(1, -1) -> [C, D, H, W]
        img = tio.ScalarImage(tensor=torch.zeros(1, 5, 6, 7))
        out = image_to_tensor(img)
        assert out.shape == (1, 7, 6, 5)

    def test_crop_breast_height_returns_height_256_crop(self):
        import torchio as tio

        from preprocessing.transforms import crop_breast_height

        img = tio.ScalarImage(tensor=torch.ones(1, 512, 512, 32))
        crop = crop_breast_height(img)
        assert isinstance(crop, tio.Crop)
        cropped = crop(img).data
        # Height axis (index 2) is cropped to 256.
        assert cropped.shape[2] == 256


class TestPreprocessPipeline:
    def _write_sub_nifti(self, tmp_path):
        import nibabel as nib
        import numpy as np

        # A small anisotropic volume; values give a non-trivial foreground.
        arr = (np.random.default_rng(0).random((64, 64, 16)) * 1000).astype("float32")
        path = tmp_path / "sub.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))
        return path

    def test_yields_two_labelled_views_of_correct_shape(self, tmp_path):
        from preprocessing.pipeline import preprocess

        views = preprocess(self._write_sub_nifti(tmp_path), device="cpu")
        assert [v.label for v in views] == ["left", "right"]
        for v in views:
            assert v.tensor.shape == (1, 1, 32, 224, 224)

    def test_deterministic(self, tmp_path):
        from preprocessing.pipeline import preprocess

        sub = self._write_sub_nifti(tmp_path)
        a = preprocess(sub, device="cpu")
        b = preprocess(sub, device="cpu")
        for va, vb in zip(a, b):
            assert torch.equal(va.tensor, vb.tensor)


class TestDispatch:
    def test_defaults_to_mediswarm_pipeline(self):
        from preprocessing import resolve_preprocessor
        from preprocessing.pipeline import preprocess as default_preprocess

        # An unknown model has no override -> default.
        assert resolve_preprocessor("Pimed") is default_preprocess

    def test_model_local_override_wins(self, monkeypatch):
        import types

        import preprocessing.dispatch as dispatch

        sentinel = lambda sub, device: []  # noqa: E731
        fake = types.ModuleType("models.pimed.preprocess")
        fake.preprocess = sentinel

        monkeypatch.setattr(dispatch, "available_models", lambda: {"Pimed": "pimed"})
        monkeypatch.setitem(__import__("sys").modules, "models.pimed.preprocess", fake)

        assert dispatch.resolve_preprocessor("Pimed") is sentinel
