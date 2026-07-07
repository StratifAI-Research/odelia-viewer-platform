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
