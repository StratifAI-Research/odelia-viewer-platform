"""Unit tests for the router-facing bilateral response shape.

Under tests/unit/MLIntegration/odelia_classification/ so the autouse
_force_odelia_path fixture puts the service dir at sys.path[0]; service modules
are imported inside each test.
"""

import pytest

_INFO = {
    "model_name": "Pimed",
    "architecture": "ResNet",
    "version": "0.1.0",
    "num_classes": 3,
    "device": "cpu",
    "weights": "init-only",
}

_RESULTS = [
    ("left", [0.1, 0.7, 0.2]),
    ("right", [0.8, 0.1, 0.1]),
]


class TestBilateralResponse:
    def test_emits_left_and_right_keys(self):
        from response_builder import build_bilateral_response

        resp = build_bilateral_response(_RESULTS, _INFO)
        assert resp["left"]["prediction"] == "Benign"
        assert resp["right"]["prediction"] == "No lesion"

    def test_confidence_is_a_percentage(self):
        from response_builder import build_bilateral_response

        resp = build_bilateral_response(_RESULTS, _INFO)
        assert resp["left"]["confidence"] == pytest.approx(70.0)
        assert resp["right"]["confidence"] == pytest.approx(80.0)

    @pytest.mark.parametrize(
        ("probs", "expected"),
        [
            ([0.9, 0.05, 0.05], "No lesion"),
            ([0.05, 0.9, 0.05], "Benign"),
            ([0.05, 0.05, 0.9], "Malignant"),
        ],
        ids=["no-lesion", "benign", "malignant"],
    )
    def test_every_class_index_maps_to_its_name(self, probs, expected):
        from response_builder import build_bilateral_response

        resp = build_bilateral_response([("left", probs), ("right", probs)], _INFO)
        assert resp["left"]["prediction"] == expected

    def test_views_payload_is_preserved(self):
        from response_builder import build_bilateral_response

        resp = build_bilateral_response(_RESULTS, _INFO)
        assert [v["label"] for v in resp["views"]] == ["left", "right"]
        assert resp["views"][0]["probabilities"] == [0.1, 0.7, 0.2]

    def test_model_metadata_carries_identity_only(self):
        """Weight provenance must not reach the SR via model_metadata."""
        from response_builder import build_bilateral_response

        resp = build_bilateral_response(_RESULTS, _INFO)
        assert resp["model_metadata"] == {
            "model_name": "Pimed",
            "architecture": "ResNet",
            "version": "0.1.0",
        }

    def test_missing_model_info_does_not_raise(self):
        from response_builder import build_bilateral_response

        resp = build_bilateral_response(_RESULTS, None)
        assert resp["model_metadata"]["model_name"] == "ODELIA"
        assert resp["left"]["prediction"] == "Benign"


class TestInferAndRespondShape:
    """_infer_and_respond picks its response shape from the view labels."""

    def _service(self, monkeypatch, labels):
        from types import SimpleNamespace

        from model_service import ModelService
        from preprocessing.types import VolumeView

        svc = ModelService.__new__(ModelService)
        svc.config = SimpleNamespace(model_name="Pimed", device="cpu")
        svc.model_info = _INFO

        views = [VolumeView(label=label, tensor=object()) for label in labels]
        monkeypatch.setattr(
            "model_service.resolve_preprocessor",
            lambda name: lambda path, device: views,
        )
        monkeypatch.setattr(
            ModelService, "_run_inference", lambda self, tensor: [0.1, 0.7, 0.2]
        )
        return svc

    def test_left_right_labels_produce_bilateral_shape(self, monkeypatch):
        from pathlib import Path

        svc = self._service(monkeypatch, ["left", "right"])
        resp = svc._infer_and_respond(Path("/tmp/sub.nii.gz"))

        assert resp["left"]["prediction"] == "Benign"
        assert resp["right"]["prediction"] == "Benign"
        assert resp["model_metadata"]["model_name"] == "Pimed"

    def test_non_bilateral_labels_keep_views_shape(self, monkeypatch):
        from pathlib import Path

        svc = self._service(monkeypatch, ["volume"])
        resp = svc._infer_and_respond(Path("/tmp/sub.nii.gz"))

        assert "left" not in resp
        assert "model_metadata" not in resp
        assert [v["label"] for v in resp["views"]] == ["volume"]

    def test_empty_views_still_raise(self, monkeypatch):
        from pathlib import Path

        from exceptions import InferenceError

        svc = self._service(monkeypatch, [])
        with pytest.raises(InferenceError, match="no views"):
            svc._infer_and_respond(Path("/tmp/sub.nii.gz"))
