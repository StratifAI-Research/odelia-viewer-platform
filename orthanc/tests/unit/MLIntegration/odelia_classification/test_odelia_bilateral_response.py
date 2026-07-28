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
