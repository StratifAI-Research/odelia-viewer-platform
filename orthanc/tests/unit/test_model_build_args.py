"""Run the Dockerfile's argument guards in POSIX sh before any model tooling."""

import os
import subprocess
from pathlib import Path

import pytest


def weight_step():
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "MLIntegration/odelia-classification/Dockerfile"
    ).read_text()
    # Join Docker continuations, then execute the actual RUN body with a harmless
    # Python stand-in. No downloads, model imports or Docker daemon are needed.
    logical_lines = dockerfile.replace("\\\n", " ").splitlines()
    prefix = "RUN --mount=type=secret,id=hf_token "
    return next(line[len(prefix):] for line in logical_lines if line.startswith(prefix))


@pytest.mark.parametrize(
    "repo,revision,weights,error",
    [
        ("owner/model", "a" * 40, "weights/model.safetensors", "Choose HF_REPO or WEIGHT_PATH"),
        ("", "a" * 40, "", "HF_REVISION requires HF_REPO"),
        ("", "a" * 40, "weights/model.safetensors", "HF_REVISION requires HF_REPO"),
    ],
)
def test_invalid_weight_arguments_stop_before_model_tools(repo, revision, weights, error):
    result = run_weight_step(repo, revision, weights)
    assert result.returncode != 0
    assert error in result.stderr
    assert result.stdout == "", "Invalid arguments must not reach fetch or bake"


@pytest.mark.parametrize(
    "repo,revision,weights,tool",
    [
        ("owner/model", "a" * 40, "", "tools.fetch_model"),
        ("", "", "weights/model.safetensors", "tools.bake_weights"),
        ("", "", "", "tools.bake_weights"),
    ],
)
def test_valid_weight_arguments_reach_selected_tool(repo, revision, weights, tool):
    result = run_weight_step(repo, revision, weights)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == tool


def run_weight_step(repo, revision, weights):
    env = dict(os.environ, MODEL="MST", HF_REPO=repo, HF_REVISION=revision, WEIGHT_PATH=weights)
    return subprocess.run(
        ["sh", "-c", 'python() { printf "%s\\n" "$2"; }; ' + weight_step()],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
