#!/usr/bin/env python3
"""
Benchmark script comparing Ollama vs llama.cpp on /v1/chat/completions.

Supports text-only and multimodal (DICOM image) prompts.

Usage:
  # Text-only benchmark:
  python benchmark.py --ollama http://localhost:11434 --llamacpp http://localhost:8090

  # With real DICOM images from Orthanc:
  python benchmark.py --ollama http://localhost:11434 --llamacpp http://localhost:8090 \
      --series-uid 1.2.276.0.7230010.3.1.3.8323329.105762.1742335625.4202572

  # llama.cpp only with images:
  python benchmark.py --llamacpp http://localhost:8090 \
      --series-uid 1.2.276.0.7230010.3.1.3.8323329.105762.1742335625.4202572
"""
import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import requests


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    backend: str
    prompt_name: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    token_count: int = 0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None
    full_output: str = ""


# ---------------------------------------------------------------------------
# Orthanc image retrieval
# ---------------------------------------------------------------------------

def fetch_series_images(
    orthanc_url: str,
    series_uid: str,
    num_slices: int = 5,
) -> List[str]:
    """
    Fetch rendered PNG slices from Orthanc and return as base64 data URIs.
    Picks evenly-spaced instances from the middle of the series.
    """
    base = orthanc_url.rstrip("/")

    # Resolve SeriesInstanceUID -> Orthanc internal ID
    resp = requests.post(f"{base}/tools/lookup", data=series_uid, timeout=10)
    resp.raise_for_status()
    matches = resp.json()
    series_matches = [m for m in matches if m.get("Type") == "Series"]
    if not series_matches:
        raise RuntimeError(f"Series {series_uid} not found in Orthanc")
    orthanc_series_id = series_matches[0]["ID"]

    # List instances in the series
    resp = requests.get(f"{base}/series/{orthanc_series_id}/instances", timeout=10)
    resp.raise_for_status()
    instances = resp.json()
    if not instances:
        raise RuntimeError(f"No instances in series {orthanc_series_id}")

    # Sort by InstanceNumber (tag 0020,0013) for consistent ordering
    for inst in instances:
        try:
            tags = requests.get(
                f"{base}/instances/{inst['ID']}/simplified-tags", timeout=5
            ).json()
            inst["_sort_key"] = int(tags.get("InstanceNumber", 0))
        except Exception:
            inst["_sort_key"] = 0
    instances.sort(key=lambda x: x["_sort_key"])

    total = len(instances)
    n = min(num_slices, total)

    # Pick from central 60% (matching chat-middleware default)
    margin = int(total * 0.20)
    start = margin
    end = total - margin
    if end - start < n:
        start, end = 0, total

    if n == 1:
        indices = [(start + end) // 2]
    else:
        step = (end - start - 1) / (n - 1)
        indices = [int(start + i * step) for i in range(n)]

    print(f"  Fetching {n} slices from series ({total} instances total) ...")

    images: List[str] = []
    for idx in indices:
        inst_id = instances[idx]["ID"]
        r = requests.get(f"{base}/instances/{inst_id}/preview", timeout=15)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode("utf-8")
        images.append(f"data:image/png;base64,{b64}")

    print(f"  Retrieved {len(images)} images (total payload ~{sum(len(i) for i in images) // 1024} KB)")
    return images


def build_multimodal_content(images: List[str], user_text: str) -> list:
    """
    Build interleaved content array matching chat-middleware's PromptBuilder format:
      [image_url, "SLICE 1", image_url, "SLICE 2", ..., user_text]
    """
    content = []
    for idx, data_uri in enumerate(images, start=1):
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
        content.append({"type": "text", "text": f"SLICE {idx}"})
    content.append({"type": "text", "text": user_text})
    return content


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEXT_PROMPTS = [
    {
        "name": "short_text",
        "messages": [
            {"role": "system", "content": "You are a helpful radiology assistant."},
            {"role": "user", "content": "In two sentences, what are the key features of a malignant breast lesion on MRI?"},
        ],
    },
    {
        "name": "reasoning",
        "messages": [
            {"role": "system", "content": "You are a helpful radiology assistant."},
            {"role": "user", "content": (
                "A 52-year-old woman presents with a 1.5 cm enhancing mass in the upper outer quadrant "
                "of the left breast on dynamic contrast-enhanced MRI. The mass shows irregular margins "
                "and a washout kinetic curve. There is no associated axillary lymphadenopathy. "
                "What is your differential diagnosis and recommended next step?"
            )},
        ],
    },
]

IMAGE_PROMPTS = [
    {
        "name": "image_describe",
        "user_text": "Describe the findings in this breast MRI series.",
    },
    {
        "name": "image_classify",
        "user_text": (
            "Based on these MRI slices, classify the findings as: "
            "no lesion, benign, or malignant. Explain your reasoning."
        ),
    },
]


def make_image_messages(images: List[str], prompt_cfg: dict) -> list:
    return [
        {"role": "system", "content": "You are a helpful radiology assistant."},
        {"role": "user", "content": build_multimodal_content(images, prompt_cfg["user_text"])},
    ]


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def stream_request(
    base_url: str, model: str, messages: list, max_tokens: int = 256
) -> RunResult:
    """Send a streaming /v1/chat/completions request and measure timing."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }

    result = RunResult(backend=base_url)
    tokens: List[str] = []
    t_start = time.perf_counter()
    t_first_token = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                result.total_ms = (time.perf_counter() - t_start) * 1000
                return result

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            tokens.append(content)
                except json.JSONDecodeError:
                    continue

    except requests.RequestException as e:
        result.error = str(e)
        result.total_ms = (time.perf_counter() - t_start) * 1000
        return result

    t_end = time.perf_counter()
    result.total_ms = (t_end - t_start) * 1000
    result.ttft_ms = ((t_first_token - t_start) * 1000) if t_first_token else 0.0
    result.token_count = len(tokens)
    gen_sec = (t_end - t_first_token) if t_first_token else 0.0
    result.tokens_per_sec = (result.token_count / gen_sec) if gen_sec > 0 else 0.0
    result.full_output = "".join(tokens)

    return result


def check_health(base_url: str, backend_type: str) -> bool:
    endpoint = "/health" if backend_type == "llamacpp" else "/api/tags"
    try:
        r = requests.get(f"{base_url.rstrip('/')}{endpoint}", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    backends: list,
    prompts: list,
    rounds: int = 3,
    max_tokens: int = 256,
    warmup: bool = True,
    show_responses: bool = False,
):
    """Run the benchmark suite across all backends and prompts."""
    if warmup:
        print("\n--- Warmup round (discarded) ---")
        for be in backends:
            msg = [{"role": "user", "content": "Hi"}]
            stream_request(be["url"], be["model"], msg, max_tokens=16)
        print("Warmup complete.\n")

    all_results: dict[str, dict[str, list[RunResult]]] = {}

    for prompt_cfg in prompts:
        pname = prompt_cfg["name"]
        all_results[pname] = {}
        for be in backends:
            label = be["label"]
            all_results[pname][label] = []
            for r in range(rounds):
                print(f"  [{pname}] {label} round {r + 1}/{rounds} ...", end=" ", flush=True)
                res = stream_request(be["url"], be["model"], prompt_cfg["messages"], max_tokens)
                res.backend = label
                res.prompt_name = pname
                all_results[pname][label].append(res)
                if res.error:
                    print(f"ERROR: {res.error}")
                else:
                    print(f"TTFT={res.ttft_ms:.0f}ms  tok/s={res.tokens_per_sec:.1f}  tokens={res.token_count}")

    print_report(all_results, rounds)

    if show_responses:
        print_responses(all_results)


def print_report(all_results: dict, rounds: int):
    sep = "-" * 90
    print(f"\n{'=' * 90}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 90}")

    for pname, backend_results in all_results.items():
        print(f"\nPrompt: {pname}")
        print(sep)
        print(f"{'Backend':<20} {'TTFT (ms)':>12} {'Total (ms)':>12} {'Tokens':>8} {'Tok/s':>10} {'Errors':>8}")
        print(sep)

        for label, runs in backend_results.items():
            ok = [r for r in runs if r.error is None]
            errs = len(runs) - len(ok)
            if not ok:
                print(f"{label:<20} {'n/a':>12} {'n/a':>12} {'n/a':>8} {'n/a':>10} {errs:>8}")
                continue
            avg_ttft = sum(r.ttft_ms for r in ok) / len(ok)
            avg_total = sum(r.total_ms for r in ok) / len(ok)
            avg_tok = sum(r.token_count for r in ok) / len(ok)
            avg_tps = sum(r.tokens_per_sec for r in ok) / len(ok)
            print(f"{label:<20} {avg_ttft:>12.1f} {avg_total:>12.1f} {avg_tok:>8.0f} {avg_tps:>10.1f} {errs:>8}")

        print(sep)

    print(f"\nRounds per backend per prompt: {rounds}")
    print(f"{'=' * 90}\n")


def print_responses(all_results: dict):
    """Print full text responses from each backend, last round only."""
    print(f"{'=' * 90}")
    print("MODEL RESPONSES  (last round only)")
    print(f"{'=' * 90}")

    for pname, backend_results in all_results.items():
        for label, runs in backend_results.items():
            last = runs[-1] if runs else None
            if not last or last.error:
                continue
            print(f"\n--- [{pname}] {label} ---")
            print(last.full_output)
            print()

    print(f"{'=' * 90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama vs llama.cpp (text + multimodal)"
    )
    parser.add_argument("--ollama", type=str, help="Ollama base URL (e.g. http://localhost:11434)")
    parser.add_argument("--llamacpp", type=str, help="llama.cpp server base URL (e.g. http://localhost:8090)")
    parser.add_argument("--model", type=str, default="thiagomoraes/medgemma-1.5-4b-it:F16",
                        help="Model name for Ollama")
    parser.add_argument("--llamacpp-model", type=str, default="medgemma-1.5-4b-it-BF16",
                        help="Model name for llama.cpp")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per prompt per backend")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup round")
    parser.add_argument("--show-responses", action="store_true",
                        help="Print full text output from each model after the benchmark")

    # Multimodal options
    parser.add_argument("--series-uid", type=str,
                        help="DICOM SeriesInstanceUID to benchmark with real images")
    parser.add_argument("--orthanc-url", type=str, default="http://localhost:8000",
                        help="Orthanc REST API URL (default: http://localhost:8000)")
    parser.add_argument("--num-slices", type=int, default=5,
                        help="Number of slices to extract from the series (default: 5)")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip text prompts and only run image prompts")
    parser.add_argument("--image-only", action="store_true",
                        help="Skip text prompts and only run image prompts")

    args = parser.parse_args()

    if not args.ollama and not args.llamacpp:
        parser.error("Provide at least one of --ollama or --llamacpp")

    # --- Discover backends ---
    backends = []
    if args.ollama:
        ok = check_health(args.ollama, "ollama")
        status = "OK" if ok else "UNREACHABLE"
        print(f"Ollama    @ {args.ollama}  [{status}]")
        if not ok:
            print("  WARNING: Ollama not reachable -- will attempt requests anyway.")
        backends.append({"label": "ollama", "url": args.ollama, "model": args.model})

    if args.llamacpp:
        ok = check_health(args.llamacpp, "llamacpp")
        status = "OK" if ok else "UNREACHABLE"
        print(f"llama.cpp @ {args.llamacpp}  [{status}]")
        if not ok:
            print("  WARNING: llama.cpp not reachable -- will attempt requests anyway.")
        backends.append({"label": "llamacpp", "url": args.llamacpp, "model": args.llamacpp_model})

    # --- Build prompt list ---
    prompts = []

    if not args.image_only:
        prompts.extend(TEXT_PROMPTS)

    if args.series_uid:
        print(f"\nFetching DICOM series: {args.series_uid}")
        try:
            images = fetch_series_images(args.orthanc_url, args.series_uid, args.num_slices)
            for pcfg in IMAGE_PROMPTS:
                prompts.append({
                    "name": pcfg["name"],
                    "messages": make_image_messages(images, pcfg),
                })
        except Exception as e:
            print(f"  ERROR fetching images: {e}")
            print("  Falling back to text-only benchmark.")
    elif args.image_only:
        parser.error("--image-only requires --series-uid")

    if not prompts:
        parser.error("No prompts to run")

    print(f"\nPrompts: {[p['name'] for p in prompts]}")
    print(f"Backends: {[b['label'] for b in backends]}")
    print(f"Rounds: {args.rounds}, Max tokens: {args.max_tokens}\n")

    run_benchmark(
        backends,
        prompts,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        warmup=not args.no_warmup,
        show_responses=args.show_responses,
    )


if __name__ == "__main__":
    main()
