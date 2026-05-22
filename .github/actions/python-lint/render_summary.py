"""Render lint outputs (ruff, ruff format, mypy) into a step-summary table.

Reads ./outputs/{ruff.json,ruff-format.txt,mypy.txt} and writes:
  - ./outputs/summary.md       (concatenated section per tool, top-10 tables)
  - ./outputs/metadata.json    (sha, ref, event, timestamps, tool versions, scope checksum)

Appends summary.md to GITHUB_STEP_SUMMARY when that env var is set.
Pure function on file contents — no network, no GH API calls.

Exit code:
  0  — no violations and no tool crash, or PYTHON_LINT_WARN_ONLY=true
  1  — tool crash (non-zero exit with zero parsed output), OR violations in gating mode

Tool crash detection: each tool step writes `outputs/<tool>.exit` with its exit
code. A non-zero exit code paired with zero parsed violations is treated as a
crash (config error, import failure, etc.) and always fails the job — even when
warn-only is on, because warn-only is for "violations are debt", not "the linter
itself broke".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


OUT = Path("outputs")


def _read_text(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    return path.read_text()


def _read_exit_code(path: Path) -> int | None:
    """Return the exit code written by the action.yml step, or None if absent."""
    raw = _read_text(path).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _tool_version(tool: str) -> str:
    try:
        result = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, check=False
        )
        raw = result.stdout or result.stderr
        return raw.strip().splitlines()[0] if raw else ""
    except FileNotFoundError:
        return ""


def _render_ruff(data: list[dict]) -> str:
    violations = len(data)
    file_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    for v in data:
        fn = v.get("filename", "") or (v.get("location", {}) or {}).get("file", "")
        file_counts[fn] = file_counts.get(fn, 0) + 1
        code = v.get("code") or "(unknown)"
        rule_counts[code] = rule_counts.get(code, 0) + 1

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:10]
    top_rules = sorted(rule_counts.items(), key=lambda x: -x[1])[:10]

    lines = ["## Lint PY (ruff)\n"]
    lines.append(f"- Files with violations: **{len(file_counts)}**")
    lines.append(f"- Total violations: **{violations}**")
    if top_rules:
        lines.append("\n**Top 10 rules:**\n")
        lines.append("| Rule | Count |")
        lines.append("|---|---:|")
        for r, n in top_rules:
            lines.append(f"| `{r}` | {n} |")
    if top_files:
        lines.append("\n**Top 10 files:**\n")
        lines.append("| File | Violations |")
        lines.append("|---|---:|")
        for fp, n in top_files:
            lines.append(f"| `{fp}` | {n} |")
    return "\n".join(lines) + "\n"


def _render_ruff_format(text: str) -> str:
    files_to_reformat = re.findall(r"^Would reformat:\s+(\S+)", text, re.MULTILINE)
    lines = ["## Lint PY (ruff format)\n"]
    lines.append(f"- Files needing reformat: **{len(files_to_reformat)}**")
    if files_to_reformat:
        lines.append("\n**Files:**\n")
        for f in files_to_reformat[:20]:
            lines.append(f"- `{f}`")
    return "\n".join(lines) + "\n"


def _render_mypy(text: str) -> str:
    error_lines = [line for line in text.splitlines() if ": error:" in line]
    file_counts: dict[str, int] = {}
    code_counts: dict[str, int] = {}
    pattern = re.compile(r"^([^:]+):\d+(?::\d+)?:\s+error:\s+.+?\[(\S+)\]\s*$")
    for line in error_lines:
        m = pattern.match(line)
        if not m:
            continue
        file_counts[m.group(1)] = file_counts.get(m.group(1), 0) + 1
        code_counts[m.group(2)] = code_counts.get(m.group(2), 0) + 1

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:10]
    top_codes = sorted(code_counts.items(), key=lambda x: -x[1])[:10]

    lines = ["## Lint PY (mypy)\n"]
    lines.append(f"- Files with errors: **{len(file_counts)}**")
    lines.append(f"- Total errors: **{len(error_lines)}**")
    if top_codes:
        lines.append("\n**Top 10 error codes:**\n")
        lines.append("| Code | Count |")
        lines.append("|---|---:|")
        for c, n in top_codes:
            lines.append(f"| `{c}` | {n} |")
    if top_files:
        lines.append("\n**Top 10 files:**\n")
        lines.append("| File | Errors |")
        lines.append("|---|---:|")
        for fp, n in top_files:
            lines.append(f"| `{fp}` | {n} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    ruff_raw = _read_text(OUT / "ruff.json")
    ruff_data: list[dict] = []
    if ruff_raw:
        try:
            ruff_data = json.loads(ruff_raw)
        except json.JSONDecodeError:
            ruff_data = []

    ruff_format_text = _read_text(OUT / "ruff-format.txt")
    mypy_text = _read_text(OUT / "mypy.txt")

    sha = os.environ.get("GITHUB_SHA", "")
    short_sha = sha[:7] if sha else "local"
    event = os.environ.get("GITHUB_EVENT_NAME", "local")

    header = f"# Lint PY suite — `{short_sha}` — `{event}`\n\n"
    body = (
        header
        + _render_ruff(ruff_data)
        + "\n"
        + _render_ruff_format(ruff_format_text)
        + "\n"
        + _render_mypy(mypy_text)
    )

    (OUT / "summary.md").write_text(body)

    scope_str = "ruff|ruff-format|mypy"
    metadata = {
        "sha": sha,
        "ref": os.environ.get("GITHUB_REF", ""),
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "python-lint-suite",
        "ruff_version": _tool_version("ruff"),
        "mypy_version": _tool_version("mypy"),
        "scope_checksum": hashlib.sha256(scope_str.encode()).hexdigest(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(body)

    # --- Exit-code logic ---
    total_ruff = len(ruff_data)
    total_ruff_format = len(re.findall(r"^Would reformat:", ruff_format_text, re.MULTILINE))
    total_mypy = len([line for line in mypy_text.splitlines() if ": error:" in line])

    tool_counts = {
        "ruff": total_ruff,
        "ruff-format": total_ruff_format,
        "mypy": total_mypy,
    }
    tool_crashes: list[str] = []
    for tool, count in tool_counts.items():
        exit_code = _read_exit_code(OUT / f"{tool}.exit")
        if exit_code is not None and exit_code != 0 and count == 0:
            tool_crashes.append(f"{tool} (exit={exit_code})")

    warn_only = os.environ.get("PYTHON_LINT_WARN_ONLY") == "true"
    mode_label = "warn-only" if warn_only else "gating"
    print(
        f"Lint PY: ruff={total_ruff} format={total_ruff_format} mypy={total_mypy} ({mode_label})",
        file=sys.stderr,
    )

    if tool_crashes:
        # Tool crash always fails the job, regardless of warn-only.
        print(
            f"::error::Tool crash with no parsed output: {', '.join(tool_crashes)}. "
            "Check the step logs — the tool likely failed before producing violations.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not warn_only and (total_ruff > 0 or total_ruff_format > 0 or total_mypy > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
