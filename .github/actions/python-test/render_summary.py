"""Render a GitHub step summary from pytest junit XML and coverage XML."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import importlib.metadata


def main() -> None:
    out = Path("outputs")
    junit_path = out / "pytest.xml"
    cov_path = out / "pytest-coverage.xml"

    def write_and_maybe_fail(summary: str, fail_reason: str | None = None) -> None:
        (out / "summary.md").write_text(summary)
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write(summary)
        if fail_reason:
            print(f"::error::{fail_reason}")
            sys.exit(1)

    short_sha = (os.environ.get("GITHUB_SHA", "") or "")[:7]
    event = os.environ.get("GITHUB_EVENT_NAME", "")

    if not junit_path.exists() or junit_path.stat().st_size == 0:
        write_and_maybe_fail(
            f"## Test PY (pytest) — `{short_sha}` — `{event}`\n\n"
            "**Failed:** pytest did not produce `pytest.xml`. "
            "Likely causes: install failure or pytest crashed before writing output.\n",
            "pytest produced no junit XML",
        )

    collected = passed = failed = errors = skipped = 0
    try:
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        for s in suites:
            collected += int(s.get("tests", 0))
            failed += int(s.get("failures", 0))
            errors += int(s.get("errors", 0))
            skipped += int(s.get("skipped", 0))
        passed = collected - failed - errors - skipped
    except ET.ParseError as e:
        write_and_maybe_fail(
            f"## Test PY (pytest) — `{short_sha}` — `{event}`\n\n"
            f"**Failed:** `pytest.xml` is not valid XML: {e}\n",
            "pytest junit XML is malformed",
        )

    lines_covered = lines_valid = 0
    file_covs: list[tuple[str, float]] = []
    if cov_path.exists() and cov_path.stat().st_size > 0:
        try:
            cov_root = ET.parse(cov_path).getroot()
            lines_covered = int(float(cov_root.get("lines-covered", 0) or 0))
            lines_valid = int(float(cov_root.get("lines-valid", 0) or 0))
            for cls in cov_root.iter("class"):
                fn = cls.get("filename", "")
                rate = float(cls.get("line-rate", 0) or 0)
                file_covs.append((fn, round(rate * 100, 1)))
        except (ET.ParseError, ValueError) as e:
            print(f"::warning::coverage XML parse failed; coverage summary will read 0%: {e}")

    line_pct = (lines_covered / lines_valid * 100) if lines_valid else 0
    file_covs.sort(key=lambda x: x[1])
    lowest = file_covs[:10]

    # Keep in sync with coverage.cfg `omit` — divergence silently changes scope_checksum.
    excludes = [
        "tests", "MLIntegration/tests", "sample_data", "screenshots",
        ".venv", "__pycache__", ".pytest_cache",
    ]
    scope_str = "|".join(["**/*.py"] + ["!" + e for e in sorted(excludes)])
    metadata = {
        "sha": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "pytest+coverage",
        "tool_version": importlib.metadata.version("pytest"),
        "scope_checksum": hashlib.sha256(scope_str.encode()).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    out_lines = [f"## Test PY (pytest) — `{short_sha}` — `{event}`\n"]
    out_lines.append("### Tests\n")
    out_lines.append(f"- Collected: **{collected}**")
    out_lines.append(
        f"- Passed: **{passed}** — Failed: **{failed}** "
        f"— Errors: **{errors}** — Skipped: **{skipped}**"
    )
    out_lines.append("\n### Coverage\n")
    out_lines.append(f"- Lines covered: **{lines_covered}/{lines_valid}** → **{line_pct:.1f}%**")
    if lowest:
        out_lines.append("\n**Lowest-coverage files (top 10):**\n")
        out_lines.append("| File | Line % |")
        out_lines.append("|---|---:|")
        for fp, pct in lowest:
            out_lines.append(f"| `{fp}` | {pct}% |")

    summary = "\n".join(out_lines) + "\n"

    if collected == 0:
        write_and_maybe_fail(
            summary + "\n**Failed:** zero tests collected. "
            "Check that test files exist under the configured test directory.\n",
            "pytest collected zero tests",
        )

    write_and_maybe_fail(summary)


if __name__ == "__main__":
    main()
