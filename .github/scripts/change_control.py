#!/usr/bin/env python3
"""Change control: checks and exports for pull request descriptions and the changelog.

odelia-viewer-platform and odelia-viewer use the same file; keep the two copies
identical. Standard library only, so CI runs it with the runner's python3.

    check-pr               the PR description answers the template's questions and its
                           release-note box matches the changelog (CI: Change control)
    check-changelog        the changelog follows the format described at its top
    release-check VERSION  VERSION's changelog section is ready to release
                           (CI: Build and release)
    notes VERSION          prints VERSION's GitHub Release text, N/A headings dropped
    export                 changelog entries as CSV or JSON rows
    changes                merged PRs with their impact assessment as CSV or JSON rows,
                           for the change control log (needs an authenticated gh)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (version file, changelog). The viewer keeps ODELIA's pair beside OHIF's CHANGELOG.md,
# so VERSION_ODELIA decides which repository this is.
FILES = (("VERSION_ODELIA", "CHANGELOG_ODELIA.md"), ("VERSION", "CHANGELOG.md"))

# Changelog: every version has all of these headings, in this order.
SECTIONS = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Known limitations",
    "Upgrade notes",
)
UNRELEASED = "Unreleased"
# A heading with nothing new says so, on a line of its own, so it doesn't look forgotten.
NOTHING_NEW = "N/A"
VERSION_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$")

# Pull request template (.github/pull_request_template.md).
IMPACT = (
    "Hazards",
    "Requirements",
    "Verification evidence",
    "AI output",
    "Security and data",
    "Existing installations",
)
QA_APPROVES = ("Hazards", "Requirements", "AI output")
NOTE_ADDED = "Added to the changelog"
NOTE_NOT_NEEDED = "Not needed"

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# An answer follows the question mark, on the same or the next line:
# "...? **No** — docs only". Matched after the bold markers are removed.
ANSWER = re.compile(r"\?\s*(yes|no|unsure)\b[\s.,:;—–-]*(.*)", re.IGNORECASE | re.DOTALL)
REF = re.compile(r"(?<![\w/&])(?:([\w.-]+/[\w.-]+))?#(\d+)\b")


# --------------------------------------------------------------------------- changelog


@dataclass
class Entry:
    lines: list[str]
    line: int

    @property
    def markdown(self) -> str:
        return "\n".join(self.lines).rstrip()

    @property
    def text(self) -> str:
        """The entry without its bullet: wrapped lines joined, code blocks kept."""
        lines = [self.lines[0][2:]]
        lines += [ln[2:] if ln.startswith("  ") else ln.strip() for ln in self.lines[1:]]
        out: list[str] = []
        fence = False
        for ln in lines:
            starts_block = ln.lstrip().startswith(("```", "- ", "* "))
            if ln.lstrip().startswith("```"):
                fence = not fence
            if fence or starts_block or not ln.strip() or not out or not out[-1].strip() \
                    or out[-1].lstrip().startswith("```"):
                out.append(ln)
            else:
                out[-1] += " " + ln.strip()
        return "\n".join(out).strip()

    @property
    def key(self) -> str:
        return " ".join(self.text.split())


@dataclass
class Version:
    name: str
    date: str | None
    line: int
    intro: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    heading_lines: dict[str, int] = field(default_factory=dict)
    entries: dict[str, list[Entry]] = field(default_factory=dict)
    nothing_new: set[str] = field(default_factory=set)

    def all_entries(self) -> list[Entry]:
        return [e for h in self.headings for e in self.entries.get(h, [])]


def parse_changelog(text: str, name: str) -> tuple[list[Version], list[str]]:
    versions: list[Version] = []
    errors: list[str] = []
    current: Version | None = None
    heading: str | None = None
    entry: Entry | None = None
    blank = False
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        where = f"{name}:{no}"
        if line.startswith("## "):
            heading, entry = None, None
            m = VERSION_HEADING.match(line)
            if line == f"## [{UNRELEASED}]":
                current = Version(UNRELEASED, None, no)
            elif m:
                current = Version(m[1], m[2], no)
            else:
                errors.append(
                    f'{where}: a version heading is "## [{UNRELEASED}]" or '
                    f'"## [1.2.3] - 2026-01-31", not "{line}"'
                )
                current = None
                continue
            versions.append(current)
            continue
        if current is None:
            continue  # the text above the first version, or under a bad heading
        if line.startswith("### "):
            heading, entry = line[4:].strip(), None
            current.headings.append(heading)
            current.heading_lines.setdefault(heading, no)
            current.entries.setdefault(heading, [])
            continue
        if line.startswith("#"):
            errors.append(f"{where}: only ## version and ### section headings are allowed")
            continue
        if heading is None:
            current.intro.append(line)
            continue
        if not line:
            blank = True
            if entry:
                entry.lines.append("")
            continue
        if line == NOTHING_NEW:
            current.nothing_new.add(heading)
            entry = None
        elif line.startswith("- "):
            entry = Entry([line], no)
            current.entries[heading].append(entry)
        elif entry and (line.startswith((" ", "\t")) or not blank):
            entry.lines.append(line)
        else:
            errors.append(f'{where}: write entries as list items starting with "- "')
        blank = False
    for v in versions:
        for e in v.all_entries():
            while e.lines and not e.lines[-1]:
                e.lines.pop()
        while v.intro and not v.intro[-1]:
            v.intro.pop()
        while v.intro and not v.intro[0]:
            v.intro.pop(0)
    return versions, errors


def _number(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def check_format(versions: list[Version], name: str) -> list[str]:
    errors: list[str] = []
    if not versions or versions[0].name != UNRELEASED:
        errors.append(f'{name}: the first section must be "## [{UNRELEASED}]"')
    newer: Version | None = None
    seen: set[str] = set()
    for i, v in enumerate(versions):
        where = f"{name}:{v.line}"
        if v.headings != list(SECTIONS):
            missing = [s for s in SECTIONS if s not in v.headings]
            extra = [h for h in v.headings if h not in SECTIONS]
            problem = "; ".join(
                p
                for p in (
                    missing and "missing " + ", ".join(missing),
                    extra and "not allowed: " + ", ".join(extra),
                )
                if p
            ) or "duplicated or out of order"
            errors.append(
                f"{where}: [{v.name}] needs exactly these headings, in this order: "
                f"{', '.join(SECTIONS)} ({problem})"
            )
        for heading in v.headings:
            at = f"{name}:{v.heading_lines[heading]}: [{v.name}] {heading}"
            entries = v.entries.get(heading, [])
            if entries and heading in v.nothing_new:
                errors.append(f"{at}: remove {NOTHING_NEW}, the heading has entries")
            elif not entries and heading not in v.nothing_new:
                errors.append(f"{at}: write {NOTHING_NEW} under a heading with nothing new")
            for e in entries:
                if e.key.lower().rstrip(".") in ("n/a", "na", "none", "-"):
                    errors.append(
                        f"{name}:{e.line}: write {NOTHING_NEW} on its own line, not as a list item"
                    )
        if v.name == UNRELEASED:
            if i:
                errors.append(f'{where}: "## [{UNRELEASED}]" must appear once, at the top')
            continue
        if v.name in seen:
            errors.append(f"{where}: [{v.name}] appears twice")
        seen.add(v.name)
        try:
            dt.date.fromisoformat(v.date or "")
        except ValueError:
            errors.append(f"{where}: [{v.name}] has an invalid date {v.date}")
        if newer and _number(newer.name) <= _number(v.name):
            errors.append(f"{where}: versions must be newest first ([{newer.name}] above [{v.name}])")
        elif newer and (newer.date or "") < (v.date or ""):
            errors.append(f"{where}: [{v.name}] is dated after the newer [{newer.name}]")
        if not v.all_entries():
            errors.append(f"{where}: [{v.name}] has no entries")
        newer = v
    return errors


def release_problems(versions: list[Version], version: str, name: str) -> list[str]:
    """What stops VERSION from being released, apart from format errors."""
    errors: list[str] = []
    released = [v for v in versions if v.name != UNRELEASED]
    top = released[0].name if released else "none"
    if len(versions) < 2 or versions[0].name != UNRELEASED or versions[1].name != version:
        errors.append(
            f'{name}: the new version needs a section "## [{version}] - YYYY-MM-DD" directly '
            f'below "## [{UNRELEASED}]" (the newest is "{top}")'
        )
    if versions and versions[0].name == UNRELEASED and versions[0].all_entries():
        errors.append(
            f"{name}: [{UNRELEASED}] still has entries; a release moves them into [{version}] "
            f"and leaves [{UNRELEASED}] with N/A under every heading"
        )
    return errors


def release_notes(versions: list[Version], version: str) -> str:
    v = next(v for v in versions if v.name == version)
    parts = ["\n".join(v.intro)] if v.intro else []
    for heading in SECTIONS:
        entries = v.entries.get(heading, [])
        if entries:
            parts.append(f"### {heading}\n\n" + "\n".join(e.markdown for e in entries))
    return "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------- PR description


def body_sections(body: str) -> dict[str, str]:
    body = COMMENT.sub("", body.replace("\r\n", "\n"))
    sections: dict[str, list[str]] = {}
    lines: list[str] | None = None
    for line in body.split("\n"):
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            lines = sections.setdefault(m[1].strip(), [])
        elif lines is not None:
            lines.append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def bullets(text: str) -> list[str]:
    """Top-level list items, with wrapped lines joined."""
    items: list[str] = []
    blank = True
    for line in text.split("\n"):
        if re.match(r"^[-*] ", line):
            items.append(line[2:].strip())
        elif items and line.strip() and (line[:1].isspace() or not blank):
            items[-1] += " " + line.strip()
        blank = not line.strip()
    return items


def impact_answers(text: str) -> dict[str, tuple[str, str] | None]:
    """label -> (Yes|No|Unsure, reason), or None if not answered."""
    answers: dict[str, tuple[str, str] | None] = {}
    for item in bullets(text):
        plain = item.replace("**", "")
        for label in IMPACT:
            if plain.lower().startswith(label.lower() + ":"):
                m = ANSWER.search(plain)
                answers[label] = (m[1].capitalize(), m[2].strip()) if m else None
    return answers


def boxes(text: str) -> list[tuple[bool, str, str]]:
    """(ticked, label, text after the label's colon) for each checkbox item."""
    found = []
    for item in bullets(text):
        m = re.match(r"\[([ xX])\]\s*(.*)", item, re.DOTALL)
        if m:
            label, _, rest = m[2].replace("**", "").partition(":")
            found.append((m[1] != " ", label.strip(), rest.strip()))
    return found


def has_words(text: str) -> bool:
    return bool(re.search(r"[^\W\d_]{2}", text))


def own_text(text: str, template: str) -> str:
    """TEXT without the lines it shares with the template, such as guiding questions."""
    prompts = {ln.strip() for ln in COMMENT.sub("", template).splitlines() if ln.strip()}
    return "\n".join(ln for ln in text.splitlines() if ln.strip() not in prompts).strip()


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    answers: dict[str, tuple[str, str] | None] = field(default_factory=dict)
    verification: str = ""
    release_note: str = ""
    route: str = ""


def approval_route(answers: dict[str, tuple[str, str] | None]) -> str:
    flagged = {k for k, a in answers.items() if a and a[0] in ("Yes", "Unsure")}
    if flagged & set(QA_APPROVES):
        return "QA approves on this pull request"
    if flagged:
        return "a code owner acting for QA approves"
    return "one engineering review"


def check_description(body: str, template: str = "") -> Report:
    """TEMPLATE: the PR template, whose own lines (the guiding questions) don't count as answers."""
    r = Report()
    sections = body_sections(body or "")
    for name in ("What and why", "Impact assessment", "How it was verified", "Release note"):
        if name not in sections:
            r.errors.append(f'Description: keep the template\'s "## {name}" section')
    if "What and why" in sections and not has_words(own_text(sections["What and why"], template)):
        r.errors.append("What and why: say what changes and why")

    r.answers = impact_answers(sections.get("Impact assessment", ""))
    for label in IMPACT:
        answer = r.answers.get(label)
        if label not in r.answers:
            r.errors.append(f'Impact assessment: the question "{label}" is missing')
        elif answer is None:
            r.errors.append(
                f'Impact assessment: answer "{label}" with Yes, No or Unsure and a short '
                'reason, e.g. "**No** — docs only"'
            )
        elif not has_words(answer[1]):
            r.errors.append(f'Impact assessment: add a short reason after "{answer[0]}" for "{label}"')

    r.verification = own_text(sections.get("How it was verified", ""), template)
    if "How it was verified" in sections and not has_words(r.verification):
        r.errors.append(
            "How it was verified: name the tests or manual checks that show this works, "
            'or say why none are needed, e.g. "None needed — docs only"'
        )

    notes = [label for ticked, label, _ in boxes(sections.get("Release note", "")) if ticked]
    if len(notes) != 1:
        r.errors.append("Release note: tick exactly one box")
    elif notes[0].startswith(NOTE_ADDED):
        r.release_note = NOTE_ADDED
    elif notes[0].startswith(NOTE_NOT_NEEDED):
        r.release_note = NOTE_NOT_NEEDED
    r.route = approval_route(r.answers)
    return r


def check_changelog_change(
    r: Report, head_log: str, base_log: str | None, head_version: str, base_version: str, name: str
) -> None:
    versions, errors = parse_changelog(head_log, name)
    r.errors += errors + check_format(versions, name)
    base_unreleased: set[str] = set()
    if base_log is not None:
        base, base_errors = parse_changelog(base_log, name)
        if base and base[0].name == UNRELEASED and not base_errors:
            base_unreleased = {e.key for e in base[0].all_entries()}
    if head_version != base_version:
        r.errors += release_problems(versions, head_version, name)
        section = next((v for v in versions if v.name == head_version), None)
    else:
        section = versions[0] if versions and versions[0].name == UNRELEASED else None
    new = {e.key for e in section.all_entries()} - base_unreleased if section else set()
    if r.release_note == NOTE_ADDED and not new:
        where = f"[{head_version}]" if head_version != base_version else f"[{UNRELEASED}]"
        r.errors.append(
            f'Release note: you ticked "{NOTE_ADDED}", but {name} has no new line under {where}'
        )
    elif r.release_note == NOTE_NOT_NEEDED and new:
        r.warnings.append(
            f'Release note: you ticked "{NOTE_NOT_NEEDED}", but {name} has new lines; '
            f'did you mean "{NOTE_ADDED}"?'
        )


# ------------------------------------------------------------------------- plumbing


def repo_files() -> tuple[str, str]:
    for version_file, changelog in FILES:
        if (ROOT / version_file).exists():
            return version_file, changelog
    sys.exit(f"none of {', '.join(v for v, _ in FILES)} found in {ROOT}")


def git(*args: str) -> str | None:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return p.stdout if p.returncode == 0 else None


def repo_name() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    url = (git("remote", "get-url", "origin") or "").strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return m[1] if m else ROOT.name


def refs(text: str, repo: str) -> list[str]:
    return [f"{m[1] or repo}#{m[2]}" for m in REF.finditer(text)]


def report_problems(errors: list[str], warnings: list[str], draft: bool = False) -> None:
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    for level, messages in (("error", errors), ("warning", warnings)):
        if level == "error" and draft:
            level = "warning"
        for msg in messages:
            print(f"::{level}::{msg}" if annotate else f"{level}: {msg}")


def write_summary(r: Report, draft: bool) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    cell = lambda s: s.replace("|", "\\|").replace("\n", " ")
    lines = ["## Change control", "", "| Question | Answer | Reason |", "| --- | --- | --- |"]
    for label in IMPACT:
        a = r.answers.get(label)
        lines.append(f"| {label} | {a[0] if a else '—'} | {cell(a[1]) if a else ''} |")
    lines += [
        "",
        f"**How it was verified:** {cell(r.verification) or '—'}  ",
        f"**Release note:** {r.release_note or '—'}  ",
        f"**Approval:** {r.route}",
    ]
    if r.errors:
        state = "not checked: draft" if draft else "failing"
        lines += ["", f"**To fix ({state}):**", *[f"- {e}" for e in r.errors]]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_rows(rows: list[dict[str, str]], fmt: str) -> None:
    if fmt == "json":
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return
    if rows:
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------- commands


def load(name: str) -> list[Version]:
    versions, errors = parse_changelog((ROOT / name).read_text(encoding="utf-8"), name)
    errors += check_format(versions, name)
    if errors:
        report_problems(errors, [])
        sys.exit(1)
    return versions


def cmd_check_pr(args: argparse.Namespace) -> int:
    version_file, changelog = repo_files()
    author = os.environ.get("PR_AUTHOR", "")
    draft = os.environ.get("PR_DRAFT") == "true"
    if author.endswith("[bot]"):
        r = Report(route="the reviewer's approval is the record")
        print(f"{author} opened this pull request: description not checked.")
    else:
        template = ROOT / ".github" / "pull_request_template.md"
        r = check_description(
            os.environ.get("PR_BODY", ""),
            template.read_text(encoding="utf-8") if template.exists() else "",
        )
    head_version = (ROOT / version_file).read_text().strip()
    base_version = (git("show", f"{args.base}:{version_file}") or "").strip()
    check_changelog_change(
        r,
        (ROOT / changelog).read_text(encoding="utf-8"),
        git("show", f"{args.base}:{changelog}"),
        head_version,
        base_version or head_version,
        changelog,
    )
    report_problems(r.errors, r.warnings, draft)
    write_summary(r, draft)
    if not r.errors:
        print(f"Description and {changelog} are complete. Approval: {r.route}.")
    return 1 if r.errors and not draft else 0


def cmd_check_changelog(args: argparse.Namespace) -> int:
    _, changelog = repo_files()
    load(changelog)
    print(f"{changelog} follows the format.")
    return 0


def cmd_release_check(args: argparse.Namespace) -> int:
    _, changelog = repo_files()
    errors = release_problems(load(changelog), args.version, changelog)
    report_problems(errors, [])
    return 1 if errors else 0


def cmd_notes(args: argparse.Namespace) -> int:
    _, changelog = repo_files()
    versions = load(changelog)
    if not any(v.name == args.version for v in versions):
        report_problems([f"{changelog} has no section [{args.version}]"], [])
        return 1
    sys.stdout.write(release_notes(versions, args.version))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    _, changelog = repo_files()
    repo = repo_name()
    rows = [
        {
            "repo": repo,
            "version": v.name,
            "date": v.date or "",
            "section": heading,
            "text": e.text,
            "refs": "; ".join(refs(e.text, repo)),
        }
        for v in load(changelog)
        if v.name != UNRELEASED or args.unreleased
        for heading in SECTIONS
        for e in v.entries.get(heading, [])
    ]
    write_rows(rows, args.format)
    return 0


def first_release(commit: str) -> str:
    tags = (git("tag", "--contains", commit, "--list", "v*") or "").split()
    versions = [t[1:] for t in tags if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return min(versions, key=_number) if versions else ""


def cmd_changes(args: argparse.Namespace) -> int:
    fields = "number,title,url,author,mergedAt,mergeCommit,body,reviews"
    out = subprocess.run(
        ["gh", "pr", "list", "--state", "merged", "--base", "main",
         "--limit", str(args.limit), "--json", fields],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for pr in sorted(json.loads(out), key=lambda p: p["mergedAt"]):
        if args.since and pr["mergedAt"][:10] < args.since:
            continue
        sections = body_sections(pr.get("body") or "")
        answers = impact_answers(sections.get("Impact assessment", ""))
        notes = [label for t, label, _ in boxes(sections.get("Release note", "")) if t]
        row = {
            "repo": repo_name(),
            "pr": str(pr["number"]),
            "title": pr["title"],
            "url": pr["url"],
            "author": pr["author"]["login"],
            "merged": pr["mergedAt"][:10],
            "approved_by": "; ".join(sorted(
                {rv["author"]["login"] for rv in pr.get("reviews") or [] if rv.get("state") == "APPROVED"}
            )),
            "release": first_release((pr.get("mergeCommit") or {}).get("oid", "")),
        }
        for label in IMPACT:
            a = answers.get(label)
            row[label] = a[0] if a else ""
            row[f"{label} (reason)"] = a[1] if a else ""
        row["how_verified"] = sections.get("How it was verified", "")
        row["release_note"] = "; ".join(notes)
        row["approval"] = approval_route(answers) if answers else ""
        rows.append(row)
    write_rows(rows, args.format)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check-pr", help="check the PR description (PR_BODY) and the changelog change")
    c.add_argument("--base", default="HEAD^1", help="the commit to compare with (default: HEAD^1)")
    c.set_defaults(func=cmd_check_pr)
    sub.add_parser("check-changelog", help="check the changelog format").set_defaults(func=cmd_check_changelog)
    for name, func, text in (
        ("release-check", cmd_release_check, "check that VERSION is ready to release"),
        ("notes", cmd_notes, "print VERSION's release notes"),
    ):
        c = sub.add_parser(name, help=text)
        c.add_argument("version")
        c.set_defaults(func=func)
    c = sub.add_parser("export", help="changelog entries as rows")
    c.add_argument("--format", choices=("csv", "json"), default="csv")
    c.add_argument("--unreleased", action="store_true", help="include [Unreleased]")
    c.set_defaults(func=cmd_export)
    c = sub.add_parser("changes", help="merged pull requests as rows (needs gh)")
    c.add_argument("--format", choices=("csv", "json"), default="csv")
    c.add_argument("--since", help="only PRs merged on or after this date (YYYY-MM-DD)")
    c.add_argument("--limit", type=int, default=1000)
    c.set_defaults(func=cmd_changes)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
