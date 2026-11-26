"""Tests for change_control.py: python3 -m unittest discover -s .github/scripts"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import change_control as cc

EMPTY = "".join(f"### {s}\n\nN/A\n\n" for s in cc.SECTIONS)


def section(fixed: str) -> str:
    """All headings with N/A, except Fixed, which gets FIXED if it is not empty."""
    return EMPTY.replace("### Fixed\n\nN/A\n\n", f"### Fixed\n\n{fixed}\n\n") if fixed else EMPTY


def changelog(unreleased: str = "", *versions: tuple[str, str, str]) -> str:
    """versions: (version, date, entries under Fixed)."""
    text = "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n" + section(unreleased)
    for version, date, fixed in versions:
        text += f"## [{version}] - {date}\n\n" + section(fixed)
    return text


TEMPLATE = (Path(__file__).parents[1] / "pull_request_template.md").read_text()
RELEASED = ("1.1.0", "2026-02-01", "- Fixed the thing. (#3)")
OLDER = ("1.0.0", "2026-01-01", "- First fix. (#1)")
LOG = changelog("", RELEASED, OLDER)


def problems(text: str) -> list[str]:
    versions, errors = cc.parse_changelog(text, "CHANGELOG.md")
    return errors + cc.check_format(versions, "CHANGELOG.md")


FILLED = """## What and why
Fixes the wrong study being shown after switching patients (#81).

## Impact assessment
- **Hazards:** Could this change ... shown as valid)? **Yes** — fixes HAZ-007, wrong study shown
- **Requirements:** Does it add ... intended use? **No** — behaviour as specified
- **Verification evidence:** Does it affect ...
  or need new ones? **No** — existing VER-042 test still applies
- **AI output:** Does it change ...? **No** — no AI code touched
- **Security and data:** Does it change ...? **No** — no access changes
- **Existing installations:** Do sites ...? **No** — no configuration

## How it was verified
<!-- A few sentences, answering whichever of these apply. -->
`test_study_switch` fails on main and passes with the fix. Clicked through in Chrome.

## Release note
- [x] Added to the changelog (under Unreleased, or under the new version if this PR releases)
- [ ] Not needed: nothing users or admins would notice
"""


class ChangelogFormat(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(problems(LOG), [])

    def test_missing_heading(self) -> None:
        errors = problems(LOG.replace("### Security\n\nN/A\n\n", "", 1))
        self.assertTrue(any("missing Security" in e for e in errors), errors)

    def test_empty_heading_needs_na(self) -> None:
        errors = problems(LOG.replace("### Security\n\nN/A\n\n", "### Security\n\n", 1))
        self.assertTrue(any("Security: write N/A" in e for e in errors), errors)

    def test_na_with_entries(self) -> None:
        errors = problems(LOG.replace("- Fixed the thing. (#3)", "N/A\n- Fixed the thing. (#3)"))
        self.assertTrue(any("Fixed: remove N/A" in e for e in errors), errors)

    def test_na_is_not_a_list_item(self) -> None:
        errors = problems(changelog("- N/A", RELEASED))
        self.assertTrue(any("not as a list item" in e for e in errors), errors)

    def test_unknown_heading(self) -> None:
        errors = problems(LOG.replace("### Fixed", "### Fixes", 1))
        self.assertTrue(any("not allowed: Fixes" in e for e in errors), errors)

    def test_out_of_order(self) -> None:
        text = LOG.replace("### Added\n\nN/A\n\n### Changed", "### Changed\n\nN/A\n\n### Added", 1)
        self.assertTrue(any("out of order" in e for e in problems(text)))

    def test_entries_are_list_items(self) -> None:
        text = changelog("Plain text, not a bullet.", RELEASED)
        self.assertTrue(any("list items" in e for e in problems(text)))

    def test_wrapped_and_indented_lines_belong_to_the_entry(self) -> None:
        text = changelog("- First line\n  second line\n\n  ```bash\n  echo hi\n  ```", RELEASED)
        self.assertEqual(problems(text), [])
        versions, _ = cc.parse_changelog(text, "CHANGELOG.md")
        self.assertIn("echo hi", versions[0].entries["Fixed"][0].text)

    def test_old_heading_style(self) -> None:
        self.assertTrue(any("version heading" in e for e in problems(LOG.replace("## [1.1.0] - 2026-02-01", "## 1.1.0"))))

    def test_unreleased_first(self) -> None:
        text = changelog("", RELEASED).replace("## [Unreleased]", "## [0.9.0] - 2025-01-01")
        self.assertTrue(any("first section" in e for e in problems(text)))

    def test_versions_newest_first(self) -> None:
        self.assertTrue(any("newest first" in e for e in problems(changelog("", OLDER, RELEASED))))

    def test_released_version_needs_entries(self) -> None:
        text = changelog("", ("1.1.0", "2026-02-01", ""))
        self.assertTrue(any("no entries" in e for e in problems(text)))

    def test_bad_date(self) -> None:
        text = changelog("", ("1.1.0", "2026-02-30", "- x"))
        self.assertTrue(any("invalid date" in e for e in problems(text)))


class Release(unittest.TestCase):
    def test_ready(self) -> None:
        versions, _ = cc.parse_changelog(LOG, "CHANGELOG.md")
        self.assertEqual(cc.release_problems(versions, "1.1.0", "CHANGELOG.md"), [])

    def test_wrong_version_on_top(self) -> None:
        versions, _ = cc.parse_changelog(LOG, "CHANGELOG.md")
        self.assertTrue(cc.release_problems(versions, "1.2.0", "CHANGELOG.md"))

    def test_unreleased_must_be_empty(self) -> None:
        versions, _ = cc.parse_changelog(changelog("- Left behind.", RELEASED), "CHANGELOG.md")
        errors = cc.release_problems(versions, "1.1.0", "CHANGELOG.md")
        self.assertTrue(any("still has entries" in e for e in errors), errors)

    def test_notes_drop_empty_headings(self) -> None:
        versions, _ = cc.parse_changelog(LOG, "CHANGELOG.md")
        self.assertEqual(cc.release_notes(versions, "1.1.0"), "### Fixed\n\n- Fixed the thing. (#3)\n")

    def test_refs(self) -> None:
        self.assertEqual(
            cc.refs("Fix (Org/viewer#147, #68); not &#39;", "Org/platform"),
            ["Org/viewer#147", "Org/platform#68"],
        )


class Description(unittest.TestCase):
    def test_filled(self) -> None:
        r = cc.check_description(FILLED)
        self.assertEqual(r.errors, [])
        self.assertEqual(r.answers["Hazards"], ("Yes", "fixes HAZ-007, wrong study shown"))
        self.assertEqual(r.route, "QA approves on this pull request")
        self.assertEqual(r.release_note, cc.NOTE_ADDED)

    def test_unanswered(self) -> None:
        r = cc.check_description(FILLED.replace(" **No** — no AI code touched", ""))
        self.assertTrue(any('answer "AI output"' in e for e in r.errors), r.errors)

    def test_answer_on_the_next_line(self) -> None:
        r = cc.check_description(FILLED.replace(" **No** — no AI code touched", "\n  **No** — no AI code touched"))
        self.assertEqual(r.errors, [])
        self.assertEqual(r.answers["AI output"], ("No", "no AI code touched"))

    def test_answer_without_reason(self) -> None:
        r = cc.check_description(FILLED.replace("**No** — no AI code touched", "**No** —"))
        self.assertTrue(any('reason after "No" for "AI output"' in e for e in r.errors), r.errors)

    def test_plain_answer_is_accepted(self) -> None:
        r = cc.check_description(FILLED.replace("**No** — no AI code touched", "no, docs only"))
        self.assertEqual(r.errors, [])

    def test_routes(self) -> None:
        all_no = FILLED.replace("**Yes** — fixes HAZ-007", "**No** — fixes")
        self.assertEqual(cc.check_description(all_no).route, "one engineering review")
        security = all_no.replace("**No** — no access changes", "**Unsure** — new header")
        self.assertEqual(cc.check_description(security).route, "a code owner acting for QA approves")

    def test_comments_do_not_count(self) -> None:
        r = cc.check_description(FILLED.replace(
            "Fixes the wrong study being shown after switching patients (#81).", "<!-- todo -->"
        ))
        self.assertIn("What and why: say what changes and why", r.errors)

    def test_verification_needed(self) -> None:
        r = cc.check_description(FILLED.replace(
            "`test_study_switch` fails on main and passes with the fix. Clicked through in Chrome.", ""
        ))
        self.assertTrue(any(e.startswith("How it was verified:") for e in r.errors), r.errors)

    def test_no_verification_needed_with_reason(self) -> None:
        r = cc.check_description(FILLED.replace(
            "`test_study_switch` fails on main and passes with the fix. Clicked through in Chrome.",
            "None needed — docs only.",
        ))
        self.assertEqual(r.errors, [])

    def test_answers_below_the_template_questions(self) -> None:
        questions = cc.body_sections(TEMPLATE)["How it was verified"]
        r = cc.check_description(FILLED.replace("## How it was verified", f"## How it was verified\n{questions}"), TEMPLATE)
        self.assertEqual(r.errors, [])
        self.assertTrue(r.verification.startswith("`test_study_switch`"), r.verification)

    def test_template_questions_alone_are_not_an_answer(self) -> None:
        questions = cc.body_sections(TEMPLATE)["How it was verified"]
        body = FILLED.replace(
            "`test_study_switch` fails on main and passes with the fix. Clicked through in Chrome.", questions
        )
        r = cc.check_description(body, TEMPLATE)
        self.assertTrue(any(e.startswith("How it was verified:") for e in r.errors), r.errors)

    def test_one_release_note_box(self) -> None:
        r = cc.check_description(FILLED.replace("- [ ] Not needed", "- [x] Not needed"))
        self.assertIn("Release note: tick exactly one box", r.errors)

    def test_template_itself_is_incomplete(self) -> None:
        r = cc.check_description(TEMPLATE, TEMPLATE)
        for label in cc.IMPACT:
            self.assertTrue(any(f'answer "{label}"' in e for e in r.errors), label)
        self.assertTrue(any(e.startswith("How it was verified:") for e in r.errors), r.errors)
        self.assertFalse(any("missing" in e or "keep the template" in e for e in r.errors), r.errors)


class ChangelogChange(unittest.TestCase):
    def run_check(self, body: str, head: str, base: str, head_version: str = "1.1.0") -> cc.Report:
        r = cc.check_description(body)
        cc.check_changelog_change(r, head, base, head_version, "1.1.0", "CHANGELOG.md")
        return r

    def test_line_added_under_unreleased(self) -> None:
        r = self.run_check(FILLED, changelog("- New fix. (#81)", RELEASED, OLDER), LOG)
        self.assertEqual(r.errors, [])

    def test_ticked_added_but_no_line(self) -> None:
        r = self.run_check(FILLED, LOG, LOG)
        self.assertTrue(any("no new line under [Unreleased]" in e for e in r.errors), r.errors)

    def test_not_needed_but_line_added_warns(self) -> None:
        body = FILLED.replace("- [x] Added", "- [ ] Added").replace("- [ ] Not needed", "- [x] Not needed")
        r = self.run_check(body, changelog("- New fix. (#81)", RELEASED, OLDER), LOG)
        self.assertEqual(r.errors, [])
        self.assertTrue(r.warnings)

    def test_release_in_the_same_pr(self) -> None:
        base = changelog("- Earlier fix. (#80)", RELEASED, OLDER)
        head = changelog("", ("1.2.0", "2026-03-01", "- Earlier fix. (#80)\n- New fix. (#81)"), RELEASED, OLDER)
        self.assertEqual(self.run_check(FILLED, head, base, "1.2.0").errors, [])

    def test_release_without_renaming_unreleased(self) -> None:
        head = changelog("- New fix. (#81)", RELEASED, OLDER)
        r = self.run_check(FILLED, head, LOG, "1.2.0")
        self.assertTrue(any('"## [1.2.0] - YYYY-MM-DD"' in e for e in r.errors), r.errors)


class RepositoryFiles(unittest.TestCase):
    def test_changelog_follows_the_format(self) -> None:
        _, name = cc.repo_files()
        self.assertEqual(problems((cc.ROOT / name).read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
