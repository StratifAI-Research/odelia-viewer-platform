## What and why
<!-- What changes and why. Link the issue or ticket if there is one.
     No patient data, secrets, or details of unfixed security problems. -->

## Impact assessment
<!-- After each question, answer **Yes**, **No** or **Unsure**, followed by a short reason,
     e.g. "**No** — docs only". Think about indirect effects too:
     a weakened test or a changed sentence in the docs can matter. -->

- **Hazards:** Could this change introduce a new hazard or affect an existing one (e.g. a
  result shown for the wrong patient, an image displayed incorrectly, a failed AI result
  shown as valid)?
- **Requirements:** Does it add, change or remove behaviour described in a requirement,
  or affect the intended use?
- **Verification evidence:** Does it affect existing verification items or evidence tests,
  or need new ones?
- **AI output:** Does it change how AI results are produced (model, preprocessing,
  thresholds) or how users see and interact with them?
- **Security and data:** Does it change logins, permissions, network exposure, stored data
  or data sent to third parties?
- **Existing installations:** Do sites need to do anything when updating (configuration,
  migration, new dependency)?

<!-- For any Yes or Unsure: describe it in the reason; name HAZ-, requirement or VER- IDs if
     you know them (QA can add them). -->

## How it was verified
<!-- A few sentences, answering whichever of these apply: -->
- Which automated tests did you add or change, and what do they cover?
- Which existing tests already cover the change (e.g. for a refactoring or an update)?
- Which manual checks did you do: what, on which build, browser or data, and the result?
- If no verification is needed, why not?

## Release note
<!-- Users or admins would notice the change: under ## [Unreleased] in the changelog,
     replace N/A below the right heading with a line ending in this PR's number. If this PR
     releases (bumps the version), rename ## [Unreleased] to ## [X.Y.Z] - YYYY-MM-DD and add
     a fresh ## [Unreleased] above it, with N/A under every heading. -->
- [ ] Added to the changelog (under Unreleased, or under the new version if this PR releases)
- [ ] Not needed: nothing users or admins would notice
