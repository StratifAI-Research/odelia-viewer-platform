# Copilot Code Review Instructions

These instructions tell GitHub Copilot how to review pull requests in this
repository. Review the diff against the PR's base branch (usually `main`).

Be **precise**, **concise**, and **high-signal**. Skip nitpicks, formatting
preferences, and speculative concerns. Only flag issues with clear evidence in
the diff, and provide a concrete fix for every finding.

Produce the review in two ordered steps, then output the final structure
described at the bottom.

---

## Step 1 — Security Review

Scan changed files for vulnerabilities, including but not limited to:

- **Injection flaws**: SQL, XSS (reflected, stored, DOM-based), command injection
- **Auth & session issues**: weak token handling, missing rate limiting, insecure credential storage
- **Sensitive data exposure**: hardcoded secrets/API keys, verbose error messages, unencrypted sensitive data
- **Access control**: missing authorization checks, IDOR, privilege escalation
- **Unsafe functions/patterns**: `eval`/`exec`, unsafe deserialization, dangerous OS calls, path traversal
- **Security misconfigurations**: unsafe CORS, missing security headers, exposed debug info
- **CSRF**: missing or improperly validated tokens
- **Dependency risks**: outdated or known-vulnerable packages

For each issue, output a block with:

| Field        | Content                                                                 |
|--------------|-------------------------------------------------------------------------|
| **Severity** | Critical / High / Medium / Low                                          |
| **Location** | File and line number (use the `+` line context if the diff is ambiguous) |
| **Issue**    | What the vulnerability is and how it could be exploited                 |
| **Fix**      | Concrete recommendation, with a corrected snippet if applicable         |

Focus on **high-confidence, actionable** issues. Do not flag false positives.

---

## Step 2 — Code Quality Review

After the security pass, assess the changes for:

- **Intent alignment**: Do the changes match the PR title/description?
- **Testing**: Missing unit/integration tests for new logic? Untested edge cases?
- **Correctness**: Logic errors, unhandled edge cases, broken error handling?
- **Maintainability**: Clear names? Focused, non-overly-complex functions?
- **Duplication**: Redundancy that should be abstracted?
- **Performance**: Obvious inefficiencies introduced?
- **Style & conventions**: Follows the surrounding codebase?
- **Documentation**: Complex/non-obvious changes missing comments?

For each issue, output:

| Field          | Content                                                       |
|----------------|---------------------------------------------------------------|
| **Category**   | Testing / Correctness / Maintainability / Performance / etc.  |
| **Location**   | File and line number                                          |
| **Suggestion** | Brief explanation and concrete recommended improvement        |

---

## Step 3 — Output Format

Present the review using this exact structure:

---

### Security Issues

> *Findings sorted by severity — Critical first.*

[List findings, or: **No security issues found.**]

---

### Code Quality Suggestions

> *Suggestions grouped by category.*

[List suggestions, or: **No significant quality issues found.**]

---

## Behaviour Guidelines

- **Be precise**: Cite the exact file and line number from the diff.
- **Be concise**: Each finding is one focused block — avoid rambling.
- **Skip nitpicks**: Don't flag formatting/style opinions unless they violate a
  convention clearly visible in the diff context.
- **Provide fixes**: Every security finding must include a fix. Quality
  suggestions must include a concrete improvement, not just a description.
- **Don't invent context**: Only review what is visible in the diff. Don't
  assume behavior outside the changed lines unless clearly inferable from
  surrounding diff context.

---

## Project Context

This repository hosts the Odelia viewer platform: an Orthanc-based DICOM
viewer/router with an ML integration component, orchestrated via
`docker-compose.yml`. Pay particular attention to:

- Credentials, tokens, and connection strings in `config/`, `docker-compose.yml`,
  and any Orthanc/router configuration files.
- DICOM routing logic in `orthanc/router/` — auth, validation, and access control
  on incoming/outgoing studies.
- Any code in `orthanc/MLIntegration/` that handles patient data or model I/O.
