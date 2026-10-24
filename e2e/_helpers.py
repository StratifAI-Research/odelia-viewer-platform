"""Shared helpers for the Odelia OHIF frontend e2e suite.

Reusable login / banner-dismiss / screenshot-clip / event-wiring utilities.
All scripts import from here so selectors and flow stay in one place.
"""
import os
import time
import json

# Target the local stack by default; override with VIEWER_BASE_URL if needed.
BASE = os.environ.get("VIEWER_BASE_URL", "http://localhost:8081")
SHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
STUDY_UID = "1.3.46.670589.16.2.2.10.75.20.10.20100804.123124.5106477"

# Right-hand AI panel clip region (1680x1000 viewport).
PANEL_CLIP = {"x": 1330, "y": 40, "width": 350, "height": 950}

# Known-benign OHIF extension-registration warnings (NOT failures).
BENIGN_WARN_KEYS = ["orthanc-ai-routing", "labeling", "view-ai-result"]


def log(*a):
    print(*a, flush=True)


class Recorder:
    """Per-area capture of screenshots + console/page/network errors."""

    def __init__(self, area):
        self.area = area
        self.console = []      # (type, text)
        self.pageerrors = []   # str
        self.bad = []          # (status, url)
        self.ups = []          # (status, method, url) for AI endpoints
        self.shots = []        # list of dicts {path, caption}
        self._n = 0

    def wire(self, pg):
        pg.on("console", lambda m: self.console.append((m.type, m.text)))
        pg.on("pageerror", lambda e: self.pageerrors.append(str(e)))

        def onr(r):
            try:
                if r.status >= 400:
                    self.bad.append((r.status, r.url))
                if any(k in r.url for k in
                       ["/ups-rs", "/send-to-ai", "/workitem", "/analyz",
                        "/feedback", "/chat"]):
                    self.ups.append((r.status, r.request.method, r.url[-80:]))
            except Exception as _e:
                log("response handler: non-fatal", repr(_e)[:120])
        pg.on("response", onr)

    def shot(self, pg, label, caption, full=False, clip=None):
        self._n += 1
        path = f"{SHOTS_DIR}/{self.area}_{self._n:02d}_{label}.png"
        try:
            if full:
                pg.screenshot(path=path, full_page=True)
            elif clip:
                pg.screenshot(path=path, clip=clip)
            else:
                pg.screenshot(path=path)
            self.shots.append({"path": path, "caption": caption})
            log("SHOT", path)
        except Exception as e:
            log("SHOT-FAIL", label, repr(e)[:100])
        return path

    def panel(self, pg, label, caption):
        return self.shot(pg, label, caption, clip=PANEL_CLIP)

    # --- error classification -------------------------------------------
    def console_errors(self):
        seen, out = set(), []
        for t, x in self.console:
            if t == "error" and x[:140] not in seen:
                seen.add(x[:140])
                out.append(x)
        return out

    def is_benign(self, text):
        low = text.lower()
        return any(k in low for k in BENIGN_WARN_KEYS) and (
            "regist" in low or "warn" in low or "not found" in low
            or "missing" in low or "extension" in low)

    def app_errors(self):
        """Console errors that are NOT known-benign extension warnings."""
        return [e for e in self.console_errors() if not self.is_benign(e)]

    def bad_responses(self):
        seen, out = set(), []
        for s, u in self.bad:
            if u not in seen:
                seen.add(u)
                out.append((s, u))
        return out

    def summary_dict(self, status, notes=""):
        return {
            "area": self.area,
            "status": status,
            "notes": notes,
            "console_errors": self.console_errors(),
            "app_errors": self.app_errors(),
            "page_errors": self.pageerrors,
            "bad_responses": self.bad_responses(),
            "ups": self.ups,
            "shots": self.shots,
        }

    def write_summary(self, status, notes=""):
        d = self.summary_dict(status, notes)
        path = f"{SHOTS_DIR}/{self.area}_summary.json"
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        log("SUMMARY", path, status)
        return d


def browser(p):
    b = p.chromium.launch(headless=True,
                          args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = b.new_context(ignore_https_errors=True,
                        viewport={"width": 1680, "height": 1000})
    return b, ctx


def dismiss_banner(pg):
    """Dismiss the investigational-use banner that intercepts clicks."""
    try:
        b = pg.get_by_role("button", name="Confirm and Hide")
        if b.count():
            b.first.click(timeout=2500)
            return True
    except Exception as _e:
        log("banner dismiss: non-fatal", repr(_e)[:120])
    return False


def login(pg):
    """Keycloak login -> lands in OHIF worklist. Dismisses banner."""
    pg.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("#username", timeout=30000)
    pg.fill("#username", "viewer")
    pg.fill("#password", "viewer")
    for sel in ["#kc-login", "input[type=submit]", "button[type=submit]"]:
        if pg.locator(sel).count():
            pg.click(sel)
            break
    try:
        pg.wait_for_load_state("networkidle", timeout=45000)
    except Exception as _e:
        log("login networkidle wait: non-fatal", repr(_e)[:120])
    time.sleep(5)
    dismiss_banner(pg)


def open_study(pg):
    """From worklist, expand the study row and click AI Analysis Mode -> viewer."""
    pg.wait_for_selector("tr", timeout=30000)
    pg.locator("tr").first.click(timeout=8000)
    time.sleep(2)
    pg.get_by_text("AI Analysis Mode", exact=False).first.click(timeout=6000)
    try:
        pg.wait_for_load_state("networkidle", timeout=60000)
    except Exception as _e:
        log("open_study networkidle wait: non-fatal", repr(_e)[:120])
    time.sleep(10)
    dismiss_banner(pg)
    time.sleep(1)


def advance(pg, names):
    """Dismiss banner, then click the first matching, visible, enabled button."""
    dismiss_banner(pg)
    time.sleep(0.4)
    for n in names:
        try:
            el = pg.get_by_role("button", name=n, exact=False)
            if el.count() and el.first.is_visible() and el.first.is_enabled():
                el.first.scroll_into_view_if_needed(timeout=1500)
                el.first.click(timeout=4000)
                return n
        except Exception as _e:
            log("advance button click: non-fatal", repr(_e)[:120])
    return None
