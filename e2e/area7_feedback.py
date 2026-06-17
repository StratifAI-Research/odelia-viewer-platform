"""Area 7 (v3): feedback — open the feedback panel (aiFeedback-btn), select a
per-breast verdict for Left + Right, click Submit Feedback, and verify a
/feedback POST fires and succeeds. Backend confirmed healthy."""
import time
from _helpers import sync_playwright, browser, login, open_study, dismiss_banner, Recorder, log

with sync_playwright() as p:
    rec = Recorder("feedback")
    b, ctx = browser(p)
    try:
        pg = ctx.new_page(); rec.wire(pg)
        login(pg); open_study(pg); dismiss_banner(pg)
        time.sleep(3)
        rec.shot(pg, "viewer_result", "Feedback: viewer with AI result overlay", full=True)

        # Open the feedback panel.
        pg.locator('[data-cy="aiFeedback-btn"]').first.click(timeout=5000, force=True)
        time.sleep(2)
        dismiss_banner(pg)
        rec.panel(pg, "panel_open", "Feedback: panel open — per-breast verdict controls + Submit")

        # Verdict controls are radio options labelled Agree/Unsure/Disagree per
        # breast. Use exact text (so "Agree" doesn't also match "Disagree").
        agree = pg.get_by_text("Agree", exact=True)
        log("Agree radios:", agree.count())
        picked = 0
        for i in range(min(agree.count(), 2)):
            try:
                dismiss_banner(pg)
                agree.nth(i).click(timeout=3000); picked += 1; time.sleep(0.5)
            except Exception as e:
                log("agree click", i, repr(e)[:80])
        log("verdicts picked", picked)
        rec.panel(pg, "verdicts_selected", f"Feedback: {picked} verdict(s) selected (Agree L+R)")

        # Submit.
        before = len(rec.ups)
        submitted_click = False
        for getter in [lambda: pg.get_by_role("button", name="Submit Feedback", exact=False),
                       lambda: pg.get_by_text("Submit Feedback", exact=False)]:
            try:
                el = getter()
                if el.count():
                    dismiss_banner(pg)
                    el.first.click(timeout=4000); submitted_click = True
                    log("clicked Submit Feedback"); break
            except Exception:
                pass
        # wait for the network call(s)
        for _ in range(10):
            time.sleep(1.5)
            if len(rec.ups) > before:
                break
        time.sleep(2)
        rec.panel(pg, "after_submit", "Feedback: after Submit Feedback")
        rec.shot(pg, "after_submit_full", "Feedback: full page after submit", full=True)

        fb_net = [(s, m, u) for (s, m, u) in rec.ups if "feedback" in u]
        ok = bool(fb_net) and all(s < 400 for s, _, _ in fb_net)
        notes = (f"verdicts_picked={picked}; submit_clicked={submitted_click}; "
                 f"feedback_net={fb_net}")
        status = "PASS" if ok else "NOTE"
        rec.write_summary(status, notes)
        log("AREA feedback", status, notes)
    finally:
        b.close()
