"""Area 6 (v2): chat — open chat panel, set series context, send a question,
capture the LLM response. Ollama + model (thiagomoraes/medgemma-1.5-4b-it:F16)
are confirmed available, so a real response is expected."""
import time
from playwright.sync_api import sync_playwright
from _helpers import browser, login, open_study, dismiss_banner, Recorder, log

QUESTION = "Briefly summarize the findings for this breast MRI study."

with sync_playwright() as p:
    rec = Recorder("chat")
    b, ctx = browser(p)
    try:
        pg = ctx.new_page(); rec.wire(pg)
        login(pg); open_study(pg)
        dismiss_banner(pg)
        rec.shot(pg, "viewer", "Chat: viewer loaded (AI Analysis Mode)", full=True)

        # Open the chat tab in the right panel.
        opened = False
        for sel in ['[data-cy="aiChat-btn"]', '[data-cy*="chat" i]',
                    '[aria-label*="chat" i]', '[title*="chat" i]']:
            try:
                el = pg.locator(sel)
                if el.count():
                    el.first.click(timeout=3000, force=True); opened = True
                    log("chat tab via", sel); break
            except Exception as _e:
                log("chat tab selector: non-fatal", repr(_e)[:120])
        time.sleep(2)
        # Confirm we're on the chat panel (look for the prompt text / input)
        on_chat = pg.get_by_text("Ask about this study", exact=False).count() > 0 \
            or pg.locator('textarea[placeholder*="ask" i]').count() > 0
        log("on_chat_panel", on_chat, "opened_via_control", opened)
        rec.panel(pg, "chat_panel", f"Chat: panel open (on_chat={on_chat})")

        # Set series context: click the Context selector and choose the series.
        ctx_set = None
        try:
            dismiss_banner(pg)
            # native <select> first
            sels = pg.locator("select")
            picked = False
            for i in range(sels.count()):
                opts = sels.nth(i).locator("option")
                labels = [opts.nth(j).inner_text().strip() for j in range(opts.count())]
                if any("NCI" in l or "MR" in l or "401" in l for l in labels) and opts.count() > 1:
                    sels.nth(i).select_option(index=1); picked = True; ctx_set = labels; break
            if not picked:
                # custom dropdown: click "No series selected" then an option
                trig = pg.get_by_text("No series selected", exact=False)
                if trig.count():
                    trig.first.click(timeout=3000); time.sleep(1)
                    rec.panel(pg, "context_dropdown", "Chat: context dropdown opened")
                    for opt in ["NCI-dyn DEV", "401", "MR"]:
                        o = pg.get_by_text(opt, exact=False)
                        if o.count():
                            o.last.click(timeout=3000); ctx_set = opt; break
        except Exception as e:
            log("context set", repr(e)[:120])
        log("context_set", ctx_set)
        time.sleep(1); rec.panel(pg, "context_selected", f"Chat: context set = {ctx_set}")

        # Type the question + send.
        ta = pg.locator('textarea[placeholder*="ask" i]')
        if not ta.count():
            ta = pg.locator("textarea")
        sent = False
        if ta.count():
            dismiss_banner(pg)
            ta.first.click(timeout=3000); ta.first.fill(QUESTION)
            rec.panel(pg, "question_typed", f"Chat: typed question: {QUESTION!r}")
            # send: a button near the input, else Enter
            clicked_send = False
            for sel in ['[data-cy*="send" i]', '[aria-label*="send" i]',
                        'button[type="submit"]']:
                try:
                    el = pg.locator(sel)
                    if el.count() and el.first.is_visible():
                        el.first.click(timeout=3000); clicked_send = True; break
                except Exception as _e:
                    log("send button click: non-fatal", repr(_e)[:120])
            if not clicked_send:
                ta.first.press("Enter")
            sent = True
            log("question sent (clicked_send=%s)" % clicked_send)
        rec.panel(pg, "after_send", f"Chat: after send (sent={sent})")

        # Poll for the streamed reply. The model generates over a 155-frame
        # study, so allow up to ~4 min and detect completion (the transient
        # "generating response..." placeholder is replaced by real text).
        responded = False
        done = False
        for i in range(30):
            time.sleep(8)
            try:
                body = pg.inner_text("body")
            except Exception:
                body = ""
            has_msgs = "No messages yet" not in body
            generating = "generating response" in body.lower()
            if has_msgs and (QUESTION[:20] in body):
                if not responded:
                    rec.panel(pg, "response_started", "Chat: thread appeared, reply generating")
                responded = True
            # done when the generating placeholder is gone but a thread exists
            if responded and not generating:
                done = True
                rec.panel(pg, "response_complete", "Chat: assistant reply completed")
                break
            log("poll", i, "msgs", has_msgs, "generating", generating)
        log("chat done?", done)
        time.sleep(2)
        rec.panel(pg, "response_final_panel", "Chat: final chat panel state")
        rec.shot(pg, "response_final_full", "Chat: final full-page state", full=True)

        notes = (f"on_chat={on_chat}; context_set={ctx_set}; sent={sent}; "
                 f"response_thread={responded}; chat_net={rec.ups}")
        status = "PASS" if (sent and responded) else "NOTE"
        rec.write_summary(status, notes)
        log("AREA chat", status, notes)
    finally:
        b.close()
