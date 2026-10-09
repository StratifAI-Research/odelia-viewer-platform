"""Area 1: auth — keycloak login renders and succeeds, lands in app."""
import time
from playwright.sync_api import sync_playwright
from _helpers import browser, dismiss_banner, Recorder, BASE, log


def run():
    rec = Recorder("auth")
    with sync_playwright() as p:
        b, ctx = browser(p)
        pg = ctx.new_page()
        rec.wire(pg)
        status, notes = "FAIL", ""
        try:
            pg.goto(BASE + "/", wait_until="domcontentloaded", timeout=60000)
            pg.wait_for_selector("#username", timeout=30000)
            rec.shot(pg, "keycloak_form", "Auth: Keycloak login form rendered (#username/#password)", full=True)
            pg.fill("#username", "viewer")
            pg.fill("#password", "viewer")
            for sel in ["#kc-login", "input[type=submit]", "button[type=submit]"]:
                if pg.locator(sel).count():
                    pg.click(sel)
                    break
            try:
                pg.wait_for_load_state("networkidle", timeout=45000)
            except Exception as _e:
                log("networkidle wait: non-fatal", repr(_e)[:120])
            time.sleep(5)
            dismiss_banner(pg)
            # success = landed in app (worklist rows present, not on login form)
            on_login = pg.locator("#username").count() > 0
            has_rows = pg.locator("tr").count() > 0
            url = pg.url
            rec.shot(pg, "landed", f"Auth: post-login landing (url={url[-50:]}, rows={pg.locator('tr').count()})", full=True)
            if (not on_login) and has_rows:
                status = "PASS"
                notes = f"Logged in; landed at {url[-60:]} with {pg.locator('tr').count()} table rows."
            else:
                status = "FAIL"
                notes = f"Login did not land in app (on_login={on_login}, rows={has_rows}, url={url})."
        except Exception as e:
            notes = f"Exception: {repr(e)[:200]}"
            try:
                rec.shot(pg, "exception", f"Auth: state at exception: {repr(e)[:80]}", full=True)
            except Exception as _e:
                log("screenshot: non-fatal", repr(_e)[:120])
        b.close()
    rec.write_summary(status, notes)
    log("AREA auth", status, notes)


if __name__ == "__main__":
    run()
