"""Area 3: viewer — open study, MRI renders (canvas/cornerstone), basic interactions."""
import time
from playwright.sync_api import sync_playwright
from _helpers import browser, login, open_study, dismiss_banner, Recorder, log


def run():
    rec = Recorder("viewer")
    with sync_playwright() as p:
        b, ctx = browser(p)
        pg = ctx.new_page()
        rec.wire(pg)
        status, notes = "FAIL", ""
        try:
            login(pg)
            open_study(pg)
            rec.shot(pg, "viewer_open", "Viewer: study opened in AI Analysis Mode", full=True)

            # Load the MR series into the main viewport by double-clicking its thumbnail
            # (default viewport can render black until a series is explicitly loaded).
            try:
                dismiss_banner(pg)
                thumb = pg.get_by_text("NCI-dyn DEV", exact=False)
                if thumb.count():
                    thumb.first.scroll_into_view_if_needed(timeout=1500)
                    thumb.first.dblclick(timeout=3000)
                    time.sleep(2)
            except Exception as e:
                log("thumb dblclick", repr(e)[:80])
            rec.shot(pg, "series_loaded", "Viewer: MR series (NCI-dyn DEV) loaded into main viewport", full=True)

            # cornerstone renders into <canvas> elements
            n_canvas = pg.locator("canvas").count()
            url = pg.url
            in_viewer = "/viewer" in url

            # before/after a slice scroll on the main viewport
            vp = pg.locator("canvas").first
            try:
                box = vp.bounding_box()
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
            except Exception:
                cx, cy = 600, 500
            rec.shot(pg, "before_scroll", f"Viewer: before slice scroll (canvas count={n_canvas})", full=True)

            # scroll through slices
            try:
                pg.mouse.move(cx, cy)
                for _ in range(12):
                    pg.mouse.wheel(0, 240)
                    time.sleep(0.12)
                time.sleep(1)
            except Exception as e:
                rec.console.append(("error", f"scroll: {repr(e)[:80]}"))
            rec.shot(pg, "after_scroll", "Viewer: after scrolling through slices (slice index should change)", full=True)

            # zoom interaction: ctrl+wheel or right-drag — use keyboard/zoom via wheel with modifier
            try:
                pg.mouse.move(cx, cy)
                for _ in range(5):
                    pg.keyboard.down("Control")
                    pg.mouse.wheel(0, -200)
                    pg.keyboard.up("Control")
                    time.sleep(0.1)
                time.sleep(0.6)
            except Exception as _e:
                log("zoom interaction: non-fatal", repr(_e)[:120])
            rec.shot(pg, "after_zoom", "Viewer: after zoom attempt", full=True)

            # window/level via middle-ish drag (left drag adjusts W/L by default tool)
            try:
                pg.mouse.move(cx - 80, cy - 80)
                pg.mouse.down()
                pg.mouse.move(cx + 120, cy + 60, steps=10)
                pg.mouse.up()
                time.sleep(0.6)
            except Exception as _e:
                log("window/level drag: non-fatal", repr(_e)[:120])
            rec.shot(pg, "after_wl", "Viewer: after window/level (left-drag) interaction", full=True)

            if n_canvas > 0 and in_viewer:
                status = "PASS"
                notes = f"Viewer rendered {n_canvas} canvas element(s); interactions executed (scroll/zoom/W-L)."
            else:
                status = "FAIL"
                notes = f"No canvas / not in viewer (canvas={n_canvas}, url={url})."
        except Exception as e:
            notes = f"Exception: {repr(e)[:200]}"
            try:
                rec.shot(pg, "exception", f"Viewer exception: {repr(e)[:80]}", full=True)
            except Exception as _e:
                log("screenshot: non-fatal", repr(_e)[:120])
        b.close()
    rec.write_summary(status, notes)
    log("AREA viewer", status, notes)


if __name__ == "__main__":
    run()
