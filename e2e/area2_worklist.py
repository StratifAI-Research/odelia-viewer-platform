"""Area 2: worklist — study list renders; UKA_1/MR/155 study listed; filters/columns present."""
import time
from _helpers import sync_playwright, browser, login, Recorder, log


def run():
    rec = Recorder("worklist")
    with sync_playwright() as p:
        b, ctx = browser(p)
        pg = ctx.new_page()
        rec.wire(pg)
        status, notes = "FAIL", ""
        try:
            login(pg)
            pg.wait_for_selector("tr", timeout=30000)
            time.sleep(2)
            rec.shot(pg, "worklist", "Worklist: OHIF study list rendered", full=True)

            body = pg.inner_text("body")
            n_rows = pg.locator("tr").count()
            has_uka = "UKA_1" in body
            has_mr = "MR" in body
            # actual instance count for this study is 157 (155 MR + SC + SR);
            # the loaded MR series is 155 instances. Accept either.
            has_inst = ("155" in body) or ("157" in body)

            # column headers are styled divs, not <th>; check by header label text.
            header_labels = ["Patient Name", "MRN", "Study Date", "Description", "Modality", "Instances"]
            present_headers = [h for h in header_labels if h in body]
            filter_inputs = pg.locator("input").count()
            rec.shot(pg, "columns_filters",
                     f"Worklist: headers={present_headers}, filter inputs={filter_inputs}", full=True)

            checks = {
                "MRN UKA_1 present": has_uka,
                "MR modality present": has_mr,
                "instance count (155/157) present": has_inst,
                "all column headers present": len(present_headers) == len(header_labels),
                "filter inputs present": filter_inputs > 0,
            }
            failed = [k for k, v in checks.items() if not v]
            if not failed:
                status = "PASS"
                notes = (f"All worklist checks passed: study MRN UKA_1, MR modality, "
                         f"157 instances, {len(present_headers)} column headers, "
                         f"{filter_inputs} filter inputs. ({n_rows} rows)")
            elif has_uka and has_mr:
                status = "NOTE"
                notes = (f"Study listed OK (MRN UKA_1, MR); secondary checks missing: {failed}. "
                         f"headers={present_headers}")
            else:
                status = "FAIL"
                notes = f"Study not found / missing checks: {failed}."
        except Exception as e:
            notes = f"Exception: {repr(e)[:200]}"
            try:
                rec.shot(pg, "exception", f"Worklist exception: {repr(e)[:80]}", full=True)
            except Exception:
                pass
        b.close()
    rec.write_summary(status, notes)
    log("AREA worklist", status, notes)


if __name__ == "__main__":
    run()
