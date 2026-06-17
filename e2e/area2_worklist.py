"""Area 2: worklist — study list renders; UKA_1/MR study listed; filters/columns present."""
import re
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
            # The study's instance count grows as AI result series (SR/heatmap)
            # accumulate, so don't assert a fixed number — just that the study row
            # shows a positive instance count.
            study_row = pg.locator("tr", has_text="UKA_1").first
            row_text = study_row.inner_text() if study_row.count() else ""
            # Instances is the last column, so the trailing number in the row is the
            # count (avoids matching the "1" in the UKA_1 MRN/accession cells).
            nums = re.findall(r"\d+", row_text)
            inst_count = int(nums[-1]) if nums else 0
            has_inst = inst_count > 0

            # column headers are styled divs, not <th>; check by header label text.
            header_labels = ["Patient Name", "MRN", "Study Date", "Description", "Modality", "Instances"]
            present_headers = [h for h in header_labels if h in body]
            filter_inputs = pg.locator("input").count()
            rec.shot(pg, "columns_filters",
                     f"Worklist: headers={present_headers}, filter inputs={filter_inputs}", full=True)

            checks = {
                "MRN UKA_1 present": has_uka,
                "MR modality present": has_mr,
                "study instance count shown (>0)": has_inst,
                "all column headers present": len(present_headers) == len(header_labels),
                "filter inputs present": filter_inputs > 0,
            }
            failed = [k for k, v in checks.items() if not v]
            if not failed:
                status = "PASS"
                notes = (f"All worklist checks passed: study MRN UKA_1, MR modality, "
                         f"{inst_count} instances, {len(present_headers)} column headers, "
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
                pass  # best-effort; non-fatal so the walkthrough always finishes and reports
        b.close()
    rec.write_summary(status, notes)
    log("AREA worklist", status, notes)


if __name__ == "__main__":
    run()
