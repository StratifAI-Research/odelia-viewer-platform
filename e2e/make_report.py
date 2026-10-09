"""Compile area summaries + screenshots into a sectioned PDF walkthrough.

Format: one section per functional area (A, B, C, ...), each with a PASS/FAIL
status in the header and up to 3 stage screenshots. No descriptive prose — the
screenshots are the evidence; headless captures are too fragile to narrate.
"""
import json
import glob
import os
import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS_DIR = os.path.join(HERE, "shots")
OUT = os.path.join(HERE, "odelia_frontend_e2e_report.pdf")
DATE = datetime.date.today().isoformat()
MAX_SHOTS = 3

# (letter, summary-area-key, section title)
AREAS = [
    ("A", "auth", "Authentication"),
    ("B", "worklist", "Study list"),
    ("C", "viewer", "Image viewer"),
    ("D", "send_to_ai", "Send to AI"),
    ("E", "ai_results", "AI results"),
    ("F", "chat", "Chat"),
    ("G", "feedback", "Feedback"),
]

# Ordered filename-substrings selecting the stage shots to show (<= MAX_SHOTS).
CURATE = {
    "auth": ["keycloak", "landed"],
    "worklist": ["worklist", "columns"],
    "viewer": ["viewer_open", "after_scroll", "after_zoom"],
    "send_to_ai": ["step1_model", "step4_confirm", "08_after_trigger"],
    "ai_results": ["02_result", "03_result_full"],
    "chat": ["context_selected", "question_typed", "response_complete"],
    "feedback": ["panel_open", "verdicts_selected", "after_submit"],
}

PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch


def status_color(s):
    return {"PASS": colors.green, "FAIL": colors.red,
            "NOTE": colors.orange}.get(s, colors.grey)


def pick_shots(area, shots):
    """Select <= MAX_SHOTS stage shots: curated substrings first, then even fallback."""
    chosen, seen = [], set()
    for sub in CURATE.get(area, []):
        for sh in shots:
            if sub in os.path.basename(sh["path"]) and sh["path"] not in seen:
                chosen.append(sh); seen.add(sh["path"]); break
        if len(chosen) >= MAX_SHOTS:
            break
    if not chosen and shots:  # fallback: evenly spaced
        idx = sorted({0, len(shots) // 2, len(shots) - 1})
        chosen = [shots[i] for i in idx][:MAX_SHOTS]
    return chosen[:MAX_SHOTS]


def title_page(c, summaries):
    c.setFillColor(colors.HexColor("#1a2b4a"))
    c.rect(0, PAGE_H - 2.0 * inch, PAGE_W, 2.0 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN, PAGE_H - 1.1 * inch, "Odelia OHIF Viewer — Frontend Walkthrough")
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, PAGE_H - 1.55 * inch, "odv-193   |   " + DATE)

    y = PAGE_H - 2.7 * inch
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN, y, "Contents")
    y -= 22
    for letter_, key, title in AREAS:
        d = summaries.get(key, {"status": "MISSING"})
        st = d.get("status", "MISSING")
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(MARGIN, y, f"{letter_}.  {title}")
        c.setFillColor(status_color(st))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(MARGIN + 3.0 * inch, y, st)
        y -= 20
    c.showPage()


def shot_page(c, header, status, idx, total, img_path, footer):
    # header (status-colored)
    c.setFillColor(status_color(status))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, PAGE_H - MARGIN, f"{header}  —  {status}")
    c.setFillColor(colors.grey)
    c.setFont("Helvetica", 10)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN, f"{idx}/{total}")

    top = PAGE_H - MARGIN - 24
    bottom = MARGIN + 20
    box_w, box_h = PAGE_W - 2 * MARGIN, top - bottom
    try:
        ir = ImageReader(img_path)
        iw, ih = ir.getSize()
        scale = min(box_w / iw, box_h / ih)
        dw, dh = iw * scale, ih * scale
        dx = MARGIN + (box_w - dw) / 2
        dy = bottom + (box_h - dh) / 2
        c.drawImage(ir, dx, dy, dw, dh, preserveAspectRatio=True, mask="auto")
        c.setStrokeColor(colors.lightgrey)
        c.rect(dx, dy, dw, dh, fill=0, stroke=1)
    except Exception as e:
        c.setFillColor(colors.grey)
        c.setFont("Helvetica", 10)
        c.drawString(MARGIN, (top + bottom) / 2, f"[image unavailable: {e}]")

    c.setFillColor(colors.lightgrey)
    c.setFont("Helvetica", 7)
    c.drawString(MARGIN, MARGIN - 2, footer)
    c.showPage()


def main():
    summaries = {}
    for path in glob.glob(f"{SHOTS_DIR}/*_summary.json"):
        with open(path) as f:
            d = json.load(f)
        summaries[d["area"]] = d

    c = rl_canvas.Canvas(OUT, pagesize=letter)
    title_page(c, summaries)

    for letter_, key, title in AREAS:
        d = summaries.get(key)
        status = d.get("status", "MISSING") if d else "MISSING"
        header = f"{letter_}.  {title}"
        shots = pick_shots(key, d.get("shots", [])) if d else []
        if not shots:
            shot_page(c, header, status, 1, 1, "__none__", f"{key}: no screenshot captured")
            continue
        for i, sh in enumerate(shots, 1):
            shot_page(c, header, status, i, len(shots), sh["path"],
                      os.path.basename(sh["path"]))

    c.save()
    print("PDF written:", OUT)
    for letter_, key, title in AREAS:
        d = summaries.get(key, {"status": "MISSING"})
        n = len(pick_shots(key, d.get("shots", []))) if d else 0
        print(f"  {letter_}. {title}: {d.get('status','MISSING')} ({n} shots)")


if __name__ == "__main__":
    main()
