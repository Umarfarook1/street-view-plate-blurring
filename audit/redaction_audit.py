"""Audit the redaction in assets/plot_40_9.png.

The README claims the blur removes the readable part of a plate. This script checks
that claim against the figure the README actually publishes, so it needs no GPU, no
dataset and no trained weights: only the committed PNG.

Two measurements per frame:

1. What changed. The figure is a 4x2 grid, original on the left and redacted on the
   right. Diffing the two panels gives the exact pixels the redaction touched, which
   says whether the blur landed on the subject vehicle's own plate or on something
   else in the frame.
2. What survives. EasyOCR is run on each panel at the published scale, and on the
   redacted region alone at 1x through 8x with recall-friendly thresholds, which is
   what someone trying to recover the text would do.

Scope: this audits the committed figure, not the full-resolution pipeline output.
The source images and the trained weights are not in the repo (see .gitignore), so
a reader without them can still reproduce every number below.

    pip install easyocr pillow numpy
    python audit/redaction_audit.py

Writes audit/redaction_audit.txt and assets/redaction_closeup.png.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "assets" / "plot_40_9.png"
REPORT = ROOT / "audit" / "redaction_audit.txt"
CLOSEUP = ROOT / "assets" / "redaction_closeup.png"

# Row order in the figure. Each row is one test image.
FRAMES = ["sedan", "taxi", "hummer", "van"]

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def bands(mask, min_len, max_gap):
    """Contiguous runs of True, merging gaps up to max_gap, keeping runs >= min_len."""
    out, start, gap = [], None, 0
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if i - gap - start >= min_len:
                    out.append((start, i - gap))
                start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def clusters(mask, axis_gap=10):
    """Split a boolean change-mask into column-separated clusters."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        return []
    runs, start, prev = [], cols[0], cols[0]
    for x in cols[1:]:
        if x - prev > axis_gap:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    out = []
    for x1, x2 in runs:
        sub = mask[:, x1 : x2 + 1]
        ys = np.nonzero(sub.any(axis=1))[0]
        out.append((int(x1), int(x2), int(ys.min()), int(ys.max()), int(sub.sum())))
    return out


def main():
    import easyocr
    import torch

    rgb = np.array(Image.open(FIGURE).convert("RGB")).astype(int)
    ink = rgb.min(axis=2) < 245

    say(f"figure      : assets/{FIGURE.name}  {rgb.shape[1]}x{rgb.shape[0]}")
    say(f"easyocr     : {easyocr.__version__}   torch: {torch.__version__}")
    say("")

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    row_bands = bands(ink.any(axis=1), min_len=100, max_gap=3)
    assert len(row_bands) == len(FRAMES), row_bands

    closeups = []

    for name, (y1, y2) in zip(FRAMES, row_bands):
        strip = ink[y1:y2]
        col_blocks = bands(strip.any(axis=0), min_len=100, max_gap=3)
        assert len(col_blocks) == 2, col_blocks
        (lx1, lx2), (rx1, rx2) = col_blocks
        left = rgb[y1:y2, lx1:lx2]
        right = rgb[y1:y2, rx1:rx2]
        w = min(left.shape[1], right.shape[1])
        left, right = left[:, :w], right[:, :w]

        say(f"=== {name} ===")
        say(f"  panels      : original x{lx1}-{lx2}, redacted x{rx1}-{rx2}, rows y{y1}-{y2}")

        delta = np.abs(left - right).max(axis=2)
        for thr in (0, 8, 24):
            m = delta > thr
            say(f"  changed >{thr:<3d}: {int(m.sum()):7d} px  ({100 * m.sum() / m.size:6.3f}% of panel)")

        m = delta > 0
        if not m.any():
            say("  redaction   : nothing changed")
            say("")
            continue

        say("  regions     : (x1,x2,y1,y2,px) in panel coords")
        for c in clusters(m):
            say(f"                {c}")

        ys, xs = np.nonzero(m)
        bx1, bx2, by1, by2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pad = 8
        cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
        cx2, cy2 = min(w, bx2 + pad), min(left.shape[0], by2 + pad)

        # OCR the whole panel as published, both sides.
        for label, panel in (("original", left), ("redacted", right)):
            res = reader.readtext(panel.astype(np.uint8), detail=1)
            got = [(t, round(float(c), 3)) for _, t, c in res]
            say(f"  OCR panel {label}: {got if got else 'no text found'}")

        # OCR the redacted region alone, upscaled, thresholds lowered for recall.
        for label, panel in (("original", left), ("redacted", right)):
            crop = Image.fromarray(panel[cy1:cy2, cx1:cx2].astype(np.uint8))
            row = []
            for s in (1, 2, 4, 8):
                up = crop.resize((crop.width * s, crop.height * s), Image.LANCZOS)
                res = reader.readtext(np.array(up), detail=1, text_threshold=0.3, low_text=0.3)
                got = [(t, round(float(c), 3)) for _, t, c in res if t.strip()]
                row.append(f"{s}x={got if got else '[]'}")
            say(f"  OCR crop  {label}: " + "  ".join(row))
            if name == "van":
                closeups.append(crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS))
        say("")

    if len(closeups) == 2:
        a, b = closeups
        gap, h = 16, max(a.height, b.height)
        out = Image.new("RGB", (a.width + gap + b.width, h), "white")
        out.paste(a, (0, 0))
        out.paste(b, (a.width + gap, 0))
        out.save(CLOSEUP)
        say(f"wrote {CLOSEUP.relative_to(ROOT).as_posix()}  (van plate, original left, redacted right, 4x)")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
