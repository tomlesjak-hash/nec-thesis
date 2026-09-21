#!/usr/bin/env python3
"""Build a paste-ready PDF of every qualifying GP alpha.

    python3 make_alpha_pdf.py

Reads gp_results/gp_results_topk.csv and writes Trexquant/GP_Alphas.pdf,
ordered by HELD-OUT valid_ir (not in-sample fitness — see the PDF's own note).
Also writes GP_Alphas.txt, because copying long expressions out of a PDF can
silently insert line breaks.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import sys

ROOT = Path(__file__).resolve().parents[1]
# Optional arg picks which run to render, e.g.
#     python3 make_alpha_pdf.py run1_topk.csv
# Defaults to the newest run's output.
_name = sys.argv[1] if len(sys.argv) > 1 else "gp_results_topk.csv"
CSV = ROOT / "gp_results" / _name
_tag = "" if _name == "gp_results_topk.csv" else "_" + _name.split("_")[0]
PDF = ROOT / f"GP_Alphas{_tag}.pdf"
TXT = ROOT / f"GP_Alphas{_tag}.txt"

ACCENT = colors.HexColor("#1a4f7a")
MUTED = colors.HexColor("#5b6770")
BG = colors.HexColor("#f4f6f8")


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main() -> None:
    d = pd.read_csv(CSV)
    q = d[d.qualifies].copy()
    if "robust_all_windows" not in q.columns:
        q["robust_all_windows"] = False
    if "test_ir" not in q.columns:
        q["test_ir"] = float("nan")
    # ROBUST FIRST. Positive in train AND valid AND test is the strongest
    # local evidence available that an alpha is not fitted to one regime.
    q = q.sort_values(["robust_all_windows", "valid_ir"],
                      ascending=[False, False]).reset_index(drop=True)

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=4,
                        textColor=ACCENT)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9.5,
                         textColor=MUTED, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13,
                        textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5,
                          leading=13.5, spaceAfter=6)
    note = ParagraphStyle("note", parent=body, fontSize=8.5, textColor=MUTED)
    code = ParagraphStyle("code", parent=ss["Code"], fontSize=6.6, leading=8.4,
                          alignment=TA_LEFT, backColor=BG, borderPadding=5,
                          spaceBefore=3, spaceAfter=6)

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, title="Trexquant GP Alphas",
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    S = []

    S.append(Paragraph("Trexquant — GP Alphas", h1))
    S.append(Paragraph(
        f"{len(q)} qualifying alphas. Run 2 config: 30 seeds x 600 population "
        f"x 40 generations = 738,000 expressions, tree depth capped at 5, "
        f"trained 2006-2018 (includes 2008), validated 2019-2020 (COVID), "
        f"tested 2021-2022. Source: {_name}.", sub))

    S.append(Paragraph("How to use this", h2))
    S.append(Paragraph(
        "<b>Ordered by three-window robustness first, then held-out IR — not "
        "by in-sample fitness.</b> In run 1 the rank correlation between "
        "in-sample fitness and out-of-sample IR was <b>-0.70</b>: negative, so "
        "the in-sample ranking was actively misleading. Rows marked YES under "
        "3-win are positive in training, validation AND the untouched "
        "2021-2022 test block. Those are the only ones with evidence from "
        "three independent periods. Start there.", body))
    S.append(Paragraph(
        "Every alpha here clears all four platform gates on LOCAL data: daily "
        "IR &gt; 0.07, turnover &lt; 0.5, stock count &gt; 160, positive "
        "held-out IR. Be aware that run 1 produced 21 such alphas and none of "
        "them cleared 0.07 on Trexsim itself — the local universe is ~370 "
        "large caps against the platform's top-1000, and 212 delisted names "
        "are missing. Treat these as candidates to test, not as results.", body))
    S.append(Paragraph(
        "<b>Two things that will silently break an alpha.</b> (1) The <i>- 1</i> "
        "after every cs_rank is deliberate: Trexsim's cs_rank returns [1,2] "
        "while the GP worked in [0,1]. Removing it changes the alpha. (2) Give "
        "slow mode a generous lookback — nested operators compound their "
        "windows, so an expression with w60 inside w20 needs far more history "
        "than 60 days. If fast and slow disagree, raise the lookback first.", body))
    S.append(Paragraph(
        "Copying from a PDF can insert stray line breaks into long expressions. "
        "GP_Alphas.txt next to this file has the identical strings as plain "
        "text — prefer it for pasting.", note))

    S.append(Paragraph("Summary", h2))
    rows = [["#", "valid IR", "test IR", "3-win", "daily IR", "TVR", "net IR", "stocks", "corr"]]
    for i, r in q.iterrows():
        ti = r["test_ir"]
        rows.append([str(i + 1), f"{r['valid_ir']:.2f}",
                     ("--" if pd.isna(ti) else f"{float(ti):.2f}"),
                     ("YES" if r["robust_all_windows"] else ""),
                     f"{r['ir_daily_vs_platform_0.07']:.4f}",
                     f"{r['tvr']:.3f}", f"{r['net_ir']:.2f}",
                     f"{r['numstk']:.0f}", f"{r['max_corr']:.2f}"])
    t = Table(rows, hAlign="LEFT",
              colWidths=[8*mm, 18*mm, 16*mm, 13*mm, 18*mm, 16*mm, 16*mm, 16*mm, 15*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0d8")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    S.append(t)
    S.append(Paragraph(
        "Turnover is worth watching. The tutorial's own worked examples run "
        "0.21-1.58; everything here sits well below that, which is the one "
        "shared trait of the run-1 alphas that failed live. If two look "
        "equally good, take the higher-TVR one.", note))
    S.append(PageBreak())

    txt = ["TREXQUANT GP ALPHAS — ordered by held-out valid_ir", "=" * 78, ""]
    for i, r in q.iterrows():
        blk = [Paragraph(f"Alpha {i + 1} &mdash; csv rank {int(r['rank'])}, "
                         f"seed {int(r['seed'])}", h2)]
        m = (f"held-out IR <b>{r['valid_ir']:.2f}</b> &nbsp;|&nbsp; "
             f"daily IR <b>{r['ir_daily_vs_platform_0.07']:.4f}</b> "
             f"(gate 0.07) &nbsp;|&nbsp; TVR <b>{r['tvr']:.3f}</b> "
             f"(limit 0.5) &nbsp;|&nbsp; net IR {r['net_ir']:.2f} "
             f"&nbsp;|&nbsp; stocks {r['numstk']:.0f} (min 160) "
             f"&nbsp;|&nbsp; drawdown {r['max_drawdown']*100:.1f}% "
             f"&nbsp;|&nbsp; max corr {r['max_corr']:.2f}"
             + ("" if pd.isna(r["test_ir"]) else
                f" &nbsp;|&nbsp; test IR {float(r['test_ir']):.2f}")
             + (" &nbsp;|&nbsp; <b>positive in all 3 windows</b>"
                if r["robust_all_windows"] else ""))
        blk.append(Paragraph(m, note))
        blk.append(Paragraph("<b>Paste into Trexsim:</b>", body))
        wrapped = "<br/>".join(esc(x) for x in
                               textwrap.wrap(str(r["trexsim"]), 150,
                                             break_long_words=True))
        blk.append(Paragraph(f"alpha = {wrapped}", code))
        blk.append(Paragraph(
            f"<i>GP form:</i> {esc(str(r['formula'])[:200])}", note))
        blk.append(Spacer(1, 4))
        S.append(KeepTogether(blk))

        txt += [f"--- Alpha {i+1}  (csv rank {int(r['rank'])}, seed {int(r['seed'])})",
                f"    valid_ir {r['valid_ir']:.3f} | daily_ir "
                f"{r['ir_daily_vs_platform_0.07']:.4f} | tvr {r['tvr']:.3f} | "
                f"net_ir {r['net_ir']:.2f} | numstk {r['numstk']:.0f} | "
                f"max_corr {r['max_corr']:.2f}",
                "", f"alpha = {r['trexsim']}", ""]

    doc.build(S)
    TXT.write_text("\n".join(txt))
    print(f"wrote {PDF}  ({len(q)} alphas)")
    print(f"wrote {TXT}")


if __name__ == "__main__":
    main()
