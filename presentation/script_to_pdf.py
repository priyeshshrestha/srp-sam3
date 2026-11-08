"""Render presentation/speaker_script.md to a readable PDF (one slide per section)."""
import re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

for name, idx in (("HN", 0), ("HN-B", 1), ("HN-I", 2), ("HN-BI", 3)):
    pdfmetrics.registerFont(TTFont(name, "/System/Library/Fonts/HelveticaNeue.ttc", subfontIndex=idx))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("HN", normal="HN", bold="HN-B", italic="HN-I", boldItalic="HN-BI")

INK, GREY, GREEN = HexColor("#1a1a1a"), HexColor("#777777"), HexColor("#2a4d38")
title = ParagraphStyle("t", fontName="HN-B", fontSize=22, leading=27, textColor=INK, spaceAfter=6)
sub = ParagraphStyle("s", fontName="HN", fontSize=11, leading=15, textColor=GREY, spaceAfter=4)
head = ParagraphStyle("h", fontName="HN-B", fontSize=16, leading=20, textColor=GREEN, spaceBefore=4, spaceAfter=10)
body = ParagraphStyle("b", fontName="HN", fontSize=13.5, leading=20.5, textColor=INK, spaceAfter=9)
stage = ParagraphStyle("d", parent=body, fontName="HN-I", textColor=GREY, fontSize=11.5, leading=16)

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = t.replace("[PAUSE]", '<font color="#b45309"><b>[PAUSE]</b></font>')
    return t

src, out = sys.argv[1], sys.argv[2]
blocks = open(src).read().split("\n## ")
story = []
intro = blocks[0].splitlines()
story.append(Paragraph(inline(intro[0].lstrip("# ")), title))
for line in intro[1:]:
    if line.strip():
        story.append(Paragraph(inline(line), sub))
for b in blocks[1:]:
    lines = b.splitlines()
    story.append(PageBreak())
    story.append(Paragraph(inline(lines[0]), head))
    for para in "\n".join(lines[1:]).split("\n\n"):
        p = para.strip()
        if not p:
            continue
        p = " ".join(p.split("\n"))
        is_stage = p.startswith("[") and p.endswith("]") and "PAUSE" not in p
        story.append(Paragraph(inline(p), stage if is_stage else body))

def footer(c, d):
    c.saveState(); c.setFont("HN", 8.5); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, "Group 7 · Few-Shot Learning for Biodiversity Assessment · speaker script")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(d.page)); c.restoreState()

SimpleDocTemplate(out, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                  bottomMargin=2 * cm, title="Speaker script").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", out)
