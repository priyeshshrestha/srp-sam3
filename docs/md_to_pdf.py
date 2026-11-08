"""Render a simple Markdown document (headings, paragraphs, bullet / numbered lists, pipe tables,
![caption](path) images) to an A4 PDF in the deck's style.

    python docs/md_to_pdf.py docs/group_update.md presentation/out/Group7_Project_Status.pdf ["footer label"]

Image paths are relative to the repository root. Fonts: Helvetica Neue (macOS).
"""
import os
import re
import sys

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name, idx in (("HN", 0), ("HN-B", 1), ("HN-I", 2), ("HN-BI", 3)):
    pdfmetrics.registerFont(TTFont(name, "/System/Library/Fonts/HelveticaNeue.ttc", subfontIndex=idx))
registerFontFamily("HN", normal="HN", bold="HN-B", italic="HN-I", boldItalic="HN-BI")

INK, GREY, GREEN, RULE, ROWBG = (HexColor(c) for c in ("#1a1a1a", "#666666", "#2a4d38", "#b0b0b0", "#e5e5e5"))
TITLE = ParagraphStyle("t", fontName="HN-B", fontSize=21, leading=26, textColor=INK, spaceAfter=8)
SUB = ParagraphStyle("s", fontName="HN", fontSize=10.5, leading=14, textColor=GREY, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="HN-B", fontSize=15, leading=19, textColor=GREEN, spaceBefore=12, spaceAfter=7, keepWithNext=1)
H3 = ParagraphStyle("h3", fontName="HN-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("b", fontName="HN", fontSize=10.3, leading=14.6, textColor=INK, spaceAfter=6)
CELL = ParagraphStyle("c", fontName="HN", fontSize=8.8, leading=11.2, textColor=INK)
CELLH = ParagraphStyle("ch", parent=CELL, fontName="HN-B")
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10.2, textColor=INK, backColor=HexColor("#f2f2f2"),
                      borderPadding=5, spaceBefore=4, spaceAfter=10, leftIndent=4)
CAP = ParagraphStyle("cap", fontName="HN-I", fontSize=8.6, leading=11.5, textColor=GREY, alignment=TA_CENTER,
                     spaceBefore=3, spaceAfter=10)
WIDTH = A4[0] - 4.2 * cm


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', t)
    return t


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    n = len(rows[0])
    data = [[Paragraph(inline(c), CELLH if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    first = 0.34 if n > 2 else 0.3
    widths = [WIDTH * first] + [WIDTH * (1 - first) / (n - 1)] * (n - 1)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ROWBG), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
    return [t, Spacer(1, 8)]


def image(caption, path):
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return [Paragraph(f"[missing figure: {path}]", CAP)]
    w, h = PILImage.open(p).size
    width = WIDTH if w / h > 1.6 else WIDTH * 0.72
    return [KeepTogether([Image(p, width=width, height=width * h / w), Paragraph(inline(caption), CAP)])]


def build(src, out, label="project status"):
    lines = open(src, encoding="utf-8").read().splitlines()
    story, i, para, first_h2 = [], 0, [], True

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY))
            para.clear()

    while i < len(lines):
        l = lines[i]
        if l.startswith("# "):
            flush(); story.append(Paragraph(inline(l[2:]), TITLE))
        elif l.startswith("## "):
            flush()
            if not first_h2 and l[3:].startswith(("1.", "Glossary")):
                story.append(PageBreak())
            first_h2 = False
            story.append(Paragraph(inline(l[3:]), H2))
        elif l.startswith("### "):
            flush(); story.append(Paragraph(inline(l[4:]), H3))
        elif l.startswith("```"):
            flush()
            block, i = [], i + 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(block), CODE))
        elif l.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            story += table(block); continue
        elif re.match(r"!\[(.*)\]\((.+)\)", l):
            flush()
            m = re.match(r"!\[(.*)\]\((.+)\)", l)
            story += image(m.group(1), m.group(2))
        elif re.match(r"^(- |\d+\. )", l):
            flush()
            numbered = bool(re.match(r"^\d+\. ", l))
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                text = re.sub(r"^(- |\d+\. )", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    text += " " + lines[i].strip(); i += 1
                items.append(ListItem(Paragraph(inline(text), BODY), leftIndent=14))
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                                      bulletFontName="HN", bulletFontSize=9.5 if numbered else 7, leftIndent=14,
                                      bulletColor=GREY))
            continue
        elif not l.strip():
            flush()
        elif story and len(story) == 1 and not para:
            story.append(Paragraph(inline(l), SUB))         # the line right under the title
        else:
            para.append(l.strip())
        i += 1
    flush()

    def footer(c, d):
        c.saveState(); c.setFont("HN", 8); c.setFillColor(GREY)
        c.drawString(2.1 * cm, 1.2 * cm, f"Group 7 · Few-Shot Learning for Biodiversity Assessment · {label}")
        c.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(d.page)); c.restoreState()

    SimpleDocTemplate(out, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm,
                      bottomMargin=2 * cm, title="Group 7 project status").build(story, onFirstPage=footer,
                                                                                  onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], *sys.argv[3:4])
