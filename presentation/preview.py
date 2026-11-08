"""Approximate PNG preview of a .pptx made by build_deck.py, for layout QA without Keynote/PowerPoint.

Draws text boxes (word-wrapped with the real Helvetica Neue metrics), shapes, connectors, tables and
pictures. Flags text that overflows its box and table rows that grow past their nominal height.

    python presentation/preview.py presentation/out/SRP_Group7_Final.pptx --out presentation/out/preview
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pptx.util import Emu

DPI = 160
FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
_cache = {}


def font(pt, bold=False, italic=False):
    key = (pt, bold, italic)
    if key not in _cache:
        idx = {(False, False): 0, (True, False): 1, (False, True): 2, (True, True): 3}[(bool(bold), bool(italic))]
        try:
            _cache[key] = ImageFont.truetype(FONT_PATH, max(6, int(round(pt * DPI / 72))), index=idx)
        except Exception:
            _cache[key] = ImageFont.load_default()
    return _cache[key]


def px(emu):
    return int(round(Emu(emu).inches * DPI))


def colour(run_or_none, default=(26, 26, 26)):
    try:
        c = run_or_none.font.color
        if c is not None and c.type is not None:
            return tuple(int(str(c.rgb)[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return default


def para_layout(p, width_px):
    """Return list of lines: each line = list of (text, font, colour); plus metrics."""
    pPr = p._p.find(qn("a:pPr"))
    marL = int(pPr.get("marL", 0)) if pPr is not None else 0
    indent = int(pPr.get("indent", 0)) if pPr is not None else 0
    bullet = pPr is not None and pPr.find(qn("a:buChar")) is not None
    words = []
    size = 12
    from pptx.text.text import _Run
    for child in p._p:
        if child.tag == qn("a:br"):
            words.append(["\n", None, None, False])
            continue
        if child.tag != qn("a:r"):
            continue
        r = _Run(child, p)
        size = r.font.size.pt if r.font.size else size
        f = font(size, r.font.bold, r.font.italic)
        col = colour(r)
        for i, w in enumerate(r.text.split(" ")):
            words.append([w, f, col, i > 0])
    left = px(marL)
    avail = max(20, width_px - left)
    lines, cur, cur_w = [], [], 0
    for w, f, col, space_before in words:
        if w == "\n" and f is None:
            lines.append(cur)
            cur, cur_w = [], 0
            continue
        piece = (" " if cur and space_before else "") + w
        wlen = f.getlength(piece)
        if cur and cur_w + wlen > avail:
            lines.append(cur)
            cur, cur_w = [], 0
            piece = w
            wlen = f.getlength(piece)
        cur.append((piece, f, col))
        cur_w += wlen
    if cur or not lines:
        lines.append(cur)
    spc_after = 0
    if pPr is not None:
        sa = pPr.find(qn("a:spcAft"))
        if sa is not None and sa.find(qn("a:spcPts")) is not None:
            spc_after = int(sa.find(qn("a:spcPts")).get("val")) / 100.0
    line_h = size * 1.2 * DPI / 72
    return {"lines": lines, "left": left, "bullet_x": px(marL + indent), "bullet": bullet,
            "line_h": line_h, "after": spc_after * DPI / 72, "size": size, "align": p.alignment}


def draw_text_frame(d, tf, x, y, w, h, anchor=None, center_default=False):
    layouts = [para_layout(p, w) for p in tf.paragraphs]
    total = sum(len(L["lines"]) * L["line_h"] + L["after"] for L in layouts)
    va = tf.vertical_anchor
    if va is not None and "MIDDLE" in str(va):
        cy = y + (h - total) / 2
    elif va is not None and "BOTTOM" in str(va):
        cy = y + h - total
    else:
        cy = y
    for L in layouts:
        for k, line in enumerate(L["lines"]):
            lw = sum(f.getlength(t) for t, f, _ in line)
            al = str(L["align"]) if L["align"] is not None else ("CENTER" if center_default else "LEFT")
            if "CENTER" in al:
                cx = x + L["left"] + (w - L["left"] - lw) / 2
            elif "RIGHT" in al:
                cx = x + w - lw
            else:
                cx = x + L["left"]
            if k == 0 and L["bullet"] and line:
                d.text((x + L["bullet_x"], cy), "•", font=line[0][1], fill=line[0][2])
            for t, f, col in line:
                d.text((cx, cy), t, font=f, fill=col)
                cx += f.getlength(t)
            cy += L["line_h"]
        cy += L["after"]
    return total


def hexfill(shape_fill):
    try:
        if shape_fill.type == 1:          # solid
            return tuple(int(str(shape_fill.fore_color.rgb)[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return None


def render(pptx_path, out_dir):
    prs = Presentation(pptx_path)
    W, H = px(prs.slide_width), px(prs.slide_height)
    os.makedirs(out_dir, exist_ok=True)
    issues = []
    for n, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        for sh in slide.shapes:
            x, y = px(sh.left or 0), px(sh.top or 0)
            w, h = px(sh.width or 0), px(sh.height or 0)
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pic = Image.open(sh.image._blob if False else __import__("io").BytesIO(sh.image.blob)).convert("RGBA")
                pic = pic.resize((max(1, w), max(1, h)))
                img.paste(pic, (x, y), pic)
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.LINE or sh.__class__.__name__ == "Connector":
                x1, y1, x2, y2 = px(sh.begin_x), px(sh.begin_y), px(sh.end_x), px(sh.end_y)
                try:
                    c = tuple(int(str(sh.line.color.rgb)[i:i + 2], 16) for i in (0, 2, 4))
                except Exception:
                    c = (85, 85, 85)
                d.line([(x1, y1), (x2, y2)], fill=c, width=max(1, int(DPI / 110)))
                ln = sh.line._get_or_add_ln()
                if ln.find(qn("a:tailEnd")) is not None:
                    import math
                    ang = math.atan2(y2 - y1, x2 - x1)
                    s = DPI * 0.06
                    pts = [(x2, y2), (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45)),
                           (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))]
                    d.polygon(pts, fill=c)
                continue
            if sh.has_table if hasattr(sh, "has_table") else False:
                tbl = sh.table
                cy = y
                for ri, row in enumerate(tbl.rows):
                    need = px(row.height)
                    cells = []
                    cx = x
                    for ci, col in enumerate(tbl.columns):
                        cw = px(col.width)
                        cell = tbl.cell(ri, ci)
                        pad = px(cell.margin_left or 0)
                        lay = [para_layout(p, cw - 2 * pad) for p in cell.text_frame.paragraphs]
                        th = sum(len(L["lines"]) * L["line_h"] for L in lay) + px(cell.margin_top or 0) * 2
                        need = max(need, int(th) + 2)
                        cells.append((cx, cw, cell))
                        cx += cw
                    if need > px(row.height) + 2:
                        issues.append(f"slide {n}: table row {ri} grows {px(row.height)}->{need}px")
                    for cx, cw, cell in cells:
                        fill = hexfill(cell.fill) or (255, 255, 255)
                        d.rectangle([cx, cy, cx + cw, cy + need], fill=fill, outline=(229, 229, 229))
                        pad = px(cell.margin_left or 0)
                        draw_text_frame(d, cell.text_frame, cx + pad, cy, cw - 2 * pad, need)
                    cy += need
                if cy > H - px(Emu(914400 * 0.4)):
                    issues.append(f"slide {n}: table bottom at {cy / DPI:.2f} in")
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                fill = hexfill(sh.fill)
                try:
                    lc = tuple(int(str(sh.line.color.rgb)[i:i + 2], 16) for i in (0, 2, 4))
                except Exception:
                    lc = None
                d.rectangle([x, y, x + w, y + h], fill=fill, outline=lc, width=max(1, int(DPI / 110)) if lc else 0)
            if sh.has_text_frame and sh.text_frame.text.strip():
                tf = sh.text_frame
                ml = px(tf.margin_left or 0)
                mt = px(tf.margin_top or 0)
                used = draw_text_frame(d, tf, x + ml, y + mt, w - 2 * ml, h - 2 * mt,
                                       center_default=sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE)
                if used > h + 3:
                    issues.append(f"slide {n}: text overflows '{tf.text[:40]}...' ({used / DPI:.2f} > {h / DPI:.2f} in)")
                if y + used > H - 0.33 * DPI and y < H - 0.4 * DPI:
                    issues.append(f"slide {n}: text reaches the footer: '{tf.text[:40]}'")
        img.save(os.path.join(out_dir, f"slide_{n:02d}.png"))
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.pptx)[0] + "_preview"
    issues = render(a.pptx, out)
    print(f"previews in {out}")
    for i in issues:
        print("  ISSUE", i)


if __name__ == "__main__":
    main()
