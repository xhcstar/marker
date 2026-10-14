"""文档复杂度探针：判定应走 fast / marker / mineru 引擎。

在 marker 的 venv 中运行（可用 PyMuPDF / pypdfium2）。仅做轻量采样分析，
目标 1~2 秒内给出建议。输出单行 JSON 到 stdout：

    {"type": "probe_result",
     "recommended": "fast" | "marker" | "mineru",
     "scores": {...},
     "reason": "..."}

判定逻辑（理工科复杂文档 -> mineru；普通排版 -> marker；纯文本/简单 -> fast）：
- 非 PDF：docx/pptx/xlsx/html/epub/txt/md -> fast；图片 -> marker（需 OCR）。
- PDF：采样若干页，统计
    * 扫描件（几乎无文本但整页图片）-> mineru（OCR + 版面）
    * 公式密度高 / 表格线多 / 代码块多 / 图表多 -> mineru
    * 有一定版面复杂度（图片、多栏）-> marker
    * 纯文本、单栏、无公式表格 -> fast
"""

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import json
import re
import sys

# 采样与阈值参数（可按需调整）
MAX_SAMPLE_PAGES = 8
FORMULA_DENSITY_STRONG = 0.015   # 每字符中数学符号占比
TABLE_LINES_STRONG = 12          # 单页水平/垂直短线数量
IMAGE_COVERAGE_SCANNED = 0.55    # 图片覆盖率视作扫描页
CODE_LINES_STRONG = 6            # 单页疑似代码行

FAST_EXTS = {".txt", ".md", ".markdown", ".docx", ".doc", ".pptx", ".ppt",
             ".xlsx", ".xls", ".html", ".htm", ".epub", ".csv", ".rtf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# 常见数学/公式 Unicode 区段
_MATH_RANGES = [
    (0x2200, 0x22FF),  # Mathematical Operators
    (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
    (0x27C0, 0x27EF),  # Misc Mathematical Symbols-A
    (0x2980, 0x29FF),  # Misc Mathematical Symbols-B
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    (0x0370, 0x03FF),  # Greek（公式常用）
    (0x2080, 0x209F),  # 下标
    (0x2070, 0x207F),  # 上标
]
_MATH_CHARS = set("=≈≠≤≥±∓×÷√∛∜∫∮∑∏∂∇∆∞→←↔⇒⇔∈∉⊂⊃∪∩∀∃∅·∝∠°′″")

_LATEXISH = re.compile(r"(\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\theta|\\partial|\\begin\{|\$[^$]+\$)")
_CODEISH = re.compile(r"(def |class |import |#include|public |private |=>|::|\{\s*$|;\s*$|函数|返回)")


def _emit(recommended, scores, reason):
    print(
        json.dumps(
            {"type": "probe_result", "recommended": recommended,
             "scores": scores, "reason": reason},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _is_math_char(ch):
    if ch in _MATH_CHARS:
        return True
    o = ord(ch)
    for lo, hi in _MATH_RANGES:
        if lo <= o <= hi:
            return True
    return False


def _probe_pdf(fpath):
    scores = {
        "pages_sampled": 0,
        "formula_density": 0.0,
        "avg_table_lines": 0.0,
        "avg_image_coverage": 0.0,
        "code_lines": 0,
        "scanned_pages": 0,
        "latexish_hits": 0,
        "engine": "pdf",
    }

    # 优先 PyMuPDF（字体/线条/图片信息最全）
    try:
        import fitz  # PyMuPDF
        return _probe_pdf_fitz(fpath, scores)
    except Exception:
        pass

    # 回退：pypdfium2 仅取文本，做粗略公式/代码判断
    try:
        return _probe_pdf_pdfium(fpath, scores)
    except Exception as e:
        # 无法探测 PDF -> 保守走 marker
        return "marker", scores, f"PDF 探测失败，回退 marker: {e}"


def _probe_pdf_fitz(fpath, scores):
    import fitz

    doc = fitz.open(fpath)
    total_pages = doc.page_count
    if total_pages == 0:
        return "fast", scores, "空 PDF"

    # 均匀采样
    step = max(1, total_pages // MAX_SAMPLE_PAGES)
    idxs = list(range(0, total_pages, step))[:MAX_SAMPLE_PAGES]

    total_chars = 0
    math_chars = 0
    total_table_lines = 0
    total_img_cov = 0.0
    code_lines = 0
    scanned = 0
    latexish = 0

    for i in idxs:
        page = doc.load_page(i)
        rect = page.rect
        page_area = max(1.0, rect.width * rect.height)

        text = page.get_text("text") or ""
        total_chars += len(text)
        for ch in text:
            if _is_math_char(ch):
                math_chars += 1
        latexish += len(_LATEXISH.findall(text))

        for line in text.splitlines():
            if _CODEISH.search(line):
                code_lines += 1

        # 表格线：统计近水平/近垂直的绘图线段
        try:
            drawings = page.get_drawings()
            line_ct = 0
            for d in drawings:
                for item in d.get("items", []):
                    if item[0] == "l":  # line
                        p1, p2 = item[1], item[2]
                        dx, dy = abs(p1.x - p2.x), abs(p1.y - p2.y)
                        if dx > 20 and dy < 2:
                            line_ct += 1
                        elif dy > 20 and dx < 2:
                            line_ct += 1
                    elif item[0] == "re":  # rectangle 也常构成表格
                        line_ct += 1
            total_table_lines += line_ct
        except Exception:
            pass

        # 图片覆盖率
        try:
            img_area = 0.0
            for img in page.get_image_info():
                bbox = img.get("bbox")
                if bbox:
                    w = abs(bbox[2] - bbox[0])
                    h = abs(bbox[3] - bbox[1])
                    img_area += w * h
            cov = min(1.0, img_area / page_area)
            total_img_cov += cov
            if len(text.strip()) < 40 and cov > IMAGE_COVERAGE_SCANNED:
                scanned += 1
        except Exception:
            pass

    n = max(1, len(idxs))
    doc.close()

    scores["pages_sampled"] = len(idxs)
    scores["formula_density"] = round(math_chars / max(1, total_chars), 5)
    scores["avg_table_lines"] = round(total_table_lines / n, 2)
    scores["avg_image_coverage"] = round(total_img_cov / n, 3)
    scores["code_lines"] = code_lines
    scores["scanned_pages"] = scanned
    scores["latexish_hits"] = latexish

    return _decide_pdf(scores)


def _probe_pdf_pdfium(fpath, scores):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(fpath)
    total_pages = len(pdf)
    if total_pages == 0:
        return "fast", scores, "空 PDF"

    step = max(1, total_pages // MAX_SAMPLE_PAGES)
    idxs = list(range(0, total_pages, step))[:MAX_SAMPLE_PAGES]

    total_chars = 0
    math_chars = 0
    code_lines = 0
    latexish = 0
    empty_text_pages = 0

    for i in idxs:
        page = pdf[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range() or ""
        total_chars += len(text)
        for ch in text:
            if _is_math_char(ch):
                math_chars += 1
        latexish += len(_LATEXISH.findall(text))
        for line in text.splitlines():
            if _CODEISH.search(line):
                code_lines += 1
        if len(text.strip()) < 40:
            empty_text_pages += 1

    n = max(1, len(idxs))
    scores["pages_sampled"] = len(idxs)
    scores["formula_density"] = round(math_chars / max(1, total_chars), 5)
    scores["code_lines"] = code_lines
    scores["latexish_hits"] = latexish
    scores["scanned_pages"] = empty_text_pages  # 无法测图片覆盖，用无文本页近似

    # pdfium 无线条/图片信息，表格线保持 0，判定偏保守
    return _decide_pdf(scores, limited=True)


def _decide_pdf(scores, limited=False):
    fd = scores["formula_density"]
    tl = scores["avg_table_lines"]
    ic = scores["avg_image_coverage"]
    cl = scores["code_lines"]
    scanned = scores["scanned_pages"]
    latex = scores["latexish_hits"]
    sampled = max(1, scores["pages_sampled"])

    # 扫描件：整篇几乎无文本 -> mineru（版面 + OCR 最强）
    if scanned >= sampled * 0.6:
        return "mineru", scores, "疑似扫描件（多页无文本/高图片覆盖），使用强引擎 OCR"

    # 理工科复杂特征：公式 / 表格线 / 代码 / LaTeX
    strong_signals = 0
    reasons = []
    if fd >= FORMULA_DENSITY_STRONG or latex >= 3:
        strong_signals += 1
        reasons.append("公式密集")
    if tl >= TABLE_LINES_STRONG:
        strong_signals += 1
        reasons.append("表格线多")
    if cl >= CODE_LINES_STRONG:
        strong_signals += 1
        reasons.append("代码块多")

    if strong_signals >= 1:
        return "mineru", scores, "检测到复杂结构（%s），使用强引擎" % "、".join(reasons)

    # 普通版面复杂度：有图片或有一定表格线 -> marker
    if ic > 0.12 or tl >= 3:
        return "marker", scores, "含图片/轻量表格的常规版面，使用 marker"

    # 纯文本 PDF：limited 模式下无线条信息也允许 fast
    return "fast", scores, "纯文本单栏 PDF，使用快速引擎"


def main(argv):
    if len(argv) < 2:
        _emit("marker", {}, "缺少文件参数，回退 marker")
        return 0

    fpath = argv[1]
    ext = os.path.splitext(fpath)[1].lower()

    try:
        if ext == ".pdf":
            rec, scores, reason = _probe_pdf(fpath)
        elif ext in IMAGE_EXTS:
            rec, scores, reason = "marker", {"engine": "image"}, "图片需 OCR，使用 marker"
        elif ext in FAST_EXTS:
            rec, scores, reason = "fast", {"engine": "office/text"}, "结构化/文本文档，使用快速引擎"
        else:
            rec, scores, reason = "marker", {"engine": "unknown"}, "未知类型，保守使用 marker"
    except Exception as e:
        rec, scores, reason = "marker", {"engine": "error"}, f"探测异常，回退 marker: {e}"

    _emit(rec, scores, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
