"""快速转换引擎：简单文档 -> Markdown。

在 marker 的 venv 中运行。依赖（由 build_marker_env.sh 追加安装）：
- pymupdf4llm : PDF -> Markdown（快速、纯文本/简单排版效果好，保留基本结构）
- markitdown  : docx/pptx/xlsx/html/epub 等 Office/网页文档 -> Markdown
- PyMuPDF(fitz): 图片抽取

CLI（与 marker convert_single.py 对齐，便于 Rust 统一调度）：
    guaxi_fast_convert.py <fpath> --output_dir <dir> [--progress_json]

输出契约（与 marker 保持一致）：把 Markdown 写到 <output_dir>/<stem>.md，
图片写到 <output_dir> 下（相对引用），并在 stdout 输出进度 JSON：
    {"type":"conversion_progress","current":..,"total":100,"message":".."}
    {"type":"conversion_result","output_file":"<abs md path>","output_exists":true}
"""

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import json
import sys
import time
import traceback


def _emit_progress(enabled, current, total, message):
    if not enabled:
        return
    print(
        json.dumps(
            {"type": "conversion_progress", "current": int(current),
             "total": int(total), "message": str(message)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _emit_result(output_file, exists):
    print(
        json.dumps(
            {"type": "conversion_result", "output_file": output_file,
             "output_exists": bool(exists)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _convert_pdf(fpath, out_dir, stem, progress):
    """PDF -> Markdown，使用 pymupdf4llm（带图片抽取）。"""
    import pymupdf4llm

    image_dir = out_dir  # 图片与 md 同目录，Markdown 内相对引用
    _emit_progress(progress, 40, 100, "解析 PDF 页面")

    md_text = pymupdf4llm.to_markdown(
        fpath,
        write_images=True,
        image_path=image_dir,
        image_format="png",
        dpi=150,
    )
    _emit_progress(progress, 85, 100, "写入 Markdown")
    return md_text


def _convert_with_markitdown(fpath, progress):
    """Office/HTML/EPUB 等 -> Markdown，使用 markitdown。"""
    from markitdown import MarkItDown

    _emit_progress(progress, 40, 100, "解析文档结构")
    md = MarkItDown()
    result = md.convert(fpath)
    _emit_progress(progress, 85, 100, "写入 Markdown")
    return result.text_content or ""


def _convert_plaintext(fpath, progress):
    _emit_progress(progress, 50, 100, "读取文本")
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(fpath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(fpath, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


PDF_EXTS = {".pdf"}
PLAIN_EXTS = {".txt", ".md", ".markdown", ".csv"}
MARKITDOWN_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                   ".html", ".htm", ".epub", ".rtf"}


def main(argv):
    parser = argparse.ArgumentParser(description="Guaxi 快速文档转换引擎")
    parser.add_argument("fpath", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--progress_json", action="store_true", default=False)
    args = parser.parse_args(argv[1:])

    progress = args.progress_json
    start = time.time()
    _emit_progress(progress, 0, 100, "开始转换")

    fpath = args.fpath
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(fpath))[0]
    md_path = os.path.join(out_dir, f"{stem}.md")
    ext = os.path.splitext(fpath)[1].lower()

    try:
        _emit_progress(progress, 15, 100, "选择快速解析器")
        if ext in PDF_EXTS:
            md_text = _convert_pdf(fpath, out_dir, stem, progress)
        elif ext in PLAIN_EXTS:
            md_text = _convert_plaintext(fpath, progress)
        elif ext in MARKITDOWN_EXTS:
            md_text = _convert_with_markitdown(fpath, progress)
        else:
            # 兜底：尝试 markitdown，失败则按纯文本
            try:
                md_text = _convert_with_markitdown(fpath, progress)
            except Exception:
                md_text = _convert_plaintext(fpath, progress)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        exists = os.path.exists(md_path)
        _emit_progress(progress, 100, 100, "转换完成" if exists else "转换结束但未生成文件")
        _emit_result(os.path.abspath(md_path), exists)
        sys.stderr.write(f"[guaxi_fast] saved {md_path} in {time.time()-start:.2f}s\n")
        return 0 if exists else 2
    except Exception as e:
        _emit_progress(progress, 100, 100, f"转换失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
