"""Guaxi 单文件转换入口（等价于 `marker_single`，但带机器可读的进度协议）。

与上游 `marker/scripts/convert_single.py` 的差异（Guaxi 桌面端需要）：

1. `--progress_json`：把进度以单行 JSON 打到 stdout，供上层调度器解析；
2. `--output_dir` 直接作为产物目录，**不再**追加文件名子目录——上层已经把
   `{基目录}/{文件名}` 算好并传进来，再追加一层会导致产物路径对不上；
3. 结束时额外输出一条 `conversion_result`，告诉上层产物在哪里、是否真的写出来了；
4. Windows 可嵌入 Python 的多进程修复（见下）。

除以上四点外，其余行为与上游保持一致。
"""

import os
import sys

# --- Windows 多进程兼容修复（必须在导入 multiprocessing / torch 之前执行）---
# 托管 Python 为 embeddable 版：建 venv 后其 base python 目录被清理，
# 导致 Windows 上 multiprocessing spawn 使用不存在的 base-executable，报 WinError 3。
# 将 _base_executable 指向当前可用的 venv python 即可修复（有真实 base 时为无操作）。
if sys.platform == "win32":
    _be = getattr(sys, "_base_executable", None)
    if not _be or not os.path.exists(_be):
        sys._base_executable = sys.executable

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import json
import time
import traceback

import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output

configure_logging()
logger = get_logger()

# output_format -> 产物扩展名（与上游 save_output 的约定一致）。
OUTPUT_EXT = {"markdown": "md", "html": "html", "json": "json", "chunks": "json"}


def _emit_progress(enabled: bool, current: int, total: int, message: str):
    if not enabled:
        return
    print(
        json.dumps(
            {
                "type": "conversion_progress",
                "current": int(current),
                "total": int(total),
                "message": str(message),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


@click.command(cls=CustomClickPrinter, help="Convert a single PDF to markdown.")
@click.argument("fpath", type=str)
@click.option("--paginate_output", is_flag=True, default=False)
@click.option("--progress_json", is_flag=True, default=False)
@ConfigParser.common_options
def convert_single_cli(fpath: str, paginate_output: bool, progress_json: bool, **kwargs):
    paginate_output = bool(paginate_output)
    start = time.time()
    _emit_progress(progress_json, 0, 100, "开始转换")

    try:
        _emit_progress(progress_json, 5, 100, "加载模型")
        # marker 2.x 起模型由共享推理服务持有，此处为懒加载：
        # 首次调用可能触发 VLM 权重下载（数 GB），会在此停留较久。
        models = create_model_dict()

        _emit_progress(progress_json, 40, 100, "初始化转换器")
        kwargs["paginate_output"] = paginate_output
        config_parser = ConfigParser(kwargs)

        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        _emit_progress(progress_json, 55, 100, "正在转换")
        rendered = converter(fpath)

        _emit_progress(progress_json, 90, 100, "正在写入结果")
        base_name = config_parser.get_base_filename(fpath)

        out_folder = kwargs.get("output_dir")
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)
        else:
            out_folder = config_parser.get_output_folder(fpath)

        save_output(rendered, out_folder, base_name)

        output_format = kwargs.get("output_format", "markdown")
        output_file = os.path.join(
            out_folder, f"{base_name}.{OUTPUT_EXT.get(output_format, 'md')}"
        )
        if progress_json:
            print(
                json.dumps(
                    {
                        "type": "conversion_result",
                        "output_dir": out_folder,
                        "output_file": output_file,
                        "output_exists": os.path.exists(output_file),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        _emit_progress(progress_json, 100, 100, "转换完成")
        logger.info(f"Saved markdown to {out_folder}")
        logger.info(f"Total time: {time.time() - start}")
    except Exception as e:
        _emit_progress(progress_json, 100, 100, f"转换失败: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    convert_single_cli()
