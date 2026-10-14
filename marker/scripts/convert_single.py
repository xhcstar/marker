import os

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


def _emit_progress(enabled: bool, current: int, total: int, message: str):
    """把进度以单行 JSON 打到 stdout，供上层（Guaxi 文档转换调度器）逐行解析。

    仅在启用 --progress_json 时输出，避免污染正常 CLI 的 stdout。
    """
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
    start = time.time()
    _emit_progress(progress_json, 0, 100, "开始转换")

    try:
        _emit_progress(progress_json, 5, 100, "加载模型")
        # 注意：marker 2.x 起模型由共享推理服务持有，这里是懒加载——
        # 首次调用可能触发 VLM 权重下载（数 GB），会停留较久。
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

        _emit_progress(progress_json, 45, 100, "正在转换")
        rendered = converter(fpath)

        _emit_progress(progress_json, 90, 100, "正在写入结果")
        # 上游 get_output_folder() 会在 --output_dir 之后追加文件名子目录；
        # Guaxi 侧传入的 --output_dir 已经带上了这一层，因此显式给定时直接用它。
        out_folder = kwargs.get("output_dir")
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)
        else:
            out_folder = config_parser.get_output_folder(fpath)
        save_output(rendered, out_folder, config_parser.get_base_filename(fpath))

        _emit_progress(progress_json, 100, 100, "转换完成")
        logger.info(f"Saved markdown to {out_folder}")
        logger.info(f"Total time: {time.time() - start}")
    except Exception as e:
        _emit_progress(progress_json, 100, 100, f"转换失败: {e}")
        traceback.print_exc()
        raise
