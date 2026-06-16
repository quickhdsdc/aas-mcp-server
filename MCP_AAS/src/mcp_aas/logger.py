
import os
import sys
from datetime import datetime

from loguru import logger as _logger

from mcp_aas.settings import LOG_DIR

_print_level = "INFO"


def define_log_level(print_level: str = "INFO", logfile_level: str = "DEBUG", name: str | None = None):
    global _print_level
    _print_level = print_level

    formatted_date = datetime.now().strftime("%Y%m%d%H%M%S")
    log_name = f"{name}_{formatted_date}" if name else formatted_date

    _logger.remove()
    # IMPORTANT: stderr only — stdout is reserved for the MCP stdio transport.
    _logger.add(
        sys.stderr,
        level=print_level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        _logger.add(os.path.join(LOG_DIR, f"{log_name}.log"), level=logfile_level, enqueue=True)
    except Exception:
        # A read-only filesystem must not stop the server from starting.
        pass
    return _logger


logger = define_log_level()
