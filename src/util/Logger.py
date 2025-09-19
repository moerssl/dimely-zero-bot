import os
import logging
import inspect
from datetime import datetime, timedelta

class Logger:
    _logger = logging.getLogger("CustomLogger")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _handler = None
    _log_dir = "logs"
    _default_prefix = "default"

    @staticmethod
    def _get_log_path(prefix):
        date_stamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{prefix}_{date_stamp}.log"
        return os.path.join(Logger._log_dir, filename)

    @staticmethod
    def _setup_logger(prefix):
        os.makedirs(Logger._log_dir, exist_ok=True)

        # Build log path
        log_path = Logger._get_log_path(prefix)

        if Logger._handler:
            Logger._logger.removeHandler(Logger._handler)
            Logger._handler.close()

        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        Logger._logger.addHandler(handler)
        Logger._handler = handler

    @staticmethod
    def set_log_prefix(prefix: str):
        Logger._setup_logger(prefix)

    @staticmethod
    def logText(message):
        if Logger._handler is None:
            Logger._setup_logger(Logger._default_prefix)

        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)[1].frame
        func_name = caller_frame.f_code.co_name

        full_message = f"{func_name}: {message}"
        Logger._logger.info(full_message)

    @staticmethod
    def log(message):
        if Logger._handler is None:
            Logger._setup_logger(Logger._default_prefix)

        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)[1].frame
        func_name = caller_frame.f_code.co_name

        args_info = inspect.getargvalues(caller_frame)
        args_str = ", ".join(
            f"{arg}={args_info.locals[arg]!r}" for arg in args_info.args if arg != "self"
        )

        full_message = f"{func_name}({args_str}): {message}"
        Logger._logger.info(full_message)
