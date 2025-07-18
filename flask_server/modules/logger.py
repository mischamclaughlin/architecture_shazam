# ./flask_server/modules/logger.py
import logging


def default_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with the specified name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
