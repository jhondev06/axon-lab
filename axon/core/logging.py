import logging
import json
import sys

LOG_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def jmsg(event: str, **kw):
    return f"{event} | {json.dumps(kw, ensure_ascii=False, default=str)}"