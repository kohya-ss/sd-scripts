import threading
from typing import *
import logging

def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()

def setup_logging(log_level=logging.INFO):
    if logging.root.handlers:  # Already configured
        return
    try:
        from rich.logging import RichHandler

        handler = RichHandler()
    except ImportError:
        handler = logging.StreamHandler()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
