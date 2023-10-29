import threading
from typing import *
import logging

def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()

def get_my_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    myformat = '%(asctime)s\t[%(levelname)s]\t%(filename)s:%(lineno)d\t%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(myformat, date_format)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
