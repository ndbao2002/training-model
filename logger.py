# my_logger.py

import logging
import os

def setup_logger(name="my_app", log_path="logs/model.log", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Nếu logger đã có handler, không thêm lại nữa
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # Gắn handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Tắt log từ các thư viện ngoài
        for lib in ['matplotlib', 'urllib3', 'asyncio', 'requests', 'PIL']:
            logging.getLogger(lib).setLevel(logging.WARNING)

        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    return logger