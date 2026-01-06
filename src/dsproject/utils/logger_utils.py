import logging

def get_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name or __name__)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    return logger