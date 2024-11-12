# logger.py
import logging

def setup_logger(name, level=logging.INFO):
    """
    Set up and return a logger with the given name and level.
    Ensures that handlers are not added multiple times.
    """
    # Create or get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger already has them
    if not logger.handlers:
        # Create a stream handler
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Create and set the formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    return logger
