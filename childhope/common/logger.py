import logging

def setup_logger(name: str, level=logging.INFO):
    """
    Sets up a logger with a given name and log level, logging to the console.

    Parameters:
    - name (str): The name of the logger.
    - level: The logging level.

    Returns:
    - logger: Configured logger instance.
    """

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers, to avoid duplicate logs
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Define formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(console_handler)

    return logger
