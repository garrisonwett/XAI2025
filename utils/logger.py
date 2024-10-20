import logging


class Logger:
    """A class to create and manage a logger with dynamic logging levels."""

    def __init__(self, name=__name__, debug=False):
        """
        Initialize the logger with an optional debug mode.

        Args:
            name (`str`): Name of the logger. Default is `__name__`.
            debug (`bool`): Whether to set the logger to debug mode or not. Default is `False`.
        
        Returns:
            `Logger`: Logger object with console handler.
        """
        self._name = name
        self._debug = debug
        self._logger = self._create_logger()

    def get_logger(self) -> logging.Logger:
        """Getter method for the logger property."""
        return self._logger
    
    def enable_debug(self) -> None:
        """Enable the debug mode and recreate the logger."""
        self._debug = True
        self._logger = self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """
        Create or recreate the logger based on the current debug setting.
        
        Returns:
            `Logger`: Logger object with console handler.
        """
        logger = logging.getLogger(self._name)

        # Remove existing handlers to avoid duplication
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set the appropriate logging level
        logger.setLevel(logging.DEBUG if self._debug else logging.INFO)

        # Create a handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self._debug else logging.INFO)

        # Set formatter for the console output
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(console_handler)

        return logger

# Initialize the default logger
default_logger = Logger()

##
## Example usage:
##
# from utils.logger import default_logger

# if __name__ == "__main__":
#     log = default_logger.get_logger()
#     log.info("Logger initialized with INFO level.")
#     log.debug("This debug message won't show without enabling debug mode.")

#     # Enable debug mode
#     default_logger.enable_debug()
#     log.debug("This is a debug message after enabling debug mode.")

