import enum
import logging
from typing import Optional


class LoggingLevel(enum.Enum):
    """The logging levels supported by the logger utility."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerUtility:
    """The logger utility class for logging messages and GPU memory information."""

    _instance: Optional["LoggerUtility"] = None
    _level: Optional[int] = None

    def __new__(cls, level: LoggingLevel = LoggingLevel.INFO) -> "LoggerUtility":
        """
        Create a new instance of the logger utility if it does not exist. Otherwise, return the existing instance.

        Args:
            level (`LoggingLevel`, optional): The logging level to set. Defaults to `LoggingLevel.INFO`.

        Returns:
            `LoggerUtility`: The logger utility instance.
        """
        if cls._instance is None:
            cls._instance = super(LoggerUtility, cls).__new__(cls)
            cls._instance.__initialize(level)
            cls._level = level.value
        elif cls._level != level.value and cls._level > level.value:
            cls._instance.__set_logging_level(level)
            cls._level = level.value
        return cls._instance

    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance.

        Returns:
            `logging.Logger`: The logger instance.
        """
        return self.logger

    def __initialize(self, level: LoggingLevel) -> None:
        """
        Initializes the logger utility with the specified logging level.

        Args:
            level (`LoggingLevel`): Enum value for the logging level (e.g., LoggingLevel.INFO).
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.__setup_logger(level)

    def __set_logging_level(self, level: LoggingLevel) -> None:
        """
        Set the logging level for the logger instance.

        Returns:
            `logging.Logger`: The logger instance.
        """
        self.logger.setLevel(level.value)
        for handler in self.logger.handlers:
            handler.setLevel(level.value)

    def __setup_logger(
        self,
        level: LoggingLevel,
    ):
        """
        Set up the logger with the specified logging level.

        Args:
            level (`LoggingLevel`): Logging level as an enum value.
        """
        numeric_level = level.value
        self.logger.setLevel(numeric_level)  # Set the logging level

        # Avoid adding multiple handlers
        if not self.logger.hasHandlers():
            # Create a console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)

            # Create a formatter and set it for the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(console_handler)
