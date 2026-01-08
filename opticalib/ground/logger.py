"""
This module provides an easy interface for setting up scripts logging within
the Opticalib framework. It includes functions to configure a rotating file logger
and a simple text file logger class.

Author(s)
---------
- Chiara Selmi : written in 2020
- Pietro Ferraiuolo : rewritten in 2024

Example Usage
-------------
To set up logging for your script, use the ``set_up_logger`` function to configure a rotating file logger.
You can then log messages using the standard logging interface or the provided ``log`` function.
For simple text logging, instantiate the ``txtLogger`` class.

Example::

    # Set up a rotating file logger
    logger = set_up_logger('my_script.log', logging.INFO)

    # Log messages using the log function
    log("This is an informational message.", "INFO")
    log("This is a debug message.", "DEBUG")

    # Use the txtLogger for simple text logging
    txt_log = txtLogger('simple_log.txt')
    txt_log.log("This is a message written to a text file.")
"""
import logging as _l
import logging.handlers as _lh

class SystemLogger():
    """
    A class to manage the system logger instance.
    """
    def __init__(self, the_class: type | None = None):
        """
        Initializes the SystemLogger instance.
        
        Parameters
        ----------
        the_class : type, optional
            The class this instance of SystemLogger is associated with. If provided, 
            as `__class__`, the class name will be included in the log messages. 
            
            The default is None.
        """
        self.logger = SystemLogger.getSystemLogger()
        self.the_class = the_class

    def log(self, **kwargs: dict[str,str]) -> None:
        """
        Logs a message using the system logger.

        Parameters
        ----------
        message : str
            The message to log.
        level : str, optional
            The logging level to use for the message. This should be one of the
            following strings: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. (can
            use lowercase too).

            The default is 'INFO'.
        no_class : bool, False
            If True, the class name will not be included in the log message, in the
            case it is available.
        """
        no_class = kwargs.pop("no_class", False)
        the_class = None if no_class else self.the_class
        log(self.logger, the_class=the_class, **kwargs)

    def info(self, message: str) -> None:
        """
        Logs an informational message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.log(message=message, level="INFO")

    def debug(self, message: str) -> None:
        """
        Logs a debug message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.log(message=message, level="DEBUG")

    def warning(self, message: str) -> None:
        """
        Logs a warning message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.log(message=message, level="WARNING")

    def error(self, message: str) -> None:
        """
        Logs an error message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.log(message=message, level="ERROR")

    def critical(self, message: str) -> None:
        """
        Logs a critical message.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.log(message=message, level="CRITICAL")

    @staticmethod
    def getSystemLogger() -> _l.Logger:
        """
        Get the root system logger.

        Returns
        -------
        logging.Logger
            The root logger instance.
        """
        return set_up_logger(
            "system", logging_level=_l.INFO, format="%(asctime)s -- [%(levelname)s] -- %(message)s"
        )


def set_up_logger(
    filename: str, logging_level: int = _l.DEBUG, format: str | None = None
) -> _l.Logger:
    """
    Set up a rotating file logger.

    This function configures a logger to write log messages to a file with
    rotation. The log file will be encoded in UTF-8 and will rotate when it
    reaches a specified size, keeping a specified number of backup files.

    Parameters
    ----------
    filename : str
        The path to the log file where log messages will be written.
    logging_level : int
        The logging level to set for the logger. This should be one of the
        logging level constants defined in the ``logging`` module (e.g.,
        ``DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50``).

    Notes
    -----
    - The log file will rotate when it reaches 10,000,000 bytes (10 MB).
    - Up to 3 backup log files will be kept.
    - The log format includes the timestamp, log level, logger name, and message.
    - The logger is configured at the root level, affecting all loggers in the application.
    - The handler will perform an initial rollover when set up.

    Examples
    --------
    .. code-block:: python

        set_up_logger('/path/to/logfile.log', logging.DEBUG)
    """
    import os
    from opticalib.core.root import LOGGING_ROOT_FOLDER

    file_path = os.path.join(LOGGING_ROOT_FOLDER, filename)
    if format is not None:
        FORMAT = format
    else:
        FORMAT = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s"
    formato = _l.Formatter(fmt=FORMAT, datefmt="%Y%m%d_%H%M%S")
    handler = _lh.RotatingFileHandler(
        file_path, encoding="utf8", maxBytes=10000000, backupCount=1
    )
    root_logger = _l.getLogger()
    root_logger.setLevel(logging_level)
    handler.setFormatter(formato)
    handler.setLevel(logging_level)
    root_logger.addHandler(handler)
    handler.doRollover()
    return root_logger

def log(logger: _l.Logger, message: str, the_class: type | None = None, level: str = "INFO") -> None:
    """
    Log a message at the specified level.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to use for logging the message.
    message : str
        The message to log.
    the_class : type, optional
        The class from which the log is being made. If provided, as `__class__`, 
        the class name will be included in the log message. The default is None.
    level : str, optional
        The logging level to use for the message. This should be one of the
        following strings: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. (can
        use lowercase too).

        The default is 'INFO'.

    Notes
    -----
    - The message will be logged with the specified level.
    - If the specified level is not recognized, the message will be logged at the
      'DEBUG' level.
    """
    if the_class is not None:
        message = f"[{the_class.__qualname__}] {message}"
    level = level.upper()
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.debug(message)
        logger.warning(f"Invalid log level '{level}'. Defaulting to 'DEBUG'.")

class txtLogger:
    """
    Simple logger class for writing log messages to a text file.

    Parameters
    ----------
    file_path : str
        Path to the log file, including the file name.

    Attributes
    ----------
    file_path : str
        Path to the log file.
    """

    def __init__(self, file_path: str):
        """
        Initializes the txtLogger with the specified file path.

        Parameters
        ----------
        file_path : str
            The path to the log file.
        """
        self.file_path = file_path

    def log(self, message: str) -> None:
        """
        Writes the log message to the `.txt` file.

        Parameters
        ----------
        message : str
            The log message to be written to the file.
        """
        with open(self.file_path, "a") as f:
            f.write(message + "\n")
