import logging
from logging import Logger
from pathlib import Path
from typing import Optional, Union
import sys


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter with color support.

    :param fmt: Format string for log messages.
    :type fmt: str
    :param datefmt: Format string for log timestamps, defaults to "%H:%M:%S".
    :type datefmt: str
    """

    def __init__(self, fmt: str, datefmt: str = "%H:%M:%S"):
        blue = "\x1b[94;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        reset = "\x1b[0m"

        self.formats = {
            logging.DEBUG: f"{blue}{fmt}{reset}",
            logging.INFO: fmt,
            logging.WARNING: f"{yellow}{fmt}{reset}",
            logging.ERROR: f"{red}{fmt}{reset}",
        }
        self.datefmt = datefmt

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def set_logger(
    name: str = "matto_radiomics",
    level: int = logging.INFO,
    outpath: Optional[Union[Path, str]] = None,
) -> Logger:
    """
    Configures and returns a logger with the specified name, logging level, and optional file output.
    This function sets the root logger with a console handler that outputs colored logs to the terminal.
    Optionally, it can also log messages to a file.

    :param name: The name of the logger, defaults to "matto_radiomics".
    :type name: str
    :param level: The logging level for the console handler (e.g., `logging.INFO`, `logging.DEBUG`),
        defaults to `logging.INFO`.
    :type level: int
    :param outpath: The file path where logs should be written. If None, no file handler is added.
    :type outpath: pathlib.Path or str, optional
    :return: The configured logger instance.
    :rtype: logging.Logger
    """

    logger = logging.getLogger("matto_radiomics")
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level desired level
    ch = logging.StreamHandler()
    ch.setLevel(level)

    format_string = "[%(asctime)s] %(name)s - %(levelname)s: %(message)s"

    # Create colored formatter
    color_formatter = ColorFormatter(format_string)
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format_string)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    if outpath:
        fh = logging.FileHandler(outpath, mode="w")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(format_string)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logging.getLogger(name)
