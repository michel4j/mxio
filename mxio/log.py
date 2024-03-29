"""This module implements utility classes and functions for logging."""

import logging

LOG_LEVEL = logging.INFO


class TermColor(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def warn(cls, text):
        return '{}{}{}'.format(cls.WARNING, text, cls.ENDC)

    @classmethod
    def success(cls, text):
        return '{}{}{}'.format(cls.OKGREEN, text, cls.ENDC)

    @classmethod
    def error(cls, text):
        return '{}{}{}'.format(cls.FAIL, text, cls.ENDC)

    @classmethod
    def emphasis(cls, text):
        return '{}{}{}'.format(cls.BOLD, text, cls.ENDC)

    @classmethod
    def debug(cls, text):
        return '{}{}{}'.format(cls.OKBLUE, text, cls.ENDC)

    @classmethod
    def normal(cls, text):
        return text

    @classmethod
    def underline(cls, text):
        return '{}{}{}'.format(cls.UNDERLINE, text, cls.ENDC)


class NullHandler(logging.Handler):
    """A do-nothing log handler."""

    def emit(self, record):
        pass


class ColoredConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno == logging.WARNING:
                msg = TermColor.warn(msg)
            elif record.levelno > logging.WARNING:
                msg = TermColor.error(msg)
            elif record.levelno == logging.DEBUG:
                msg = TermColor.debug(msg)
            self.stream.write("{}\n".format(msg))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def get_module_logger(name):
    """
    A factory which creates loggers with the given name and returns it.
    """
    name = name.split('.')[-1]
    _logger = logging.getLogger(name)
    _logger.setLevel(LOG_LEVEL)
    _logger.addHandler(NullHandler())
    return _logger


def log_to_console(level=LOG_LEVEL):
    """
    Add a log handler which logs to the console.
    """

    console = ColoredConsoleHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s', '%b/%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_to_file(filename, level=logging.DEBUG):
    """
    Add a log handler which logs to the console.
    """
    logfile = logging.FileHandler(filename)
    logfile.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s', '%b/%d %H:%M:%S')
    logfile.setFormatter(formatter)
    logging.getLogger('').addHandler(logfile)
