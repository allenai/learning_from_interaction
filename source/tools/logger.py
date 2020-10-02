import logging
import sys

LOGGER = logging.getLogger("console_logger")


def excepthook(*args):
    LOGGER.error("Uncaught exception:", exc_info=args)


class StreamToLogger:
    def __init__(self):
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                LOGGER.info(line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            LOGGER.info(self.linebuf.rstrip())
        self.linebuf = ''


def init_logging(log_format="default", log_level="debug"):
    if len(LOGGER.handlers) > 0:
        return

    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    elif log_level == "warning":
        log_level = logging.WARNING
    elif log_level == "error":
        log_level = logging.ERROR
    assert log_level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR], \
        "unknown log_level {}".format(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    if log_format == "default":
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]",
            datefmt="%m/%d %H:%M:%S",
        )
    elif log_format == "defaultMilliseconds":
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]"
        )
    else:
        formatter = logging.Formatter(fmt=log_format, datefmt="%m/%d %H:%M:%S")
    ch.setFormatter(formatter)

    LOGGER.setLevel(log_level)
    LOGGER.addHandler(ch)

    sys.excepthook = excepthook
    sys.stdout = StreamToLogger()
