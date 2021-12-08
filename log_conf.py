""" log_conf module

This module is a basic implementation of shared logging between modules

-----------------------------------------------------------
LOGGING PREPERATIONS
-----------------------------------------------------------
predefined levels for logging
CRITICAL 50
ERROR    40
WARNING  30
INFO     20
RUNTIME  15
DEBUG    10
NOTSET   0
-----------------------------------------------------------
"""

# system imports
import logging
# third party imports
# local imports

logging.RUNTIME = 15

# how to add names/levels
logging.addLevelName(logging.RUNTIME, "RUNTIME")


class MyLogger(logging.Logger):
    def runtime(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.RUNTIME):
            self._log(logging.RUNTIME, message, args, **kwargs)


logging.setLoggerClass(MyLogger)
log = logging.getLogger(__name__)

logging.basicConfig(
    # format="%(asctime)-10s [%(levelname)s] %(module)10s:%(funcName)30s: %(message)s",
    # format="%(funcName)34s: %(message)s",
    format="%(asctime)-6s[%(levelname)5s] %(module)12s: %(funcName)30s:(%(lineno)4d): %(message)s",
    # datefmt='%m/%d/%Y %I:%M:%S %p',
    # datefmt='%d %I:%M:%S ',
    datefmt='%I:%M:%S ',
    level=logging.INFO,
    # level=logging.DEBUG,
)


def setLevelCritical():
    log.setLevel(logging.CRITICAL)


def setLevelError():
    log.setLevel(logging.ERROR)


def setLevelWarning():
    log.setLevel(logging.WARNING)


def setLevelInfo():
    log.setLevel(logging.INFO)


def setLevelRuntime():
    log.setLevel(logging.RUNTIME)


def setLevelDebug():
    log.setLevel(logging.DEBUG)


def log_small_horizontal_line():
    """ Print a spacing line """
    log.debug("-"*40)


def log_large_horizontal_line(index=None):
    """Print a header if `index` is an integer, otherwise
    prints a horizontal line of length 60.
    """
    if index is not None:
        assert isinstance(index, int), f"Index parameter must be an integer not a {type(index)}"
        string = f" State {index:>3d} "
        log.debug(f"\n{string:-^60}")
    else:
        log.debug(f"\n{'':-^60}")
