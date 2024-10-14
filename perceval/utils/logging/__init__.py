# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
import sys
from exqalibur import logging as xq_log

from .config import LoggerConfig, _USE_PYTHON_LOGGER
from .loggers import ExqaliburLogger, PythonLogger


_logger = None
level = xq_log.level
channel = xq_log.channel


def get_logger():
    global _logger
    return _logger


def _my_excepthook(excType, excValue, this_traceback):
    # only works for the main thread
    _logger.critical("Uncaught exception!", channel=channel.general,
                 exc_info=(excType, excValue, this_traceback))


def deprecated(*decorator_args, **decorator_kwargs):
    def decorator_deprecated(func):
        @functools.wraps(func)
        def wrapper_deprecated(*args, **kwargs):
            log = f"DeprecationWarning: Call to deprecated function (or staticmethod) {func.__name__}."
            if "reason" in decorator_kwargs:
                log += f" ({decorator_kwargs['reason']})"
            if "version" in decorator_kwargs:
                log += f" -- Deprecated since version {decorator_kwargs['version']}"
            _logger.warn(log, channel.user)
            return func(*args, **kwargs)
        return wrapper_deprecated
    return decorator_deprecated


def use_python_logger():
    global _logger
    if isinstance(_logger, PythonLogger):
        return
    if _logger is not None:
        _logger.info("Changing to Python logger", channel.general)
    _logger = PythonLogger()
    sys.excepthook = _my_excepthook


def use_perceval_logger():
    global _logger
    if isinstance(_logger, ExqaliburLogger):
        return
    if _logger is not None:
        _logger.info("Changing to exqalibur logger", channel.general)
    _logger = ExqaliburLogger()
    _logger.initialize()
    sys.excepthook = _my_excepthook


def apply_config(config: LoggerConfig):
    if config.python_logger_is_enabled():
        use_python_logger()
    else:
        use_perceval_logger()
    global _logger
    _logger.apply_config(config)

use_perceval_logger()
