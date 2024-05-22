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

import traceback
import logging as python_logging
from abc import ABC, abstractmethod

from ..persistent_data import PersistentData
from .config import LoggerConfig, _CHANNEL_CONVERTER, _CHANNELS, _ENABLE_FILE, _LEVEL_CONVERTER, _LOGGING, _LOGGING_OFF, _USE_PYTHON_LOGGER
from exqalibur import logging as exqalibur_logging


class ILogger():
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def set_level(self, level: int, channel: str = "user"):
        pass

    @abstractmethod
    def debug(self, msg: str, channel: str = "user"):
        pass

    @abstractmethod
    def info(self, msg: str, channel: str = "user"):
        pass

    @abstractmethod
    def warn(self, msg: str, channel: str = "user"):
        pass

    @abstractmethod
    def error(self, msg: str, channel: str = "user", exc_info=None):
        pass

    @abstractmethod
    def critical(self, msg: str, channel: str = "user", exc_info=None):
        pass


class ExqaliburLogger(ILogger):
    def initialize(self):
        persistent_data = PersistentData()
        log_path = self.get_log_file_path()
        if persistent_data.is_writable():
            exqalibur_logging.initialize(log_path)
        else:
            exqalibur_logging.initialize()

        config = LoggerConfig()
        if _ENABLE_FILE in config and config[_ENABLE_FILE]:
            print(f"starting to write logs in {log_path}")
            exqalibur_logging.enable_file()
        else:
            exqalibur_logging.disable_file()

        exqalibur_logging.enable_console()

        if _CHANNELS in config:
            for name, value in config[_CHANNELS].items():
                exqalibur_logging.set_level(_LEVEL_CONVERTER[value["level"]], _CHANNEL_CONVERTER[name])

    def get_log_file_path(self):
        return PersistentData().get_full_path('log')

    def enable_file(self):
        print(f"starting to write logs in {self.get_log_file_path()}")
        exqalibur_logging.enable_file()

    def disable_file(self):
        exqalibur_logging.disable_file()

    def set_level(self, level: int, channel: str = "user"):
        exqalibur_logging.set_level(_LEVEL_CONVERTER[level], _CHANNEL_CONVERTER[channel])

    def debug(self, msg: str, channel: str = "user"):
        exqalibur_logging.debug(str(msg), _CHANNEL_CONVERTER[channel])

    def info(self, msg: str, channel: str = "user"):
        exqalibur_logging.info(str(msg), _CHANNEL_CONVERTER[channel])

    def warn(self, msg: str, channel: str = "user"):
        exqalibur_logging.warn(str(msg), _CHANNEL_CONVERTER[channel])

    def _format_exception(self, exc_info=None) -> str:
        if not exc_info:
            return ""
        return ': '+' '.join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])).replace("\n", " ").replace("    ", " ").replace("  ", ", ")

    def error(self, msg: str, channel: str = "user", exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exqalibur_logging.error(str(msg), _CHANNEL_CONVERTER[channel])

    def critical(self, msg: str, channel: str = "user", exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exqalibur_logging.critical(str(msg), _CHANNEL_CONVERTER[channel])


class PythonLogger(ILogger):
    def __init__(self) -> None:
        self._logger = python_logging.getLogger()
        self._config = LoggerConfig()
        self._logger.addFilter(self._message_has_to_be_logged)

    def _message_has_to_be_logged(self, record) -> bool:
        if "channel" in record.__dict__:
            if record.levelno < self._config[_CHANNELS][record.channel]["level"]:
                return False
        return True

    def enable_file(self):
        python_logging.warn("This method have no effect. Use module logging to configure python logger")

    def disable_file(self):
        python_logging.warn("This method have no effect. Use module logging to configure python logger")

    def set_level(self, level: int, channel: str = "user"):
        self._config.set_level(level, channel)

    def debug(self, msg: str, channel: str = "user"):
        self._logger.debug(f"[debug] {msg}", extra={"channel": channel})

    def info(self, msg: str, channel: str = "user"):
        self._logger.info(f"[info] {msg}", extra={"channel": channel})

    def warn(self, msg: str, channel: str = "user"):
        self._logger.warn(f"[warning] {msg}", extra={"channel": channel})

    def error(self, msg: str, channel: str = "user", exc_info=None):
        self._logger.error(f"[error] {msg}", exc_info=exc_info, extra={
                           "channel": channel})

    def critical(self, msg: str, channel: str = "user", exc_info=None):
        self._logger.critical(f"[critical] {msg}", exc_info=exc_info, extra={
                              "channel": channel})
