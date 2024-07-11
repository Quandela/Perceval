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
from os import path
import warnings
import traceback
import logging as py_log
from abc import ABC, abstractmethod

from exqalibur import logging as exq_log
from ..persistent_data import PersistentData
from .config import LoggerConfig, _CHANNELS, _ENABLE_FILE

DEFAULT_CHANNEL = exq_log.channel.user


class ILogger(ABC):
    @abstractmethod
    def enable_file(self):
        pass

    @abstractmethod
    def disable_file(self):
        pass

    @abstractmethod
    def set_level(self, level: int, channel: exq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def debug(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def info(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def warn(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def error(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        pass

    @abstractmethod
    def critical(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        pass


class ExqaliburLogger(ILogger):
    def initialize(self):
        persistent_data = PersistentData()
        log_path = self.get_log_file_path()
        if persistent_data.is_writable():
            exq_log.initialize(log_path)
        else:
            exq_log.initialize()

        config = LoggerConfig()
        if _ENABLE_FILE in config and config[_ENABLE_FILE]:
            print(f"starting to write logs in {log_path}")
            exq_log.enable_file()
        else:
            exq_log.disable_file()

        exq_log.enable_console()

        if _CHANNELS in config:
            channels = list(exq_log.channel.__members__)
            levels = list(exq_log.level.__members__)
            for channel, level in config[_CHANNELS].items():
                level = level['level']
                if channel not in channels:
                    warnings.warn(UserWarning(f"Unknown channel {channel}"))
                    return
                if level not in levels:
                    warnings.warn(UserWarning(f"Unknown level {level}"))
                    return
                exq_log.set_level(
                    exq_log.level.__members__[level],
                    exq_log.channel.__members__[channel])

    def get_log_file_path(self):
        return path.join(PersistentData().directory, "logs", "perceval.log")

    def enable_file(self):
        print(f"starting to write logs in {self.get_log_file_path()}")
        exq_log.enable_file()

    def disable_file(self):
        exq_log.disable_file()

    def set_level(
            self,
            level: exq_log.level,
            channel: exq_log.channel = DEFAULT_CHANNEL) -> None:
        exq_log.set_level(level, channel)

    def debug(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        exq_log.debug(str(msg), channel)

    def info(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        exq_log.info(str(msg), channel)

    def warn(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        exq_log.warn(str(msg), channel)

    def _format_exception(self, exc_info=None) -> str:
        if not exc_info:
            return ""
        return ': '+' '.join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])).replace("\n", " ").replace("    ", " ").replace("  ", ", ")

    def error(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exq_log.error(str(msg), channel)

    def critical(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exq_log.critical(str(msg), channel)


class PythonLogger(ILogger):
    def __init__(self) -> None:
        self._logger = py_log.getLogger()
        self._config = LoggerConfig()
        self._logger.addFilter(self._message_has_to_be_logged)

    def _message_has_to_be_logged(self, record) -> bool:
        if "channel" in record.__dict__:
            if record.levelno < self._config[_CHANNELS][record.channel]["level"]:
                return False
        return True

    def enable_file(self):
        py_log.warn("This method have no effect. Use module logging to configure python logger")

    def disable_file(self):
        py_log.warn("This method have no effect. Use module logging to configure python logger")

    def set_level(self, level: int, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._config.set_level(level, channel)

    def debug(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.debug(f"[debug] {msg}", extra={"channel": channel.name})

    def info(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.info(f"[info] {msg}", extra={"channel": channel.name})

    def warn(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.warn(f"[warning] {msg}", extra={"channel": channel.name})

    def error(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        self._logger.error(
            f"[error] {msg}",
            exc_info=exc_info,
            extra={"channel": channel.name})

    def critical(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        self._logger.critical(
            f"[critical] {msg}",
            exc_info=exc_info,
            extra={"channel": channel.name})
