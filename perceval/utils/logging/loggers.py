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

import json
import traceback
import warnings
import logging as py_log

from abc import ABC, abstractmethod
from os import path

from exqalibur import logging as exq_log

from ..persistent_data import PersistentData
from .config import LoggerConfig, _CHANNELS, _ENABLE_FILE

DEFAULT_CHANNEL = exq_log.channel.user


class ALogger(ABC):
    @abstractmethod
    def apply_config(self, config: LoggerConfig):
        pass

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

    def log_resources(self, my_dict: dict):
        """Log resources as a dictionary with:
             - level: info
             - channel: resources
             - serializing the dictionary as json so it can be easily deserialize

        :param my_dict: resources dictionary to log
        """
        self.info(json.dumps(my_dict), exq_log.channel.resources)


class ExqaliburLogger(ALogger):
    _ALREADY_INITIALIZED = False

    def initialize(self):
        if ExqaliburLogger._ALREADY_INITIALIZED:
            return

        ExqaliburLogger._ALREADY_INITIALIZED = True
        persistent_data = PersistentData()
        if persistent_data.is_writable():
            exq_log.initialize(self.get_log_file_path())
        else:
            exq_log.initialize()

        self.apply_config(LoggerConfig())
        exq_log.enable_console()

    def _configure_logger(self, logger_config: LoggerConfig):
        if _CHANNELS in logger_config:
            channels = list(exq_log.channel.__members__)
            levels = list(exq_log.level.__members__)
            for channel, level in logger_config[_CHANNELS].items():
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

    def apply_config(self, config: LoggerConfig):
        if config.python_logger_is_enabled():
            warnings.warn(UserWarning(
                "Cannot change type of logger from logger.apply_config, use perceval.utils.apply_config instead"))
        self._configure_logger(config)

    def get_log_file_path(self):
        return path.join(PersistentData().directory, "logs", "perceval.log")

    def enable_file(self):
        print(f"starting to write logs in {self.get_log_file_path()}")
        exq_log.enable_file()

    def disable_file(self):
        self.warn("Log to file is being stopped!")
        exq_log.disable_file()

    def set_level(
            self,
            level: exq_log.level,
            channel: exq_log.channel = DEFAULT_CHANNEL) -> None:
        self.info(f"Set log level to '{level.name}' for channel '{channel.name}'", channel.general)
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
        return '\n' + ''.join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2]))

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


class PythonLogger(ALogger):
    def __init__(self) -> None:
        self._logger = py_log.getLogger()
        self.apply_config(LoggerConfig())
        self._logger.addFilter(self._message_has_to_be_logged)

    def _get_levelno(self, level_name):
        if level_name == "debug":
            return 10
        elif level_name == "info":
            return 20
        elif level_name == "warn":
            return 30
        elif level_name == "error":
            return 40
        elif level_name == "critical":
            return 50
        else:
            return 60

    def apply_config(self, config: LoggerConfig):
        if not config.python_logger_is_enabled():
            warnings.warn(UserWarning(
                "Cannot change type of logger from logger.apply_config, use perceval.utils.apply_config instead"))
        self._level = {
            name: self._get_levelno(channel["level"]) for name, channel in config[_CHANNELS].items()
        }

    def _message_has_to_be_logged(self, record) -> bool:
        if "channel" in record.__dict__:
            if record.levelno < self._level[record.channel]:
                return False
        return True

    def enable_file(self):
        self.warn("This method have no effect. Use module logging to configure python logger")

    def disable_file(self):
        self.warn("This method have no effect. Use module logging to configure python logger")

    def set_level(self, level: int, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._config.set_level(level, channel)
        self._level[channel.name] = self._get_levelno(channel)

    def debug(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.debug(f"[debug] {msg}", extra={"channel": channel.name})

    def info(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.info(f"[info] {msg}", extra={"channel": channel.name})

    def warn(self, msg: str, channel: exq_log.channel = DEFAULT_CHANNEL):
        self._logger.warning(f"[warning] {msg}", extra={"channel": channel.name})

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
