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
import logging
import traceback
import warnings
import logging as py_log

from abc import ABC, abstractmethod
from os import path

from exqalibur import logging as xq_log

from ..persistent_data import PersistentData
from .config import LoggerConfig, _CHANNELS, _ENABLE_FILE

DEFAULT_CHANNEL = xq_log.channel.user


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
    def set_level(self, level: int, channel: xq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def debug(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def info(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def warn(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        pass

    @abstractmethod
    def error(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        pass

    @abstractmethod
    def critical(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        pass

    def log_resources(self, my_dict: dict):
        """Log resources as a dictionary with:
             - level: info
             - channel: resources
             - serializing the dictionary as json so it can be easily deserialize

        :param my_dict: resources dictionary to log
        """
        self.info(json.dumps(my_dict), xq_log.channel.resources)


class ExqaliburLogger(ALogger):

    def __init__(self, persistent_data: PersistentData = None):
        self._persistent_data = persistent_data or PersistentData()
        self._configure_logger(LoggerConfig(persistent_data))
        xq_log.enable_console()

    def _configure_logger(self, logger_config: LoggerConfig):
        if _CHANNELS in logger_config:
            channels = list(xq_log.channel.__members__)
            levels = list(xq_log.level.__members__)
            for channel, level in logger_config[_CHANNELS].items():
                level = level['level']
                if channel not in channels:
                    warnings.warn(UserWarning(f"Unknown channel {channel}"))
                    return
                if level not in levels:
                    warnings.warn(UserWarning(f"Unknown level {level}"))
                    return
                xq_log.set_level(
                    xq_log.level.__members__[level],
                    xq_log.channel.__members__[channel])

    def apply_config(self, config: LoggerConfig):
        if config.python_logger_is_enabled():
            warnings.warn(UserWarning(
                "Cannot change type of logger from logger.apply_config, use perceval.utils.apply_config instead"))
        self._configure_logger(config)

    def get_log_file_path(self):
        return path.join(self._persistent_data.directory, "logs", "perceval.log")

    def enable_file(self):
        if self._persistent_data.is_writable():
            xq_log.initialize(log_filepath=self.get_log_file_path())
        else:
            xq_log.initialize()
        print(f"starting to write logs in {self.get_log_file_path()}")
        xq_log.enable_file()

    def disable_file(self):
        self.warn("Log to file is being stopped!")
        xq_log.disable_file()

    def set_level(
            self,
            level: xq_log.level,
            channel: xq_log.channel = DEFAULT_CHANNEL) -> None:
        self.info(f"Set log level to '{level.name}' for channel '{channel.name}'", channel.general)
        xq_log.set_level(level, channel)

    def debug(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        xq_log.debug(str(msg), channel)

    def info(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        xq_log.info(str(msg), channel)

    def warn(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        xq_log.warn(str(msg), channel)

    def _format_exception(self, exc_info=None) -> str:
        if not exc_info:
            return ""
        return '\n' + ''.join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2]))

    def error(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        xq_log.error(str(msg), channel)

    def critical(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += self._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        xq_log.critical(str(msg), channel)


class PythonLogger(ALogger):
    _level_ratio = 10  # Ratio between Python levels and exqalibur levels

    def __init__(self, logger: py_log.Logger = None):
        self._levels = {  # Default values
            xq_log.channel.general.name: xq_log.level.off.value * self._level_ratio,
            xq_log.channel.user.name: xq_log.level.warn.value * self._level_ratio,
            xq_log.channel.resources.name: xq_log.level.off.value * self._level_ratio
        }

        if logger is None:  # Create our own Python logger
            self._logger = py_log.getLogger("perceval")
            self._sh = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
            self._sh.setFormatter(formatter)
            self._logger.addHandler(self._sh)
            self._configure_levels(LoggerConfig())
        else:  # Work with an external logger
            self._logger = logger
            self._sh = None
            init_level = logger.getEffectiveLevel()
            self.set_level(init_level, xq_log.channel.general)

        self._logger.addFilter(self._message_has_to_be_logged)

    @staticmethod
    def _get_levelno(level_name: str):
        return xq_log.level.__members__.get(level_name, xq_log.level.off).value * PythonLogger._level_ratio

    def _configure_levels(self, config: LoggerConfig):
        self._levels = {
            name: self._get_levelno(channel["level"]) for name, channel in config[_CHANNELS].items()
        }

    def apply_config(self, config: LoggerConfig):
        if not config.python_logger_is_enabled():
            warnings.warn(UserWarning(
                "Cannot change type of logger from logger.apply_config, use perceval.utils.apply_config instead"))
        self._configure_levels(config)

    def _message_has_to_be_logged(self, record) -> bool:
        if "channel" in record.__dict__:
            return record.levelno >= self._levels[record.channel]
        return True

    def enable_file(self):
        self.warn("This method have no effect. Use module logging to configure python logger")

    def disable_file(self):
        self.warn("This method have no effect. Use module logging to configure python logger")

    def set_level(self, level: xq_log.level, channel: xq_log.channel = DEFAULT_CHANNEL):
        if not isinstance(level, int):
            level = self._get_levelno(level.name)
        min_level = min(self._levels.values())
        if level < min_level:  # If the expected level is lower than the current min of channel levels,
            self._logger.setLevel(level)  #  we lower it in the stored Python logger instance
            if self._sh is not None:
                self._sh.setLevel(level)
        self._levels[channel.name] = level

    def debug(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        self._logger.debug(msg, extra={"channel": channel.name})

    def info(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        self._logger.info(msg, extra={"channel": channel.name})

    def warn(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL):
        self._logger.warning(msg, extra={"channel": channel.name})

    def error(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        self._logger.error(
            msg,
            exc_info=exc_info,
            extra={"channel": channel.name})

    def critical(self, msg: str, channel: xq_log.channel = DEFAULT_CHANNEL, exc_info=None):
        self._logger.critical(
            msg,
            exc_info=exc_info,
            extra={"channel": channel.name})
