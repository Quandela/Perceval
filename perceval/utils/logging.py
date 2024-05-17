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

import os
import sys
import traceback
import logging as python_logging
from abc import ABC, abstractmethod

from .persistent_data import PersistentData
from exqalibur import logging as exqalibur_logging

_LOGGING = "logging"
_CHANNELS = "channels"
_ENABLE_FILE = "enable_file"
_USE_PYTHON_LOGGER = "use_python_logger"
_CHANNEL_NAMES = ["user", "general", "resources"]

_LOGGING_OFF = python_logging.CRITICAL + 10

_LEVEL_CONVERTER = {
    python_logging.DEBUG: exqalibur_logging.level.debug,
    python_logging.INFO: exqalibur_logging.level.info,
    python_logging.WARNING: exqalibur_logging.level.warn,
    python_logging.ERROR: exqalibur_logging.level.err,
    python_logging.CRITICAL: exqalibur_logging.level.critical,
    _LOGGING_OFF: exqalibur_logging.level.off,
}

_CHANNEL_CONVERTER = {
    "general": exqalibur_logging.channel.general,
    "resources": exqalibur_logging.channel.resources,
    "user": exqalibur_logging.channel.user,
}


class LoggerConfig(dict):
    def __init__(self):
        super().__init__()
        self.set_to_default_config()
        self._load_from_persistent_data()

    def set_to_default_config(self):
        self[_USE_PYTHON_LOGGER] = False
        self[_ENABLE_FILE] = False
        self[_CHANNELS] = {}
        for name in ["general", "resources"]:
            self._init_channel(name)
        self._init_channel("user", python_logging.WARNING)

    def _init_channel(self, name: str, level: int = _LOGGING_OFF):
        if name not in _CHANNEL_NAMES:
            raise KeyError(f"Unknown channel {name}")
        if level and level not in _LEVEL_CONVERTER.keys():
            raise KeyError(f"Unknown level {level} for channel {name}")
        self[_CHANNELS][name] = {}
        self[_CHANNELS][name]["level"] = level

    def _load_from_persistent_data(self) -> str:
        persistent_data = PersistentData()
        config = persistent_data.load_config()
        if config and _LOGGING in config:
            config = config[_LOGGING]
            if _CHANNELS in config:
                for key in config[_CHANNELS]:
                    if key in _CHANNEL_NAMES:
                        self[_CHANNELS][key] = config[_CHANNELS][key]
            if _ENABLE_FILE in config:
                self[_ENABLE_FILE] = config[_ENABLE_FILE]

    def set_level(self, level: int, channel: str = "user"):
        self[_CHANNELS][channel]["level"] = level

    def save(self):
        persistent_data = PersistentData()
        persistent_data.save_config({_LOGGING: dict(self)})

    def message_has_to_be_logged(self, record) -> bool:
        if "channel" in record.__dict__:
            if record.levelno < self[_CHANNELS][record.channel]["level"]:
                return False
        return True


CONFIG = LoggerConfig()


class ExqaliburLogger():
    @staticmethod
    def initialize():
        persistent_data = PersistentData()
        log_path = ExqaliburLogger.get_log_file_path()
        if persistent_data.is_writable():
            exqalibur_logging.initialize(log_path)
        else:
            exqalibur_logging.initialize()

        if _ENABLE_FILE in CONFIG and CONFIG[_ENABLE_FILE]:
            print(f"starting to write logs in {log_path}")
            exqalibur_logging.enable_file()
        else:
            exqalibur_logging.disable_file()

        exqalibur_logging.enable_console()

        if _CHANNELS in CONFIG:
            for name, value in CONFIG[_CHANNELS].items():
                exqalibur_logging.set_level(_LEVEL_CONVERTER[value["level"]], _CHANNEL_CONVERTER[name])

    @staticmethod
    def get_log_file_path():
        return PersistentData().get_full_path('log')

    @staticmethod
    def enable_file():
        CONFIG[_ENABLE_FILE] = True
        print(f"starting to write logs in {ExqaliburLogger.get_log_file_path()}")
        exqalibur_logging.enable_file()

    @staticmethod
    def disable_file():
        CONFIG[_ENABLE_FILE] = False
        exqalibur_logging.disable_file()

    @staticmethod
    def save_config():
        CONFIG.save()

    @staticmethod
    def reset_config():
        CONFIG.set_to_default_config()

    @staticmethod
    def set_level(level: int, channel: str = "user"):
        CONFIG.set_level(level, channel)
        exqalibur_logging.set_level(_LEVEL_CONVERTER[level], _CHANNEL_CONVERTER[channel])

    @staticmethod
    def debug(msg: str, channel: str = "user"):
        exqalibur_logging.debug(str(msg), _CHANNEL_CONVERTER[channel])

    @staticmethod
    def info(msg: str, channel: str = "user"):
        exqalibur_logging.info(str(msg), _CHANNEL_CONVERTER[channel])

    @staticmethod
    def warn(msg: str, channel: str = "user"):
        exqalibur_logging.warn(str(msg), _CHANNEL_CONVERTER[channel])

    @staticmethod
    def _format_exception(exc_info=None) -> str:
        if not exc_info:
            return ""
        return ': '+' '.join(traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])).replace("\n", " ").replace("    ", " ").replace("  ", ", ")

    @staticmethod
    def error(msg: str, channel: str = "user", exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += ExqaliburLogger._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exqalibur_logging.error(str(msg), _CHANNEL_CONVERTER[channel])

    @staticmethod
    def critical(msg: str, channel: str = "user", exc_info=None):
        msg = str(msg)
        if exc_info:
            msg += ExqaliburLogger._format_exception(exc_info)
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2])
        exqalibur_logging.critical(str(msg), _CHANNEL_CONVERTER[channel])


class PythonLogger():
    def __init__(self) -> None:
        self._logger = python_logging.getLogger()
        self._logger.addFilter(CONFIG.message_has_to_be_logged)

    def enable_file(self):
        python_logging.warn("This method have no effect. Use module logging to configure python logger")

    def disable_file(self):
        python_logging.warn("This method have no effect. Use module logging to configure python logger")

    def set_level(self, level: int, channel: str = "user"):
        CONFIG.set_level(level, channel)

    def save_config(self):
        CONFIG.save()

    def reset_config(self):
        CONFIG.set_to_default_config()

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


ExqaliburLogger.initialize()
LOGGER = ExqaliburLogger


def _my_excepthook(excType, excValue, this_traceback):
    # only works for the main thread
    LOGGER.error("Logging an uncaught exception", channel="general",
                 exc_info=(excType, excValue, this_traceback))


def set_level(level: int, channel: str = "user"):
    CONFIG.set_level(level, channel)
    LOGGER.set_level()
