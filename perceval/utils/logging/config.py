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

import warnings

from exqalibur import logging as exqalibur_logging
from ..persistent_data import PersistentData

_LOGGING = "logging"
_CHANNELS = "channels"
_ENABLE_FILE = "enable_file"
_USE_PYTHON_LOGGER = "use_python_logger"
_CHANNEL_NAMES = ["user", "general", "resources"]


class LoggerConfig(dict):
    """This class represent the logger configuration as a dictionary and can be used to save it into persistent data.
    On class initialization, the configuration will be loaded from persistent data.
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self._persistent_data = PersistentData()
        self._load_from_persistent_data()

    def _init_channel(self, channel: exqalibur_logging.channel, level: exqalibur_logging.level = exqalibur_logging.level.off):
        self[_CHANNELS][channel.name] = {}
        self[_CHANNELS][channel.name]["level"] = level.name

    def reset(self):
        """Reset the logger configuration to its default value, which is:
            - Disable file
            - Channel user at level warning
            - Channels general & resources off
        """
        self[_USE_PYTHON_LOGGER] = False
        self[_ENABLE_FILE] = False
        self[_CHANNELS] = {}
        for name in [exqalibur_logging.channel.general, exqalibur_logging.channel.resources]:
            self._init_channel(name)
        self._init_channel(exqalibur_logging.channel.user, exqalibur_logging.level.warn)

    def _load_from_persistent_data(self):
        config = self._persistent_data.load_config()
        try:
            if config and _LOGGING in config:
                config = config[_LOGGING]
                if _CHANNELS in config:
                    for key in config[_CHANNELS]:
                        if key in _CHANNEL_NAMES:
                            self[_CHANNELS][key] = config[_CHANNELS][key]
                if _ENABLE_FILE in config:
                    self[_ENABLE_FILE] = config[_ENABLE_FILE]
                if _USE_PYTHON_LOGGER in config:
                    self[_USE_PYTHON_LOGGER] = config[_USE_PYTHON_LOGGER]
        except KeyError as e:
            warnings.warn(UserWarning(f"Incorrect logger config, try to reset and save it. {e}"))

    def set_level(self, level: exqalibur_logging.level, channel: exqalibur_logging.channel):
        """Set the level of a channel in the configuration

        Warning: this will not change the current logger level but only the level of the channel in the current LoggerConfig instance


        :param level: _description_
        :param channel: _description_
        """
        self[_CHANNELS][channel.name]["level"] = level.name

    def use_python_logger(self):
        """Set the config to use the python logger

        Warning: this will not change the current logger level but only the level of the channel in the current LoggerConfig instance
        """
        self[_USE_PYTHON_LOGGER] = True

    def use_perceval_logger(self):
        """Set the config to use the perceval logger

        Warning: this will not change the current logger level, but only the level of the channel in the current LoggerConfig instance
        """
        self[_USE_PYTHON_LOGGER] = False

    def python_logger_is_enabled(self):
        return self[_USE_PYTHON_LOGGER]

    def enable_file(self):
        """Enable to save the log into a file in the configuration

        Warning: this will not change the current logger file saving, but only the file saving of the current LoggerConfig instance
        """
        self[_ENABLE_FILE] = True

    def disable_file(self):
        """Disable to save the log into a file in the configuration

        Warning: this will not change the current logger file saving, but only the file saving of the current LoggerConfig instance
        """
        self[_ENABLE_FILE] = False

    def save(self):
        """Save the current logger configuration in the persistent data
        """
        self._persistent_data.save_config({_LOGGING: dict(self)})
