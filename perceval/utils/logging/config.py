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
    def __init__(self):
        super().__init__()
        self.reset()
        self._load_from_persistent_data()

    def _init_channel(self, channel: exqalibur_logging.channel, level: exqalibur_logging.level = exqalibur_logging.level.off):
        self[_CHANNELS][channel.name] = {}
        self[_CHANNELS][channel.name]["level"] = level.name

    def reset(self):
        self[_USE_PYTHON_LOGGER] = False
        self[_ENABLE_FILE] = False
        self[_CHANNELS] = {}
        for name in [exqalibur_logging.channel.general, exqalibur_logging.channel.resources]:
            self._init_channel(name)
        self._init_channel(exqalibur_logging.channel.user, exqalibur_logging.level.warn)

    def _load_from_persistent_data(self):
        persistent_data = PersistentData()
        config = persistent_data.load_config()
        try:
            if config and _LOGGING in config:
                config = config[_LOGGING]
                if _CHANNELS in config:
                    for key in config[_CHANNELS]:
                        if key in _CHANNEL_NAMES:
                            self[_CHANNELS][key] = config[_CHANNELS][key]
                if _ENABLE_FILE in config:
                    self[_ENABLE_FILE] = config[_ENABLE_FILE]
        except KeyError as e:
            warnings.warn(UserWarning(f"Incorrect logger config, try to reset and save it. {e}"))

    def set_level(self, level: exqalibur_logging.level, channel: exqalibur_logging.channel):
        self[_CHANNELS][channel.name]["level"] = level.name

    def enable_file(self):
        self[_ENABLE_FILE] = True

    def disable_file(self):
        self[_ENABLE_FILE] = False

    def save(self):
        persistent_data = PersistentData()
        persistent_data.save_config({_LOGGING: dict(self)})
