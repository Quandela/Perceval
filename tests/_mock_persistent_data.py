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


from perceval.utils import PersistentData, LoggerConfig
from perceval.utils.logging import ExqaliburLogger
from perceval.runtime.remote_config import RemoteConfig, QUANDELA_CLOUD_URL


class RemoteConfigForTest(RemoteConfig):
    """
    Overrides the file used for remote config to target a file in temporary sub-folder.
    This allows to run tests without removing actual persistent data or risking messing up system or user directories
    """

    def __init__(self, temp_dir):
        super().__init__(PersistentData(directory=temp_dir))


class LoggerConfigForTest(LoggerConfig):
    """
    Overrides the file used for logger config to target a file in temporary sub-folder.
    This allows to run tests without removing actual persistent data or risking messing up system or user directories
    """

    def __init__(self, temp_dir):
        super().__init__()
        self.reset()
        self._persistent_data = PersistentData(directory=temp_dir)
        self._load_from_persistent_data()


class ExqaliburLoggerForTest(ExqaliburLogger):
    """
    Overrides the config used for logger .
    This allows to run tests without removing actual persistent data or risking messing up system or user directories
    """

    def __init__(self, temp_dir) -> None:
        super().__init__()
        self._config = LoggerConfigForTest(temp_dir)
        self._configure_logger(self._config)
