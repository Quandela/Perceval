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
import pytest
import logging

import perceval as pcvl
from perceval import logger


def log_test():
    for use_python_logger in [False]:
        pcvl.set_logger(use_python_logger)
        for level in [logging.WARNING]:
            for channel in ["user", "resources", "general"]:
                logger.set_level(level, channel)
                logger.warn("test", channel)
                pcvl.utils.logger.warn("test", channel)
                pcvl.logging.LOGGER.warn("test", channel)


def test_file():
    file_path = logger.get_log_file_path()
    assert os.path.exists(file_path)

    logger.reset_config()
    logger.save_config()

    config = pcvl.utils.PersistentData().load_config()
    assert config["logging"] == {'use_python_logger': False, 'enable_file': False,
                                 'channels': {'general': {'level': 60}, 'resources': {'level': 60}, 'user': {'level': 30}}}

    if os.path.exists(file_path):
        open(file_path, 'w').close()

    log_test()

    assert os.stat(file_path).st_size == 0  # check writing log to file is opt in

    logger.enable_file()
    log_test()

    with open(logger.get_log_file_path(), 'r') as file:
        lines = [line for line in file]
        assert all(["U[W] test" in line for line in lines[0:2]])
        assert all(["R[W] test" in line for line in lines[3:5]])
        assert all([" [W] test" in line for line in lines[6:8]])

    logger.save_config()
    config = pcvl.utils.PersistentData().load_config()
    assert config["logging"] == {'use_python_logger': False, 'enable_file': True,
                                 'channels': {'general': {'level': 30}, 'resources': {'level': 30}, 'user': {'level': 30}}}
