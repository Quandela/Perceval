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
import time

import perceval as pcvl
from perceval import logger


def test_logger_config():
    logger_config = pcvl.LoggerConfig()
    logger_config.reset()
    logger_config.save()

    config = pcvl.utils.PersistentData().load_config()
    assert config["logging"] == {'use_python_logger': False, 'enable_file': False,
                                 'channels': {'general': {'level': 'off'}, 'resources': {'level': 'off'}, 'user': {'level': 'warn'}}}

    logger_config.enable_file()
    logger_config.set_level(pcvl.logging.level.warn, pcvl.logging.channel.general)
    logger_config.set_level(pcvl.logging.level.warn, pcvl.logging.channel.resources)
    logger_config.set_level(pcvl.logging.level.warn, pcvl.logging.channel.user)
    logger_config.save()

    config = pcvl.utils.PersistentData().load_config()
    assert config["logging"] == {'use_python_logger': False, 'enable_file': True,
                                 'channels': {'general': {'level': 'warn'}, 'resources': {'level': 'warn'}, 'user': {'level': 'warn'}}}

    logger_config.reset()
    logger_config.save()

    config = pcvl.utils.PersistentData().load_config()
    assert config["logging"] == {'use_python_logger': False, 'enable_file': False,
                                 'channels': {'general': {'level': 'off'}, 'resources': {'level': 'off'}, 'user': {'level': 'warn'}}}
