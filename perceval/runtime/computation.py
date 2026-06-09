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

from typing import Any

from perceval.components import Experiment
from perceval.serialization import register_to_serialization
from .command import Command


class Computation:

    def __init__(self, command: Command, experiment: Experiment):
        """
        Descriptor of what we want to compute.
        This is meant to be fully independent of how we will get the results for it

        :param command: A command to do, describing what kind of results we want and the allowed parameters
        :param experiment: The Experiment we want to compute results for
        """
        self.command = command
        self.experiment = experiment
        self.parameters: dict[str, Any] = dict()
        self.job_name = command.name
        self.job_group_name: str | None = None

    def add_params(self, *args, **kwargs) -> None:
        """
        Adds or replace parameters with the given values, following the signature given by the command

        :param args: The user given positional arguments
        :param kwargs: The user given keyword arguments
        """
        self.parameters |= self.command.fill(*args, **kwargs)

    def validate(self):
        """
        Checks that all non-optional parameters are filled

        :raises ValueError: if parameters are not correct
        """
        self.command.check(self.parameters)

    def __iter__(self):
        yield self

    def __repr__(self):
        s = f"Computation({self.command.name.capitalize()}"
        if len(self.parameters):
            s += f", parameters {self.parameters}"
        s += ")"
        return s

register_to_serialization(Computation)
