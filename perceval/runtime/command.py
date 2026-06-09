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

from perceval.utils.constants import KEY_MAX_SAMPLES, KEY_MAX_SHOTS
from perceval.serialization import register_to_serialization


class Command:
    """
    Describes a command through a name and a signature

    The signature is exposed with a list of (name, expected type, is_mandatory)
    """

    def __init__(self, name: str, signature: list[tuple[str, type | None, bool]], apply_emt: bool = True):
        # Signature is essentially a dict, but the ordering is important
        # Even though python preserves dict order, we prefer not to rely on it
        self.name = name
        self.signature = signature  # Does a type serialize well? If not, better removing it and delay the check to the execution
        self.apply_emt = apply_emt

    def check(self, parameters: dict[str, Any]):
        """
        Checks if the parameters are valid

        :param parameters: The final parameters that will be given as **kwargs in the command
        """
        for name, t, mandatory in self.signature:
            if name not in parameters and mandatory:
                raise ValueError(f"Didn't receive value for non-optional argument {name}")

    def fill(self, *args, **kwargs) -> dict[str, Any]:
        """
        :param args: The user given positional arguments
        :param kwargs: The user given keyword arguments
        :raises TypeError: If arguments do not match signature
        :return: A dictionary to use as **kwargs in the final command, compatible with the signature
        """
        res = dict()

        i = -1
        for i, arg in enumerate(args):
            if i >= len(self.signature):
                raise TypeError("Too many arguments")
            name, t, _ = self.signature[i]

            if t is not None and not isinstance(arg, t):
                raise TypeError(f"Argument received for {name} is not a {t.__name__}. Received {type(arg).__name__}")

            res[name] = arg

        for name, t, _ in self.signature[i + 1:]:
            if name in kwargs:
                value = kwargs.pop(name)

                if t is not None and not isinstance(value, t):
                    raise TypeError(
                        f"Argument received for {name} is not a {t.__name__}. Received {type(value).__name__}")

                res[name] = value

        for name in kwargs:
            if not any(n == name for n, *_ in self.signature):
                raise TypeError(f"Received unknown arguments {name}")
            if name in res:
                raise TypeError(f"Received duplicate arguments {name}")

        return res

    def __repr__(self):
        s = f"Command('{self.name}', signature: {self.signature}"
        if self.apply_emt:
            s += ', error mitigation compatible'
        s += ")"
        return s


class CommandFactoryClass:

    def __init__(self):
        self.probs = Command(name="probs", signature=[(KEY_MAX_SAMPLES, int, False), (KEY_MAX_SHOTS, int, False)], apply_emt=True)
        self.samples = self.create_acquisition_command("samples")
        self.sample_count = self.create_acquisition_command("sample_count")

    @staticmethod
    def create_acquisition_command(name: str) -> Command:
        return Command(name=name, signature=[(KEY_MAX_SAMPLES, int, True), (KEY_MAX_SHOTS, int, False)], apply_emt=True)

CommandFactory = CommandFactoryClass()


register_to_serialization(Command)
