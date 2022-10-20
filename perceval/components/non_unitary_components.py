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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from perceval.components.abstract_component import AParametrizedComponent


class TD(AParametrizedComponent):
    """Time delay"""
    DEFAULT_NAME = "TD"

    def __init__(self, dt):
        super().__init__(1)
        self._dt = self._set_parameter("t", dt, 0, None, False)

    def is_composite(self):  # TODO this has to go
        return False

    def get_variables(self, map_param_kid=None):  # is this useful?
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "t", "t", None, map_param_kid)
        return parameters

    def describe(self):
        if self._dt.fixed:
            value = float(self._dt)
        else:
            value = f'P("{self._dt.spv}")'
        return f"TD(t={value})"
