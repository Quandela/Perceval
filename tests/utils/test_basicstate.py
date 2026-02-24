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

from perceval.utils.states import AnnotatedFockState, BasicState, FockState, NoisyFockState
import warnings


def not_implemented_from(derivedClass, baseClass):
    ignored = [
                "__weakref__", "__final__",
                "__firstlineno__", "__static_attributes__"
              ]
    for attr in  baseClass.__dict__:
        if not hasattr(derivedClass, attr) and not attr in ignored:
            warnings.warn(FutureWarning(f"{derivedClass.__name__} does not implement '{attr}' which is present in class {baseClass.__name__}"))
    return [attr for attr in  baseClass.__dict__ if not hasattr(derivedClass, attr) and not attr.startswith('__')]

def test_FockStates_interface():
    baseClass = BasicState

    assert len( not_implemented_from(FockState, baseClass) ) == 0, f"Some methods declared in {baseClass.__name__} are not implemented"

    assert len( not_implemented_from(NoisyFockState, baseClass) ) == 0, f"Some methods declared in {baseClass.__name__} are not implemented"

    assert len( not_implemented_from(AnnotatedFockState, baseClass) ) == 0, f"Some methods declared in {baseClass.__name__} are not implemented"
