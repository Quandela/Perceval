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

import pytest

from perceval import MitigationFactory, MitigationLevel, DistinguishablePhotonMitigation, CompilationAveraging


def test_factory():
    assert MitigationFactory(level=MitigationLevel.none).build() == []
    assert MitigationFactory(level=0).build() == []  # Check that we can also call it with an int

    for level in range(1, 4):
        assert len(MitigationFactory(level=MitigationLevel(level)).build()) > 0

    with pytest.raises(ValueError):
        MitigationFactory(level=4).build()

def test_factory_advanced_use():
    factory = MitigationFactory(level=MitigationLevel.none)

    my_mitigation = CompilationAveraging(10)
    factory.set_custom_mitigation(my_mitigation)

    assert factory.build() == [my_mitigation]

    order = 5
    factory.set_distinguishable_photon_mitigation(order)
    mitigations = factory.build()
    assert len(mitigations) == 2

    for mitigation in mitigations:
        if isinstance(mitigation, DistinguishablePhotonMitigation):
            assert mitigation._order == order
        else:
            assert mitigation == my_mitigation
