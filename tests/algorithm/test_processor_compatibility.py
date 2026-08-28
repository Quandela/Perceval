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

from perceval import PauliEigenStateType
from perceval.utils import NoiseModel, FockState
from perceval.providers import QuandelaCommunicationLayer
from perceval.runtime.legacy import Processor, RemoteProcessor
from perceval.algorithm import Analyzer, StateTomography, ProcessTomography, StateTomographyMLE, ProcessTomographyMLE
from perceval.algorithm.processor_compatibility import computer_from_processor

from tests.providers.quandela import get_rpc_handler_for_tests


def test_processor_to_computer():
    noise = NoiseModel(0.8)

    # Local processor
    p = Processor("SLOS", 2, noise=noise)
    comp = computer_from_processor(p)

    assert not comp.is_remote
    assert comp.noise == noise


    # RemoteProcessor
    rp = RemoteProcessor(rpc_handler=get_rpc_handler_for_tests(), m=2, noise=noise)
    comp = computer_from_processor(rp)

    assert comp.is_remote
    assert comp.noise == noise
    assert isinstance(comp._communication_layer, QuandelaCommunicationLayer)


def test_algorithms_old():
    p = Processor("SLOS", 2)

    analyzer = Analyzer(p, [FockState([1, 0])])
    analyzer.compute()

    state_tomography = StateTomography(p)
    state_tomography = StateTomography(operator_processor=p)
    state_tomography.perform_state_tomography([PauliEigenStateType.Xm])

    process_tomography = ProcessTomography(p)
    process_tomography.chi_matrix()

    state_tomography_mle = StateTomographyMLE(p)
    state_tomography_mle.state_tomography_density_matrix()

    process_tomography_mle = ProcessTomographyMLE(p)
    # process_tomography_mle.chi_matrix()  # Takes an infinite time - There is an infinite loop in _perform_mle_tomography
