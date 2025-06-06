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
from .simulator_interface import ASimulatorDecorator
from perceval.utils import convert_polarized_state, Annotation, BasicState, StateVector, SVDistribution, BSDistribution
from perceval.utils.logging import channel, get_logger
from perceval.components import Unitary, IDetector, DetectionType, get_detection_type


class PolarizationSimulator(ASimulatorDecorator):
    def __init__(self, simulator):
        super().__init__(simulator)
        self._upol = None

    def _prepare_input(self, input_state):
        is_svd = False
        if isinstance(input_state, SVDistribution) and len(input_state) == 1:
            is_svd = True
            temp_sv = list(input_state.keys())[0]
            if len(temp_sv) == 1:
                input_state = temp_sv[0]

        if not isinstance(input_state, BasicState):
            raise NotImplementedError("Polarization simulator can only process BasicState inputs")
        spatial_input, preprocess_matrix = convert_polarized_state(input_state)
        circuit = Unitary(self._upol @ preprocess_matrix)
        self._simulator.set_circuit(circuit)
        if is_svd:
            spatial_input = SVDistribution(spatial_input)
        return spatial_input

    def set_circuit(self, circuit, m=None):
        self._prepare_circuit(circuit)

    def _prepare_circuit(self, circuit, m=None):
        self._upol = circuit.compute_unitary(use_polarization=True)

    def _prepare_detectors_impl(self, detectors: list[IDetector]):
        if get_detection_type(detectors) != DetectionType.PNR:
            get_logger().warn("Cannot use detectors in polarized circuits; giving PNR results", channel.user)
        return None

    def _split_odd_even(self, fs):
        s_odd = fs[1: fs.m: 2]
        s_even = fs[0: fs.m: 2]
        return s_odd, s_even

    def _postprocess_sv_impl(self, results: StateVector) -> StateVector:
        output = StateVector()
        for out_state, out_amplitude in results:
            s_odd, s_even = self._split_odd_even(out_state)
            # Keep annotations
            s_even.inject_annotation(Annotation("P:H"))
            s_odd.inject_annotation(Annotation("P:V"))
            reduced_out_state = s_odd.merge(s_even)
            output += out_amplitude * reduced_out_state
        return output

    def _postprocess_bsd_impl(self, results: BSDistribution) -> BSDistribution:
        output = BSDistribution()
        for out_state, output_prob in results.items():
            s_odd, s_even = self._split_odd_even(out_state)
            reduced_out_state = s_odd.merge(s_even)
            output[reduced_out_state] += output_prob
        return output

    def set_min_detected_photons_filter(self, value: int):
        super().set_min_detected_photons_filter(value)  # Transmit value to next layer
        self._min_detected_photons_filter = 0  # Photon count is kept, no need to filter results in this layer
