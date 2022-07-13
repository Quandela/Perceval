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

from .source import Source
from .circuit import ACircuit
from perceval.utils import SVDistribution, StateVector
from perceval.utils.renderer.shapes import detector_shape, source_shape
from perceval.backends import Backend
from typing import Dict, Callable, Type, Literal


class Processor:
    """
        Generic definition of processor as sources + circuit
    """
    def __init__(self, sources: Dict[int, Source], circuit: ACircuit, post_select_fn: Callable = None):
        r"""Define a processor with sources connected to the circuit and possible post_selection

        :param sources: a list of Source used by the processor
        :param circuit: a circuit define the processor internal logic
        :param post_select_fn: a post-selection function
        """
        self._sources = sources
        self._circuit = circuit
        self._post_select = post_select_fn
        self._inputs_map = None
        for k in range(circuit.m):
            if k in sources:
                distribution = sources[k].probability_distribution()
            else:
                distribution = SVDistribution(StateVector("|0>"))
            # combine distributions
            if self._inputs_map is None:
                self._inputs_map = distribution
            else:
                self._inputs_map *= distribution

    @property
    def source_distribution(self):
        return self._inputs_map

    def run(self, simulator_backend: Type[Backend]):
        """
            calculate the output probabilities - returns performance, and output_maps
        """
        # first generate all possible outputs
        sim = simulator_backend(self._circuit.compute_unitary(use_symbolic=False))
        # now generate all possible outputs
        outputs = SVDistribution()
        for input_state, input_prob in self._inputs_map.items():
            for (output_state, p) in sim.allstateprob_iterator(input_state):
                if p and (not self._post_select or self._post_select(output_state)):
                    outputs[StateVector(output_state)] += p*input_prob
        all_p = sum(v for v in outputs.values())
        if all_p == 0:
            return 0, outputs
        # normalize probabilities
        for k in outputs.keys():
            outputs[k] /= all_p
        return all_p, outputs

    def pdisplay(self,
                 map_param_kid: dict = None,
                 shift: int = 0,
                 output_format: Literal["text", "html", "mplot", "latex"] = "text",
                 recursive: bool = False,
                 compact: bool = False,
                 precision: float = 1e-6,
                 nsimplify: bool = True,
                 **opts):
        printer = self._circuit.pdisplay(map_param_kid=map_param_kid,
                                         shift=shift,
                                         output_format=output_format,
                                         recursive=recursive,
                                         compact=compact,
                                         precision=precision,
                                         nsimplify=nsimplify,
                                         complete_drawing=False,
                                         **opts)
        for k in range(self._circuit.m):
            source_display_params = {}
            if k in self._sources:
                source_display_params['name'] = self._sources[k]._name
                source_display_params['content'] = '1'
            else:
                source_display_params['content'] = '0'

            printer.add_shape_left(k, source_shape, **source_display_params)
            printer.add_shape_right(k, detector_shape)
        return printer.draw()
