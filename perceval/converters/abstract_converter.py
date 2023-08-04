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

from abc import ABC, abstractmethod

from perceval.components import Port, Circuit, Processor, Source
from perceval.utils import P, BasicState, Encoding
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp

MIN_PRECISION_GATE = 1e-4


def _create_mode_map(c_idx: int, c_data: int) -> dict:
    # todo: I want to put this inside the class with @staticmethod -> found problems
    return {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}


class AGateConverter(ABC):
    r"""
    Converter class for gate based Circuits to perceval processor
    Qiskit or MyQLM
    """

    def __init__(self, **kwargs):
        self._converted_processor = None
        self._source = kwargs.get("source", Source())
        self._backend_name = kwargs.get("backend_name", "SLOS")
        if not "catalog" in kwargs:
            raise KeyError("Missing catalog")
        catalog = kwargs["catalog"]
        self._heralded_cnot_builder = catalog["heralded cnot"]
        self._heralded_cz_builder = catalog["heralded cz"]
        self._postprocessed_cnot_builder = catalog["postprocessed cnot"]
        self._generic_2mode_builder = catalog["generic 2 mode circuit"]
        self._lower_phase_component = Circuit(2) // (0, comp.PS(P("phi2")))
        self._upper_phase_component = Circuit(2) // (1, comp.PS(P("phi1")))
        self._two_phase_component = Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))

    @property
    @abstractmethod
    def name(self) -> str:
        """Each converter would have a distinct name as a string"""

    # @staticmethod
    # @abstractmethod
    # def preferred_command() -> str:
        # todo : find out why they are used
    #    return "Gate"

    @abstractmethod
    def set_num_qbits(self, gate_circuit) -> int:
        pass

    def configure_processor(self, gate_circuit):
        """
        Sets port Encoding and default input state for the Processor
        """
        n_qbits = self.set_num_qbits(gate_circuit)
        n_moi = n_qbits * 2  # number of modes of interest = 2 * number of qbits
        input_list = [0] * n_moi
        self._converted_processor = Processor(self._backend_name, n_moi, self._source)
        for i in range(n_qbits):
            # todo : Qbit name? QISKIT has a way ; implement that
            self._converted_processor.add_port(i * 2, Port(Encoding.DUAL_RAIL, f'Q{i}'))
            input_list[i * 2] = 1
        default_input_state = BasicState(input_list)
        self._converted_processor.with_input(default_input_state)

    @abstractmethod
    def convert(self):
        """
        converts gates based circuits to one with linear optical components
        and returns a perceval processor
        Children - MyQlMConverter and QiskitConverter should define this method
        """
        pass

    def _create_generic_1_qubit_gate(self, u) -> Circuit:
        # universal method, takes in unitary and approximates one using
        # Frobenius method

        if abs(u[1, 0]) + abs(u[0, 1]) < 2 * MIN_PRECISION_GATE:
            # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
            # limited numeric precision
            if abs(u[0, 0] - 1) < MIN_PRECISION_GATE:
                if abs(u[1, 1] - 1) < MIN_PRECISION_GATE:
                    return Circuit(2, name="I")  # returns Identity/empty circuit
                ins = self._upper_phase_component.copy()
            else:
                if abs(u[1, 1] - 1) < MIN_PRECISION_GATE:
                    ins = self._lower_phase_component.copy()
                else:
                    ins = self._two_phase_component.copy()
            optimize(ins, u, frobenius, sign=-1)
        else:
            ins = self._generic_2mode_builder.build()
            optimize(ins, u, frobenius, sign=-1)
        return ins

    def _create_2_qubits_from_catalog(self, gate_name: str, n_cnot, cnot_idx, c_idx, c_data, c_first,
                                      use_postselection):
        r"""
        List of Gates implemented:
        CNOT - Heralded and post-processed
        CZ - Heralded
        SWAP
        """
        p = self._converted_processor
        if gate_name == "CNOT":
            cnot_idx += 1
            if use_postselection and cnot_idx == n_cnot:
                cnot_processor = self._postprocessed_cnot_builder.build()
            else:
                cnot_processor = self._heralded_cnot_builder.build()
            p.add(_create_mode_map(c_idx, c_data), cnot_processor)
        elif gate_name == "CSIGN":
            # Controlled Z in myqlm is named CSIGN
            cz_processor = self._heralded_cz_builder.build()
            p.add(_create_mode_map(c_idx, c_data), cz_processor)
        elif gate_name == "SWAP":
            # c_idx and c_data are consecutive - not necessarily ordered
            p.add(c_first, comp.PERM([2, 3, 0, 1]))
        else:
            raise RuntimeError(f"Gate not yet supported: {gate_name}")
        return p
