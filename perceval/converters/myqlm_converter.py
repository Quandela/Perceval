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

from perceval.components import Circuit, Processor, Source, BS, PS, catalog
from perceval.utils.logging import get_logger, channel
from .abstract_converter import AGateConverter
from perceval.utils import NoiseModel


class MyQLMConverter(AGateConverter):
    r"""myQLM quantum circuit to perceval circuit converter.

    :param catalog: component library of perceval
    :param backend_name: Backend to use in computation, defaults to SLOS
    :param source: Defines the parameters of the source, defaults to an ideal one.
    """
    def __init__(self, backend_name: str = "SLOS", source: Source = None, noise_model: NoiseModel = None):
        super().__init__(backend_name, source, noise_model)
        from qat.core.circuit_builder.matrix_util import circ_to_np
        self._circ_to_np = circ_to_np

    def count_qubits(self, gate_circuit) -> int:
        return gate_circuit.nbqbits

    def _get_qubit_names(self, myqlm_circ, n_qbits):
        return [f'{"Q"}{i}' for i in range(n_qbits)]

    def _get_gate_unitary(self, myqlm_circ, i):
        # returns the unitary matrix of gates
        gate_id = myqlm_circ.ops[i].gate
        gate_matrix = myqlm_circ.gateDic[gate_id].matrix  # gate matrix data
        return self._circ_to_np(gate_matrix)  # gate matrix to numpy

    def _get_gate_sequence(self, myqlm_circ) -> list:
        # returns a nested list of gate names with corresponding qubit positions from a myqlm circuit
        get_logger().info(f"Convert myQLM circuit ({myqlm_circ.nbqbits} qubits, {len(myqlm_circ.ops)} operations) to processor",
                          channel.general)

        invalid_gates = [instruction for instruction in myqlm_circ.iterate_simple() if instruction[0] not in myqlm_circ.gate_set]
        assert not invalid_gates, f"Invalid instructions: {', '.join(str(instr[0]) for instr in invalid_gates)}"
        # only gates are converted -> checking if instruction is in gate_set of AQASM

        gate_info = []
        for i, gate_instruction in enumerate(myqlm_circ.iterate_simple()):
            gate_name = gate_instruction[0].lower()
            need_unitary = False
            if gate_name not in catalog and len(gate_instruction[2]) == 1:
                need_unitary = True
                gate_unitary = self._get_gate_unitary(myqlm_circ, i)

            gate_info.append([gate_name,
                              gate_instruction[2],
                              gate_instruction[1][0] if gate_instruction[1] else None,
                              gate_unitary if need_unitary else None])
        return gate_info
