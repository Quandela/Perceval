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

from perceval.components import Port, Circuit, Processor, Source, BS, PS
from perceval.utils import P, BasicState, Encoding
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp


MIN_PRECISION_GATE = 1e-4


def _create_mode_map(c_idx: int, c_data: int) -> dict:
    return {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}


class MyQLMConverter:
    r"""myQLM quantum circuit to perceval circuit converter.

    :param catalog: component library of perceval
    :param backend_name: Backend to use in computation, defaults to SLOS
    :param source: Defines the parameters of the source, defaults to an ideal one.
    """
    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        self._source = source
        self._backend_name = backend_name
        self._heralded_cnot_builder = catalog["heralded cnot"]
        self._heralded_cz_builder = catalog["heralded cz"]
        self._postprocessed_cnot_builder = catalog["postprocessed cnot"]
        self._generic_2mode_builder = catalog["generic 2 mode circuit"]
        self._lower_phase_component = Circuit(2) // (0, comp.PS(P("phi2")))
        self._upper_phase_component = Circuit(2) // (1, comp.PS(P("phi1")))
        self._two_phase_component = Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))

    def convert(self, qlmc, use_postselection: bool = True) -> Processor:
        r"""Convert a myQLM quantum circuit into a perceval `Processor`.

        :param qlmc: quantum gate-based myqlm circuit
        :type qlmc: qat.core.Circuit
        :param use_postselection: when True, uses a `post-processed CNOT` as the last gate. Otherwise, uses only
            `heralded CNOT`
        :return: the converted Processor
        """
        import qat
        from qat.core.circuit_builder.matrix_util import circ_to_np
        # importing the quantum toolbox of myqlm
        # this nested import fixes automatic class reference generation

        n_moi = qlmc.nbqbits * 2  # number of modes of interest = 2 * number of qbits
        input_list = [0] * n_moi
        p = Processor(self._backend_name, n_moi, self._source)

        for i in range(qlmc.nbqbits):
            p.add_port(i * 2, Port(Encoding.DUAL_RAIL, f'Q{i}'))
            input_list[i * 2] = 1
        default_input_state = BasicState(input_list)

        # count the number of CNOT gates to use during the conversion, will give us the number of herald to handle
        n_cnot = qlmc.count("CNOT")
        cnot_idx = 0

        for i, instruction in enumerate(qlmc.iterate_simple()):
            # qlmc.iterate_simple() is a tuple containing
            # ('Name', [value of the parameter for gate], [list of qbit positions where gate is applied])

            instruction_name = instruction[0]  # name of the Gate
            instruction_qbit = instruction[-1]  # tuple with list of qbit positions

            if instruction_name not in qlmc.gate_set:
                raise ValueError(f"cannot convert {instruction_name} - Not a Gate")
            # only gates are converted -> checking if instruction is in gate_set of AQASM

            if len(instruction_qbit) == 1:
                ins = None
                if instruction_name == "H":
                    ins = Circuit(2, name='H') // BS.H()
                elif instruction_name == "PH":
                    phi = instruction[1][0]  # value of the variable parameter in gate
                    ins = Circuit(2, name='PS') // (1, PS(phi))  # apply phase shift on 2nd mode
                else:
                    gate_id = qlmc.ops[i].gate
                    gate_matrix = qlmc.gateDic[gate_id].matrix  # gate matrix data
                    gate_u = circ_to_np(gate_matrix)  # gate matrix to numpy
                    ins = self._create_one_qubit_gate(gate_u)
                p.add(instruction_qbit[0]*2, ins.copy())
            else:
                # only 2 qubit gates
                c_idx = instruction_qbit[0] * 2
                c_data = instruction_qbit[1] * 2
                c_first = min(c_idx, c_data)  # used in SWAP

                if instruction_name == "CNOT":
                    cnot_idx += 1
                    if use_postselection and cnot_idx == n_cnot:
                        cnot_processor = self._postprocessed_cnot_builder.build()
                    else:
                        cnot_processor = self._heralded_cnot_builder.build()
                    p.add(_create_mode_map(c_idx, c_data), cnot_processor)
                elif instruction_name == "CSIGN":
                    # Controlled Z in myqlm is named CSIGN
                    cz_processor = self._heralded_cz_builder.build()
                    p.add(_create_mode_map(c_idx, c_data), cz_processor)
                elif instruction_name == "SWAP":
                    # c_idx and c_data are consecutive - not necessarily ordered
                    p.add(c_first, comp.PERM([2, 3, 0, 1]))
                else:
                    raise RuntimeError(f"Gate not yet supported: {instruction_name}")
        p.with_input(default_input_state)
        return p

    def _create_one_qubit_gate(self, u) -> Circuit:
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
