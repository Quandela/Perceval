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
import numpy as np


min_precision_gate = 1e-4


class MyQLMConverter:
    r"""myQLM quantum circuit to perceval circuit converter.
    :param catalog: component library of perceval
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
        r"""Convert a myQLM quantum circuit into a `Circuit`.

        :param qlmc: quantum gate-based myqlm circuit
        :type qlmc: qat.core.Circuit
        :param use_postselection: when True, uses a `post-processed CNOT` as the last gate. Otherwise, uses only
            `heralded CNOT`
        :return: the converted Processor
        """
        import qat  # importing the quantum toolbox of myqlm
        # this nested import fixes automatic class reference generation

        # count the number of CNOT gates to use during the conversion, will give us the number of herald to handle
        n_cnot = 0
        for instruction in qlmc.iterate_simple():
            if instruction[0] == "CNOT":
                n_cnot += 1
        cnot_idx = 0

        n_moi = qlmc.nbqbits * 2  # number of modes of interest = 2 * number of qbits
        input_list = [0] * n_moi
        p = Processor(self._backend_name, n_moi, self._source)

        for i in range(qlmc.nbqbits):
            p.add_port(i * 2, Port(Encoding.DUAL_RAIL, f'{"q"}{i}'))  # todo: find if we really need qubit name!
            # Qbits are of this type : qat.lang.AQASM.bits.QRegister but this class does not have "name" attribute
            # class qat.lang.AQASM.bits.QRegister(offset, length=1, scope=None, qbits_list=None)
            input_list[i * 2] = 1
        default_input_state = BasicState(input_list)

        for i, instruction in enumerate(qlmc.iterate_simple()):
            instruction_name = instruction[0]  # name of the Gate
            instruction_qbit = instruction[-1]  # tuple with list of qbit positions
            # information carried by instruction
            # tuple ('Name', [value of the parameter for gate], [list of qbit positions where gate is applied])

            # only gates are converted -> checking if instruction is in gate_set of AQASM
            # in addition to known gates, there is "LOCK3 and "RELEASE" ->
            # todo: find out about lock and release
            assert instruction_name in qlmc.gate_set, "cannot convert (%s)" % instruction_name

            if len(instruction_qbit) == 1:
                if instruction_name == "H":
                    ins = Circuit(2, name='H') // BS.H()
                elif instruction_name == "PH":
                    phi = instruction[1][0]  # value of the variable parameter in gate
                    ins = Circuit(2, name='PS') // PS(phi)
                else:
                    gate_id = qlmc.ops[i].gate
                    gate_matrix = qlmc.gateDic[gate_id].matrix  # gate matrix data from myQLM
                    gate_u = self._myqlm_gate_unitary(gate_matrix)  # U of gate given by current instruction_name
                    ins = self._create_one_qubit_gate(gate_u)
                p.add(instruction_qbit[0]*2, ins.copy())
            else:
                # only 2 qubit gates
                c_idx = instruction_qbit[0] * 2
                c_data = instruction_qbit[1] * 2
                c_first = min(c_idx, c_data)  # used in SWAP, not implemented yet todo: implement

                if instruction_name == "CNOT":
                    cnot_idx += 1
                    if use_postselection and cnot_idx == n_cnot:
                        cnot_processor = self._postprocessed_cnot_builder.build()
                        mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    else:
                        cnot_processor = self._heralded_cnot_builder.build()
                        mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    p.add(mode_map, cnot_processor)
                elif instruction_name == "CSIGN":
                    cz_processor = self._heralded_cz_builder.build()
                    mode_map = {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}
                    p.add(mode_map, cz_processor)
                else:
                    raise RuntimeError("Gate not yet supported: %s" % instruction_name)
        p.with_input(default_input_state)
        return p

    @staticmethod
    def _myqlm_gate_unitary(gate_matrix):
        """
        Takes in GateDefinition Matrix -> as in myQLM and converts it into a numpy array of shape (nRows, nCols)
        """
        gate_u_list = []
        for val in gate_matrix.data:
            gate_u_list.append(val.re + 1j * val.im)
        u = np.array(gate_u_list).reshape(gate_matrix.nRows, gate_matrix.nCols)
        return u

    def _create_one_qubit_gate(self, u):
        # universal method, takes in unitary and approximates one using
        # Frobenius method
        if abs(u[1, 0]) + abs(u[0, 1]) < 2 * min_precision_gate:
            # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
            # limited numeric precision
            if abs(u[0, 0] - 1) < min_precision_gate:
                if abs(u[1, 1] - 1) < min_precision_gate:
                    return None
                ins = self._upper_phase_component.copy()
            else:
                if abs(u[1, 1] - 1) < min_precision_gate:
                    ins = self._lower_phase_component.copy()
                else:
                    ins = self._two_phase_component.copy()
            optimize(ins, u, frobenius, sign=-1)
        else:
            ins = self._generic_2mode_builder.build()
            optimize(ins, u, frobenius, sign=-1)
        return ins
