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

from perceval.components import Circuit, Processor, Source, BS, PS
from .abstract_converter import AGateConverter


class MyQLMConverter(AGateConverter):
    r"""myQLM quantum circuit to perceval circuit converter.

    :param catalog: component library of perceval
    :param backend_name: Backend to use in computation, defaults to SLOS
    :param source: Defines the parameters of the source, defaults to an ideal one.
    """
    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        super().__init__(catalog, backend_name, source)

    def count_qubits(self, gate_circuit) -> int:
        return gate_circuit.nbqbits

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

        n_cnot = qlmc.count("CNOT")  # count the number of CNOT gates in circuit - needed to find the num. heralds
        self._configure_processor(qlmc)    # empty processor with ports initialized

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
                    ins = self._create_generic_1_qubit_gate(gate_u)

                self._converted_processor.add(instruction_qbit[0]*2, ins.copy())
            else:
                if len(instruction_qbit) > 2:
                    # only 2 qubit gates
                    raise ValueError(f"Gates with number of Qbits higher than 2 not implemented")
                c_idx = instruction_qbit[0] * 2
                c_data = instruction_qbit[1] * 2
                c_first = min(c_idx, c_data)  # used in SWAP

                self._create_2_qubit_gates_from_catalog(instruction_name, n_cnot, c_idx, c_data,
                                                               c_first, use_postselection)

        self.apply_input_state()
        return self._converted_processor
