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

from perceval.components import Port, Circuit, Processor, Source
from perceval.utils import P, BasicState, Encoding
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp


min_precision_gate = 1e-4


class myQLMConverter:
    r"""myQLM quantum circuit to perceval circuit converter.
    # todo: myQLM circuit does nt seem to know about the type of simulation do, we may need to see Jobs
    :param catalog: component library of perceval
    """
    def __init__(self, catalog):
        self._heralded_cnot_builder = catalog["heralded cnot"]
        self._heralded_cz_builder = catalog["heralded cz"]
        self._postprocessed_cnot_builder = catalog["postprocessed cnot"]
        self._generic_2mode_builder = catalog["generic 2 mode circuit"]
        self._lower_phase_component = Circuit(2) // (0, comp.PS(P("phi2")))
        self._upper_phase_component = Circuit(2) // (1, comp.PS(P("phi1")))
        self._two_phase_component = Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))

    def convert(self, qlmc, use_postselection: bool = True) -> Circuit:
        r"""Convert a myQLM quantum circuit into a `Circuit`.

        :param qlmc: quantum gate-based myqlm circuit
        :type qlmc: qat.core.Circuit
        :param use_postselection: when True, uses a `postprocessed CNOT` as the last gate. Otherwise, uses only
            `heralded CNOT`
        :return: the converted Circuit
        """
        import qat  # importing the quantum toolbox of myqlm
        # this nested import fixes automatic class reference generation

        # count the number of cnot to use during the conversion, will give us the number of herald to handle
        n_cnot = 0
        for instruction in qlmc.iterate_simple():
            # todo: implement this loop
            # in QIskit converter it counts the number of CNOT gates; why?
            pass
        #     if instruction[0].name == "cx":
        #         n_cnot += 1
        # cnot_idx = 0

        n_moi = qlmc.nbqbits * 2  # number of modes of interest = 2 * number of qbits
        input_list = [0] * n_moi
        c = Circuit()

        # todo: confused with ports, work on it
        # qubit_names = qc.qregs[0].name
        # for i in range(qc.qregs[0].size):
        #     p.add_port(i * 2, Port(Encoding.DUAL_RAIL, f'{qubit_names}{i}'))
        #     input_list[i * 2] = 1
        # default_input_state = BasicState(input_list)

        for instruction in qlmc.iterate_simple():
            instruction_name = instruction[0]
            instruction_qbit = instruction[-1]
            # information carried by instruction
            # each instruction will be a tuple containing 'name' and 'list of qbit numbers' of gate in the 1st and
            # the last position of the tuple respectively
            # tuple ('Name', [IDK yet], [list of number of qbits where gate is applied])

            # only gates are converted
            assert isinstance(instruction[0], qat.lang.AQASM.gates.Gate), "cannot convert (%s)" % instruction[0]

            if len(instruction_qbit) == 1:
                # one mode gate
                ins = self._create_one_qubit_gate(instruction_qbit)
                # ins._name = instruction_name
                c //= ins
            else:
                # more than 1 qubit gates
                c_idx = instruction[1][0].index * 2
                c_data = instruction[1][1].index * 2
                c_first = min(c_idx, c_data)

                if instruction_name == "swap":
                    pass
                elif instruction_name == "cz":
                    pass
                elif instruction_name == "cx":
                    pass
                else:
                    raise RuntimeError("Gate not yet supported: %s" % instruction_name)
        return c

    def _create_one_qubit_gate(self, u):
        # Question : why are we not using simply a hadamard?  todo: find out
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
