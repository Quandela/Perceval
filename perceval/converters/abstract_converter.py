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

from perceval.components import Port, Circuit, Processor, Source, PS
from perceval.utils import P, BasicState, Encoding, global_params
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp


def _create_mode_map(c_idx: int, c_data: int) -> dict:
    return {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}


class UnknownGateError(Exception):
    pass


class AGateConverter(ABC):
    r"""
    Converter class for gate based Circuits to perceval processor
    Qiskit or MyQLM
    """

    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        self._converted_processor = None
        self._input_list = None  # input state in list
        self._cnot_idx = 0  # counter for CNOTS in circuit
        self._source = source
        self._backend_name = backend_name
        self._heralded_cnot_builder = catalog["klm cnot"]
        self._heralded_cz_builder = catalog["heralded cz"]
        self._postprocessed_cnot_builder = catalog["postprocessed cnot"]
        self._generic_2mode_builder = catalog["generic 2 mode circuit"]
        self._lower_phase_component = Circuit(2) // (0, comp.PS(P("phi2")))
        self._upper_phase_component = Circuit(2) // (1, comp.PS(P("phi1")))
        self._two_phase_component = Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))

    @abstractmethod
    def count_qubits(self, gate_circuit) -> int:
        pass

    def _configure_processor(self, gate_circuit, **kwargs):
        """
        Sets port Encoding and default input state for the Processor
        """
        qname = kwargs.get("qname", "Q")  # Default value, set any name provided by the gate circuit
        n_qbits = self.count_qubits(gate_circuit)

        qubit_names = kwargs.get(
            "qubit_names", [f'{qname}{i}' for i in range(n_qbits)])

        n_moi = n_qbits * 2  # number of modes of interest = 2 * number of qbits
        self._input_list = [0] * n_moi
        self._converted_processor = Processor(self._backend_name, n_moi, self._source)
        for i in range(n_qbits):
            self._converted_processor.add_port(i * 2, Port(Encoding.DUAL_RAIL, qubit_names[i]))
            self._input_list[i * 2] = 1

    def apply_input_state(self):
        default_input_state = BasicState(self._input_list)
        self._converted_processor.with_input(default_input_state)

    @abstractmethod
    def convert(self, gate_circuit, use_postselection: bool = True) -> Processor:
        pass

    def _create_generic_1_qubit_gate(self, u) -> Circuit:
        # universal method, takes in unitary and approximates one using
        # Frobenius method

        # TODO: from the name of the gate, one could instantiate a more
        # optically meaningful circuit than an opaque unitary.
        if abs(u[1, 0]) + abs(u[0, 1]) < 2 * global_params["min_precision_gate"]:
            # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
            # limited numeric precision
            if abs(u[0, 0] - 1) < global_params["min_precision_gate"]:
                if abs(u[1, 1] - 1) < global_params["min_precision_gate"]:
                    return Circuit(2, name="I")  # returns Identity/empty circuit
                ins = self._upper_phase_component.copy()
            else:
                if abs(u[1, 1] - 1) < global_params["min_precision_gate"]:
                    ins = self._lower_phase_component.copy()
                else:
                    ins = self._two_phase_component.copy()
            optimize(ins, u, frobenius, sign=-1)
        else:
            ins = self._generic_2mode_builder.build_circuit()
            optimize(ins, u, frobenius, sign=-1)
        return ins

    def _create_2_qubit_gates_from_catalog(
            self,
            gate_name: str,
            n_cnot,
            c_idx,
            c_data,
            use_postselection,
            parameter=None):
        r"""
        List of Gates implemented:
        CNOT - Heralded and post-processed
        CZ - Heralded
        CRz - Heralded and post-processed (uses two CNOTs)
        SWAP
        """
        # TODO: implement other controlled gates through AXBXC decomposition
        gate_name = gate_name.upper()
        if gate_name in ["CNOT", "CX"]:
            self._cnot_idx += 1
            if use_postselection and self._cnot_idx == n_cnot:
                cnot_processor = self._postprocessed_cnot_builder.build_processor(backend=self._backend_name)
            else:
                cnot_processor = self._heralded_cnot_builder.build_processor(backend=self._backend_name)
            self._converted_processor.add(_create_mode_map(c_idx, c_data), cnot_processor)
        elif gate_name in ["CSIGN", "CZ"]:
            # Controlled Z in myqlm is named CSIGN
            cz_processor = self._heralded_cz_builder.build_processor(backend=self._backend_name)
            self._converted_processor.add(_create_mode_map(c_idx, c_data), cz_processor)
        elif gate_name in ["CRZ", "CR", "CRK"]:
            theta = np.pi / (2**parameter) if gate_name == "CRK" else parameter
            theta /= 2
            rz_plus_name = "Rz(%.2f)" % theta
            rz_plus = Circuit(2, rz_plus_name) // (0, PS(-theta / 2)) // (1, PS(theta / 2))
            rz_minus_name = "Rz(-%.2f)" % theta
            rz_minus = Circuit(2, rz_minus_name) // (0, PS(theta / 2)) // (1, PS(-theta / 2))
            # Break down the controlled Z rotation into this circuit:
            #
            # 0: ─────────────────@──────────────────@───
            #                     │                  │
            # 1: ───Rz(theta/2)───X───Rz(-theta/2)───X───
            #
            #   Rz(0)     if the first qubit is  |0>
            #   Rz(theta) if the first qubit is |1>
            self._converted_processor.add(c_data, rz_plus)
            self._create_2_qubit_gates_from_catalog(
                "CNOT", n_cnot, c_idx, c_data, use_postselection)
            self._converted_processor.add(c_data, rz_minus)
            self._create_2_qubit_gates_from_catalog(
                "CNOT", n_cnot, c_idx, c_data, use_postselection)
        elif gate_name == "SWAP":
            # Works for any FIRST and LAST, everything in-between is unchanged.
            c_first = min(c_idx, c_data)
            c_last = max(c_idx, c_data)
            n = (c_last - c_first) // 2 + 1
            perm = [i for i in range(n * 2)]
            perm[0] = c_last - c_first
            perm[1] = perm[0] + 1
            perm[perm[0]] = 0
            perm[perm[1]] = 1
            self._converted_processor.add(c_first, comp.PERM(perm))
        else:
            raise UnknownGateError(f"Gate not yet supported: {gate_name}")

        return self._converted_processor
