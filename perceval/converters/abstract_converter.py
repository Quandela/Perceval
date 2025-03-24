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
from __future__ import annotations

from abc import ABC, abstractmethod

from perceval import AProcessor
from perceval.components import Port, Circuit, Processor, catalog
from perceval.utils import P, BasicState, Encoding, global_params, PostSelect, NoiseModel
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
import perceval.components.unitary_components as comp
from perceval.converters.converter_utils import label_cnots_in_gate_sequence


def _create_mode_map(c_idx: int, c_data: int) -> dict:
    return {c_idx: 0, c_idx + 1: 1, c_data: 2, c_data + 1: 3}


class UnknownGateError(Exception):
    pass


class AGateConverter(ABC):
    r"""
    Converter class for gate based Circuits to perceval processor
    """

    def __init__(self, backend_name: str = "SLOS", noise_model: NoiseModel = None):
        self._converted_processor: AProcessor | None = None  # TODO: replace by Experiment?
        self._input_list = None  # input state in list
        self._noise_model = noise_model
        self._backend_name = backend_name

        # Define function handler to create complex components
        # Users could override them
        self.create_hcnot_processor = catalog["heralded cnot"].build_processor
        self.create_hcz_processor = catalog["heralded cz"].build_processor
        self.create_ppcnot_processor = catalog["postprocessed cnot"].build_processor
        self.create_generic_2mode_circuit = catalog["generic 2 mode circuit"].build_circuit
        self.create_lower_phase_circuit = lambda: Circuit(2) // (0, comp.PS(P("phi2")))
        self.create_upper_phase_circuit = lambda: Circuit(2) // (1, comp.PS(P("phi1")))
        self.create_2phase_circuit = lambda: Circuit(2) // (0, comp.PS(P("phi1"))) // (1, comp.PS(P("phi2")))

    @abstractmethod
    def count_qubits(self, gate_circuit) -> int:
        pass

    @abstractmethod
    def _get_gate_sequence(self, gate_circuit) -> list[list]:
        """
        Iterates over the gate circuit to return a list of Gate operations in the circuit to convert to LO circuit.
        The information in each element of the list is  - [gate name, gate qubit positions, gate parameters, gate unitary]
        Note - Gate parameter or gate unitary can be None if value is not needed.
        """
        pass

    @abstractmethod
    def _get_qubit_names(self, gate_circuit, n_qbits) -> list:
        """
        Returns a list of Qubit names for a given 'n_qubits' gate circuit. Use 'Q' as the default name
         unless the framework provides a custom name.
        """
        pass

    def _configure_processor(self, gate_circuit, **kwargs):
        """
        Sets port Encoding and default input state for the Processor
        """
        n_qbits = self.count_qubits(gate_circuit)
        qubit_names = self._get_qubit_names(gate_circuit, n_qbits)

        n_moi = n_qbits * 2  # In dual rail, number of modes of interest = 2 * number of qbits
        self._input_list = [0] * n_moi
        self._converted_processor = Processor(self._backend_name, n_moi, noise=self._noise_model)
        for i in range(n_qbits):
            self._converted_processor.add_port(i * 2, Port(Encoding.DUAL_RAIL, qubit_names[i]))
            self._input_list[i * 2] = 1

    def apply_input_state(self):
        default_input_state = BasicState(self._input_list)
        self._converted_processor.min_detected_photons_filter(default_input_state.n)
        self._converted_processor.with_input(default_input_state)

    def convert(self, gate_circuit, use_postselection: bool = True) -> Processor:
        """Convert a gate-based quantum circuit into a `Processor`.

        :param gate_circuit: gate-based quantum circuit (Qiskit, Myqlm or CQASM)
        :param use_postselection: when True (default), uses optimized number of `postprocessed CNOT` and
            'Heralded CNOT' gates. Otherwise, uses only `heralded CNOT`.

        :return: the converted processor
        """
        self._configure_processor(gate_circuit)  # empty processor with ports initialized
        gate_sequence = self._get_gate_sequence(gate_circuit)

        return self._generate_converted_processor(gate_sequence, use_postselection=use_postselection)

    def _generate_converted_processor(self, gate_sequence: list, use_postselection: bool) -> Processor:
        # gate_inf -> [gate name, gate qubit pos, gate parameter if any or None, unitary Matrix or None]

        optimized_gate_sequence = label_cnots_in_gate_sequence(gate_sequence)

        for gate_index, gate_inf in enumerate(gate_sequence):
            gate_name = gate_inf[0]

            if len(gate_inf[1]) == 1:
                if gate_name in catalog:
                    gate_param = gate_inf[2]
                    ins = self._create_catalog_1_qubit_gate(gate_name, param=gate_param)
                else:
                    ins = self._create_generic_1_qubit_gate(gate_inf[3])
                    ins._name = gate_inf[0]
                self._converted_processor.add(gate_inf[1][0] * 2, ins.copy())

            else:
                if len(gate_inf[1]) > 2:
                    raise NotImplementedError("2+ Qubit gates not implemented")

                c_idx = gate_inf[1][0] * 2
                c_data = gate_inf[1][1] * 2
                self._create_2_qubit_gates_from_catalog(optimized_gate_sequence[gate_index], c_idx, c_data,
                                                        use_postselection)
        self.apply_input_state()
        return self._converted_processor

    def _create_catalog_1_qubit_gate(self, gate_name: str, **kwargs) -> Processor:
        param = kwargs.get("param", None)
        if gate_name in ["rx", "ry", "rz"]:
            return catalog[gate_name].build_processor(theta=param)
        elif gate_name in ["ph"]:
            return catalog[gate_name].build_processor(phi=param)
        else:
            return catalog[gate_name].build_processor()

    def _create_generic_1_qubit_gate(self, u) -> Circuit:
        # universal method, takes in unitary and approximates one using
        # Frobenius method
        if abs(u[1, 0]) + abs(u[0, 1]) < 2 * global_params["min_precision_gate"]:
            # diagonal matrix - we can handle with phases, we consider that gate unitary parameters has
            # limited numeric precision
            if abs(u[0, 0] - 1) < global_params["min_precision_gate"]:
                if abs(u[1, 1] - 1) < global_params["min_precision_gate"]:
                    return Circuit(2, name="I")  # returns Identity/empty circuit
                ins = self.create_upper_phase_circuit()
            else:
                if abs(u[1, 1] - 1) < global_params["min_precision_gate"]:
                    ins = self.create_lower_phase_circuit()
                else:
                    ins = self.create_2phase_circuit()
            optimize(ins, u, frobenius, sign=-1)
        else:
            ins = self.create_generic_2mode_circuit()
            optimize(ins, u, frobenius, sign=-1)
        return ins

    def _create_2_qubit_gates_from_catalog(self, gate_name: str, c_idx: int, c_data: int, use_postselection: bool):
        r"""
        List of Gates implemented:
        CNOT - Heralded and post-processed
        CZ - Heralded
        CRz - Heralded and post-processed (uses two CNOTs)
        SWAP
        """
        # Save and clear current post-selection data from the converted processor before adding the next gate
        if self._converted_processor.post_select_fn is not None:
            post_select_curr = self._converted_processor.post_select_fn
        else:
            post_select_curr = PostSelect()  # save empty if I need to merge incoming PostSelect to it
        self._converted_processor.clear_postselection()  # clear current post-selection

        gate_name = gate_name.upper()
        if gate_name in ["POSTPROCESSED CNOT", "HERALDED CNOT"]:
            if use_postselection and gate_name == "POSTPROCESSED CNOT":
                cnot_processor = self.create_ppcnot_processor()
            else:
                cnot_processor = self.create_hcnot_processor()

            self._converted_processor.add(_create_mode_map(c_idx, c_data), cnot_processor)
            if self._converted_processor.post_select_fn is not None:
                post_select_curr.merge(self._converted_processor.post_select_fn)

        elif gate_name in ["CSIGN", "CZ"]:
            # Controlled Z in myqlm is named CSIGN
            cz_processor = self.create_hcz_processor()
            self._converted_processor.add(_create_mode_map(c_idx, c_data), cz_processor)
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

        # re-apply the cleared post-selection
        self._converted_processor.set_postselection(post_select_curr)
        return self._converted_processor
