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
from numbers import Number

from perceval.components.component_catalog import CatalogItem
from ._helpers import generate_chained_controlled_ops, generalized_cz, apply_rotations_to_qubits
from perceval.components import Circuit, Processor, Port
from perceval.utils import Encoding, PostSelect


class QLOQAnsatz(CatalogItem):
    description = "A QLOQ Ansatz generator, linking qudits with CZ gates"
    str_repr = "User defined (depends on the parameters)"
    params_doc = {
        "group_sizes": "list of DUAL_RAIL or QUDITn Encodings",
        "layers": "list of rotation layers to apply. Values can be either 'X', 'Y', or 'Z'",
        "phases": "list of names or numerical values for qudit phases (default 'phi{i}'). "
                  "Required length can be computed using self.get_parameter_nb",
        "ctype": "name of the entanglement gate to apply. Value can be either 'cx' or 'cz' (default 'cx')",
    }
    article_ref = "https://arxiv.org/pdf/2411.03878"

    def __init__(self):
        super().__init__('qloq ansatz')
        self._circ = None
        self._lp = None
        self._angle_offset = 0
        self._layers = None

    def _apply_layer_operations(self, offset: int, size: int) -> None:
        """
        Applies a set of layer operations to a segment of the quantum circuit.

        Args:
            offset: The starting mode on which to apply the layer operations.
            size: The size of the qubit group for which to apply layer operations.
        """
        for layer in self._layers:
            self._circ.add(offset,
                           apply_rotations_to_qubits(self._lp[self._angle_offset:self._angle_offset + size], size,
                                                     layer))
            self._angle_offset += size

    def _add_single_layer(self, offset: int, size: int, ctype: str):
        """
        - Use the apply_layer_operations() function to set rotational layers based on the angles.
        - Add entanglement with generate_chained_controlled_ops.
        """
        self._apply_layer_operations(offset, size)
        self._circ.add(offset, generate_chained_controlled_ops(ctype, size))
        self._apply_layer_operations(offset, size)

    def _build_qubit_circuit(self, qubit_group_sizes: list[int], lp: list[float], layers: list[str],
                             ctype="cx") -> Processor:
        """
        Builds a quantum circuit based on specified parameters. The circuit is generated
        for multiple groups of qubits with custom operations and entanglement.

        Args:
            qubit_group_sizes (list of int): List of sizes for each group of qubits.
            lp (list of float): List of angles for the parameterized gates.
            layers (list of str): Types of rotation layers to apply ('X', 'Y', 'Z').
            ctype (str, optional): The type of controlled operation to use ("cz" or "cx"). Defaults to "cx".

        Returns:
            Processor: The constructed quantum circuit as a Processor.
        """
        ctype = ctype.upper()

        total_modes = sum((2 ** n for n in qubit_group_sizes))
        self._circ = Processor("SLOS", total_modes, name="Machine Learning")
        self._layers = layers
        self._angle_offset = 0
        self._lp = lp
        offset = 0

        previous_size = None

        for j, size in enumerate(qubit_group_sizes):
            self._add_single_layer(offset, size, ctype)

            if previous_size is not None:
                # Add a Generalized Controlled-Z (CZ) gate between the current and previous group.
                self._circ.add(offset - 2 ** previous_size, generalized_cz(previous_size, size))

                if j == len(qubit_group_sizes) - 1:
                    break

                self._add_single_layer(offset, size, ctype)

            # Update the offset for the next iteration
            offset += 2 ** size
            previous_size = size

        # Reset offset and apply the final set of operations for all groups
        offset = 0
        for size in qubit_group_sizes:
            self._add_single_layer(offset, size, ctype)
            offset += 2 ** size
        return self._circ

    @staticmethod
    def get_parameter_nb(qubit_group_sizes: list[Encoding], nb_layers: int) -> int:
        """
        Calculate the total number of parameters needed for a quantum circuit
        with the given qubit group sizes and number of layers.

        Args:
            qubit_group_sizes: A list containing the encoding of each qubit group in the circuit.
            nb_layers: the number of layers (e.g., 'X', 'Y', 'Z') in the circuit.

        Returns:
            int: The total number of parameters required for the circuit.
        """

        qubit_group_sizes = [size.logical_length for size in qubit_group_sizes]

        # Each single layer appears twice for the first and last groups, and three times for the others
        depths = [2 if (i in (0, len(qubit_group_sizes) - 1)) else 3 for i in range(len(qubit_group_sizes))]

        # Calculate parameters per depth for each group
        parameters = [depth * size * 2 * nb_layers for depth, size in zip(depths, qubit_group_sizes)]

        return sum(parameters)

    def build_circuit(self, **kwargs) -> Circuit:
        assert "group_sizes" in kwargs, "missing required argument: 'group_sizes'"
        assert "layers" in kwargs, "missing required argument: 'layers'"

        group_sizes = kwargs["group_sizes"]
        assert len(group_sizes), "group_sizes is empty"

        for size in group_sizes:
            assert isinstance(size, Encoding), f"size must be a logical Encoding, got {type(size).__name__}"
            assert size == Encoding.DUAL_RAIL or size.name.startswith("QUDIT"), "Incompatible encoding for {size}"

        layers = kwargs["layers"]

        assert len(layers), "No layers provided"
        assert all(l in ("X", "Y", "Z") for l in layers), "layers can only be 'X', 'Y', 'Z'"
        ctype = kwargs.get("ctype", "cx")

        assert ctype in ["cx", "cz"], "ctype must be either 'cx' or 'cz'"

        phases = kwargs.get("phases", None)
        parameter_nb = self.get_parameter_nb(group_sizes, len(layers))

        if phases is not None:
            assert isinstance(phases, list), "phases must be a list"
            assert len(phases) == parameter_nb, \
                f"there must be enough phases for the circuit {parameter_nb} (got {len(phases)})"
        else:
            phases = self._generate_phases(parameter_nb)

        if not all(layer == "Y" for layer in layers):
            # TODO: remove this assert when Parameter expressions are ready (PCVL-866)
            assert all(isinstance(phase, Number) for phase in phases), \
                "phases must be given as numerical values when using X or Z layer"

        phases = [self._handle_param(phase) for phase in phases]
        group_sizes = [size.logical_length for size in group_sizes]

        self._build_qubit_circuit(group_sizes, phases, layers, ctype)
        return self._circ.linear_circuit()

    def build_processor(self, **kwargs) -> Processor:
        p = self._init_processor(**kwargs)

        group_sizes = kwargs["group_sizes"]
        offset = 0
        post_select_str = ""

        for i, size in enumerate(group_sizes):
            m = size.fock_length
            p.add_port(offset, Port(size, f"Group {i}"))
            post_select_str += f" & {list(range(offset, offset + m))} == 1"
            offset += m

        p.set_postselection(PostSelect(post_select_str[3:]))

        nb_heralds = 2 * (len(group_sizes) - 1)
        for _ in range(nb_heralds):
            p.add_herald(p.m - 1, 0)

        return p

    @staticmethod
    def _generate_phases(parameter_nb: int) -> list[str]:
        return [f"phi{i}" for i in range(parameter_nb)]
