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
from perceval.components import Processor, Source, Circuit, BS, PS, PERM, catalog
from perceval.utils.logging import get_logger, channel
from .abstract_converter import AGateConverter
from .converter_utils import label_cnots_in_gate_sequence
from perceval.utils import NoiseModel

import numpy as np
import re

# A collection of gates for which meaningful optical circuits (beam splitter,
# phase shifter, permutation) exist.
# Source:
# https://qutech-delft.github.io/cQASM-spec/language_specification/instructions/gates.html#standard-gate-set

_CQASM_1_QUBIT_GATES = {
    "H", "X", "X90", "mX90", "Rx", "Y", "Y90", "mY90", "Ry", "S", "Sdag", "T", "Tdag", "Z", "Rz",
    # For v1 compatibility
    "I"
}

_CQASM_2_QUBIT_GATES = {
    "CNOT", "CZ"
}

class ConversionBadVersionError(Exception):
    pass


class ConversionSyntaxError(Exception):
    pass


class ConversionUnsupportedFeatureError(Exception):
    pass


class CQASMConverter(AGateConverter):
    r"""cQASM quantum circuit to perceval processor converter.

    :param catalog: a component library to use for the conversion. It must contain CNOT gates.
    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    """
    def __init__(self, backend_name: str = "SLOS", source: Source = None, noise_model: NoiseModel = None):
        super().__init__(backend_name, source, noise_model)
        import cqasm.v3x as cqasm

        self._qubit_list = []
        self._use_postselection = False
        self._cqasm = cqasm

    def count_qubits(self, ast) -> int:
        return len(self._qubit_list)

    def _collect_qubit_list(self, ast):
        self._qubit_list = []
        for variable in ast.variables:
            if type(variable.typ) is self._cqasm.types.QubitArray:
                for i in range(variable.typ.size):
                    self._qubit_list.append((variable.name, i))
            elif type(variable.typ) is self._cqasm.types.Qubit:
                self._qubit_list.append((variable.name, -1))
            else:
                raise ConversionUnsupportedFeatureError(f"Classical variable { variable.name } not supported")

    def _operand_to_qubit_indices(self, operand):
        name = operand.variable.name
        if type(operand) is self._cqasm.values.VariableRef:
            return [self._qubit_list.index((name, -1))]
        elif type(operand) is self._cqasm.values.IndexRef:
            return [self._qubit_list.index((name, index.value))
                      for index in operand.indices]
        else:
            raise ConversionUnsupportedFeatureError(f"Cannot map variable { name } to a declared qubit")

    def _get_gate_inf(self, statement):
        gate_name = statement.name
        num_operands = len(statement.operands)

        # For now, assume the statement pattern is OP q
        targets = self._operand_to_qubit_indices(statement.operands[0])
        controls = []
        parameter = None

        # Match other statement patterns
        if num_operands == 2:
            if type(statement.operands[1]) is self._cqasm.values.ConstFloat:
                # Statement pattern is OP(r) q
                parameter = statement.operands[1].value
            else:
                # Statement pattern is OP q, q
                controls = targets
                targets = self._operand_to_qubit_indices(statement.operands[1])
        elif num_operands == 3:
            # Statement is OP(r) q, q
            controls = targets
            targets = self._operand_to_qubit_indices(statement.operands[1])
            parameter = statement.operands[2].value
        elif num_operands >= 4:
            raise ConversionUnsupportedFeatureError(f"Statement with unsupported number of operands, n = { num_operands }")

        num_controls = len(controls)
        if num_controls >= 2:
            raise ConversionUnsupportedFeatureError(
                f"Gate { gate_name } has more than one control (n = { num_controls })")

        return gate_name, controls, targets, parameter

    def _convert_statement(self, statement, gate_index, optimized_gate_sequence):
        gate_name, controls, targets, parameter = self._get_gate_inf(statement)

        # TODO: refactor all converter code PCVL - 821
        if gate_name == 'X90':
            gate_name = 'rx'
            parameter = np.pi / 2
        elif gate_name == 'mX90':
            gate_name = 'rx'
            parameter = - np.pi / 2
        elif gate_name == 'Y90':
            gate_name = 'ry'
            parameter = np.pi / 2
        elif gate_name == 'mY90':
            gate_name = 'ry'
            parameter = - np.pi / 2

        if not controls:
            # working with 1 qubit gates
            if gate_name.lower() in catalog:
                circuit = self._create_catalog_1_qubit_gate(gate_name.lower(), param=parameter if parameter else None)
            else:
                raise ConversionUnsupportedFeatureError(f"Unsupported 1-qubit gate {gate_name}")

            for target in targets:
                self._converted_processor.add(target * 2, circuit)
        else:
            if gate_name not in _CQASM_2_QUBIT_GATES:
                raise ConversionUnsupportedFeatureError(
                    f"Unsupported 2-qubit gate { gate_name }")
            for target in targets:
                self._create_2_qubit_gates_from_catalog(optimized_gate_sequence[gate_index], controls[0] * 2,
                                                        target * 2, self._use_postselection)

    def convert(self, ast, use_postselection: bool = True) -> Processor:
        r"""Convert a cQASM quantum program into a `Processor`.

        :param ast: the AST of a cQASM program
        :type ast: a Program object, as returned by the cQASM parser
        :param use_postselection: when True, uses a `postprocessed CNOT`
        as the last gate. Otherwise, uses only `heralded CNOT`
        :return: the converted processor
        """
        if isinstance(ast, str):
            return self._convert_from_string(ast, use_postselection)

        get_logger().info(f"Convert cqasm.ast ({len(self._qubit_list)} qubits, {len(ast.block.statements)} operations) to processor",
                    channel.general)
        self._collect_qubit_list(ast)
        self._use_postselection = use_postselection

        qubit_names = [
            f'{ q }[{ i }]' if i >= 0 else q for (q, i) in self._qubit_list]
        self._configure_processor(ast, qubit_names=qubit_names)

        # for gate sequence to optimize cnots
        gate_sequence = []
        for statement in ast.block.statements:
            gate_name, controls, targets, parameter = self._get_gate_inf(statement)
            if len(targets) > 1 and gate_name == 'CNOT':
                get_logger().debug(f"Converting a multi-target CNOT {targets} to multiple Heralded CNOTs", channel.general)
                for i in range(len(targets)):
                    gate_sequence.append([gate_name, controls+[targets[i]]])
            else:
                gate_sequence.append([gate_name, controls + targets])

        optimized_gate_sequence = label_cnots_in_gate_sequence(gate_sequence)

        for gate_index, statement in enumerate(ast.block.statements):
            self._convert_statement(statement, gate_index, optimized_gate_sequence)
        self.apply_input_state()
        return self._converted_processor

    @classmethod
    def check_version(cls, source_string):
        r"""Extracts the version number string from a cQASM program

        :param source_string: the cQASM program stored in a string
        :return: a tuple (major, minor) with the version number
        """
        pattern = r"^\s*version ([0-9])(.[0-9])?\s*"
        match = re.search(pattern, source_string, re.MULTILINE)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)[1:]) if match.group(2) else 0
            return major, minor
        else:
            raise ConversionSyntaxError(f"Missing version number")

    def _convert_from_string(
            self,
            source: str,
            use_postselection: bool = True) -> Processor:
        r"""Convert a cQASM quantum program into a `Processor`.

        :param source: a string containing the cQASM program to convert, or the path to a file storing itm
        :param use_postselection: when True, uses a `postprocessed CNOT`
        as the last gate. Otherwise, uses only `heralded CNOT`
        :return: the converted processor
        """
        source_string = source

        # The string contains a single line, it might be a path
        if not "\n" in source:
            try:
                get_logger().debug(f"Reading cQASM file content at {source}", channel.general)
                with open(source) as source_file:
                    source_string = source_file.read()
            except IOError:
                # File not found, It might be a single line program instead
                # of a name, so we can attempt to parse it.
                pass

        major, minor = CQASMConverter.check_version(source_string)
        if major == 3:
            get_logger().debug("Parsing cQASM v3 description", channel.general)
            ast = self._cqasm.Analyzer().analyze_string(
                source_string)
        elif major == 1:
            get_logger().debug("Converting cQASM v1 to v3", channel.general)
            ast = self._v3_ast_from_v1_source(source_string.split('\n'))
        else:
            raise ConversionBadVersionError(f"Unsupported version {major}.{minor}")

        if not isinstance(ast, self._cqasm.semantic.Program):
            raise ConversionSyntaxError(f"cQASM parser error: { ast[0] }")

        return self.convert(ast, use_postselection=use_postselection)

    def _v3_ast_from_v1_source(self, lines):
        r""""Converts a cQASM v1 quantum program into a cQASM v3 AST"""

        # Parsing code from https://github.com/maxwell04-wq original submission,
        # with just enough changes to make it parse the example on:
        # https://www.quantum-inspire.com/kbase/cqasm/

        cqasm = self._cqasm

        # Create an empty Program object to store the v3 AST
        ast = cqasm.semantic.Program(
            api_version=cqasm.primitives.Version([3]),
            block=cqasm.semantic.Block(
                statements=cqasm.semantic.MultiStatement()),
            variables=cqasm.semantic.MultiVariable()
        )

        def _is_float(ins):
            try:
                float(ins)
                return True
            except ValueError:
                return False

        # read QASM instructions line-by-line
        for line in lines:
            try:
                line = line.strip()
                if len(line) <= 0 or line[0] == '#':
                    #empty line or comment
                    continue

                instruction = line.split(" ")
                if instruction[0] == "version":
                    version = instruction[1]
                    continue

                if instruction[0] == "qubits":
                    if not instruction[1].isdigit():
                        raise ConversionSyntaxError(
                                "Qubit number is not an integer.")
                    typ = cqasm.types.QubitArray(size=int(instruction[1]))
                    ast.variables.append(
                        cqasm.semantic.Variable(name='q', typ=typ))
                    continue

                # Ignore anything with only one keyword:
                if len(instruction) <= 1:
                    continue

                # Ignore classical bits
                if line.find("b[") != -1:
                    continue

                # Ignore non-gate instructions
                if instruction[0] in ['prep_z', 'measure']:
                    continue

                ins_qubits = []
                theta = None
                for remaining_token in instruction[1:]:
                    ins_params = remaining_token.split(",")
                    for param in ins_params:
                        if param.find(":") != -1:
                            raise ConversionSyntaxError(
                                "Unsuported token.")
                        start_indx = param.find("q[")
                        if start_indx != -1:
                            end_indx = param.find("]")
                            ins_qubits.append(param[start_indx+2:end_indx])
                        # Rotation gates
                        elif _is_float(param):
                            theta = float(param)

                if len(ins_qubits) > 2:
                    raise ConversionUnsupportedFeatureError(
                        f"Gate { instruction[0] } has more than one control (n = { len(ins_params) })"
                    )

                # Check if valid gate
                if len(ins_qubits) == 1:
                    if instruction[0] not in _CQASM_1_QUBIT_GATES:
                        raise ConversionUnsupportedFeatureError(
                            f"Unsupported 1-qubit gate { instruction[0] }")

                elif len(ins_qubits) == 2:
                    if instruction[0] not  in _CQASM_2_QUBIT_GATES:
                        raise ConversionUnsupportedFeatureError(
                            f"Unsupported 2-qubit gate { instruction[0] }")

                statement = cqasm.semantic.Instruction()
                statement.name = f'{ instruction[0] }'
                for qubit_index in ins_qubits:
                    index = cqasm.values.IndexRef(
                        variable=cqasm.semantic.Variable(name = 'q'))
                    index.indices = cqasm.values.MultiConstInt()
                    index.indices.append(cqasm.values.ConstInt(qubit_index))
                    statement.operands.append(index)
                if theta is not None:
                    angle = cqasm.values.ConstFloat(value=theta)
                    statement.operands.append(angle)
                ast.block.statements.append(statement)
            except ValueError:
                print("An error in parsing the cQASM v1 file.")
        return ast
