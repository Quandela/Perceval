from perceval.components import Processor, Source
from .abstract_converter import AGateConverter
import numpy as np

class QASMConverter(AGateConverter):
    r"""QASM to perceval processor converter.

    :param catalog: 
    :param backend_name: Backend to use in computation, defaults to SLOS
    :param source: 
    """
    def __init__(self, catalog, backend_name: str = "SLOS", source: Source = Source()):
        super().__init__(catalog, backend_name, source)

    def count_qubits(self, filename) -> int:
        lines = open(filename, 'r').readlines()
        n_qubits = 0
        for line in lines:
            line = line.rstrip('\n')
            ins = line.split(" ")
            if ins[0]=="qubits":
                n_qubits = int(ins[1])
                break
        return n_qubits  # number of qubits

    def convert(self, filename, use_postselection: bool = True) -> Processor:
        r"""Convert a QASM instruction file into a `Processor`.

        
        :return: the converted processor
        """
        lines = open(filename, 'r').readlines() # read the QASM file
        n_cnot = 0  # count the number of CNOT gates in circuit - needed to find the num. heralds
        version = 0 # QASM file version
        n_qbits = 0 # number of qubits

        qasm_gate = {
            "i": np.array([[1., 0.], [0., 1.]]),
            "h": np.array([[0.707, 0.707], [0.707, -0.707]]),
            "x": np.array([[0., 1.], [1., 0.]]),
            "y": np.array([[ 0.+0.j, -0.-1.j], [0.+1.j,  0.+0.j]], dtype=complex),
            "z": np.array([[1., 0.], [0., -1.]]),
            # todo: Implement parametrized gates
            # "rx": ,
            # "ry": ,
            # "rz": ,
            "s": np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.+1.j]], dtype=complex),
            "sdag": np.array([[1, 0], [0, -1j]], dtype=complex),
            "t": np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=complex),
            "tdag": np.array([[1, 0], [0, (1 - 1j) / np.sqrt(2)]], dtype=complex),
            "cnot": np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]]),
            "cx": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]]),
            "toffoli": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 1, 0]]),
            "swap": np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]], dtype=complex),
            # "crk": ,
            # "cr": ,
            "c-X": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]]),
            "c-Z": np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]]),
            "x90": np.array([[1/ np.sqrt(2), -(0+1j)/ np.sqrt(2)], [-(0+1j)/ np.sqrt(2), 1/ np.sqrt(2)]]),
            "mx90": np.array([[1/ np.sqrt(2), (0+1j)/ np.sqrt(2)], [(0+1j)/ np.sqrt(2), 1/ np.sqrt(2)]]),
            "y90": np.array([[1/ np.sqrt(2), -1/ np.sqrt(2)], [1/ np.sqrt(2), 1/ np.sqrt(2)]]),
            "my90": np.array([[1/ np.sqrt(2), 1/ np.sqrt(2)], [-1/ np.sqrt(2), 1/ np.sqrt(2)]]),
        }

        for line in lines: # read QASM instructions line-by-line
            if line[0] == ' ' or len(line) == 1 or line[0] == '#': #empty line or comment
                continue

            line = line.rstrip('\n')
            instruction = line.split(" ")
            
            if instruction[0] == "version":
                version = instruction[1]
                continue

            if instruction[0] == "qubits":
                n_qbits = instruction[1]
                continue

            # This code is written with the assumption that "qubits" is the second instruction in the QASM file.
            qubit_names = 'q' #change this according to cQASM format
            self._configure_processor_from_nqubits(n_qbits, qname=qubit_names)  # empty processor with ports initialized

            if instruction[0] == "CNOT" or instruction[0] == "c-X":
                n_cnot += 1

            # barrier has no effect
            if instruction[0] == "barrier":
                continue

            if instruction[1].find("b[") == -1: # skip the instruction in case of classical bit 
                continue
            # some limitation in the conversion, in particular measure
            assert instruction[0] in qasm_gate.keys(), "cannot convert (%s)" % instruction[0] # check for valid gates

            ins_n_qubits = 0
            ins_qubits = []
            ins_params = instruction[1].split(",")
            for param in ins_params:
                indx = param.find("q[")
                if indx  != -1:
                    ins_qubits.append(param[indx+2])
            ins_n_qubits = len(ins_qubits)

            if ins_n_qubits == 1:
                # one mode gate
                ins = self._create_generic_1_qubit_gate(qasm_gate[instruction[0]])
                ins_name = instruction[0].name
                self._converted_processor.add(ins_qubits[0] * 2, ins.copy())
            else:
                if ins_n_qubits > 2:
                    # only 2 qubit gates
                    raise ValueError("Gates with number of Qubits higher than 2 not implemented")
                c_idx = ins_qubits[0] * 2
                c_data = ins_qubits[0] * 2
                c_first = min(c_idx, c_data)

                self._create_2_qubit_gates_from_catalog(instruction[0], n_cnot, c_idx, c_data, c_first,
                                                           use_postselection)
        self.apply_input_state()
        return self._converted_processor
