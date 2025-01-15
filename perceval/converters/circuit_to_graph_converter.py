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

# ========================================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from perceval.utils.qmath import kmeans


def gates_and_qubits(qiskit_circuit) -> tuple[list[str], list[list[int]]]:
    # will be used in CircuitToGraphConverter and in ResourcesEstimator.
    gates_names = []
    gates_qubits = []
    for instr in qiskit_circuit.data:
        # Extract gate name and qubits
        gates_names.append(instr.operation.name)
        qubits = instr.qubits
        qubit_numbers = [qiskit_circuit.find_bit(q).index for q in qubits]
        gates_qubits.append(qubit_numbers)
    return gates_names, gates_qubits


NUMBER_REPETITIONS_IF_MIN_CNOTS = 30


class CircuitToGraphConverter:
    """
    Takes a qiskit circuit and converts it into a graph.
    Initially get the list of gates and the qubits on which each gate is acting on.
    With this output or if the user already has the two lists, it converts it into
    a graph where the vertices are the qubits and the edges the gates acting on them,
    the weight of its edges depends on the type of gate.
    """

    def __init__(self, qiskit_circuit=None, gates=None, qubits=None):
        """
        :param qiskit_circuit: Quantum circuit to convert.
        :type gates: list of strings
        :type qubits: list of lists of ints
        """
        if gates is not None and qubits is not None:
            self.gates = gates
            self.qubits = qubits
        elif qiskit_circuit is not None:
            self.gates, self.qubits = gates_and_qubits(qiskit_circuit)
        else:
            raise ValueError("Either a Qiskit circuit or both gates and qubits lists must be provided")

    def generate_graph(self) -> nx.Graph:
        """
        The qubits of the circuit are interpreted as nodes. The weight of the edges
        connecting them will depend on the number and type of gates.
        """
        elements_set = set(item for sublist in self.qubits for item in sublist)  # nodes
        g = nx.Graph()
        g.add_nodes_from(elements_set)
        edge_weights = {}

        # Iterate over every gate and add edges between pairs of nodes
        for l in range(len(self.gates)):
            subset = sorted(self.qubits[l])
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    edge = (subset[i], subset[j])
                    # Increase the weight of the edge by 1 if the gate is in the list. Increase by 2 if not.
                    if self.gates[l] in ['cx', 'cy', 'cz', 'ch', 'cp']:
                        if edge in edge_weights:
                            edge_weights[edge] += 1
                        else:
                            edge_weights[edge] = 1
                    else:
                        if edge in edge_weights:
                            edge_weights[edge] += 2
                        else:
                            edge_weights[edge] = 2

        for edge, weight in edge_weights.items():
            g.add_edge(*edge, weight=weight)
        return g

    def graph_k_clustering_and_cnots_needed(self,
                                            compute_with_min_cnots: bool = False) -> tuple[list[list[int]], list[int]]:
        """
        Computes the laplacian matrix of the graph, compute its eigenvectors sorted by their
        eigenvalues. For all the possible subpartitions of the graph, it will compute the
        kmeans clustering with the respective number of eigenvectors as features.
        Given the random initial state for the kmeans method, the user has the option to compute
        multiple repetitions and choose the partition that gives the smallest number of CNOTs.

        :param compute_with_min_cnots: Whether to compute it 30 times and choose the output yielding to the minimum of CNOTs.
        :return: Clustering result and CNOT counts.
        """
        graph = self.generate_graph()
        laplacian_matrix = nx.normalized_laplacian_matrix(graph).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        n = len(graph.nodes)  # number of qubits
        possible_partitions = []
        number_of_cnots = []

        for k in range(2, n + 1):
            # 1 is the trivial case of the complete circuit in one qudit and n is the dual rail.
            features = eigenvectors[:, :k]
            if compute_with_min_cnots:
                best_partition = None
                min_cnots = float('inf')
                for _ in range(NUMBER_REPETITIONS_IF_MIN_CNOTS):
                    labels = kmeans(features, k)
                    real_weight = 0
                    for u, v in graph.edges():
                        # Check if the nodes belong to different clusters
                        if labels[u] != labels[v]:
                            a = len(np.where(labels == labels[u])[0])
                            b = len(np.where(labels == labels[v])[0])
                            real_weight += graph[u][v]['weight'] * 2 ** (a + b - 2)  # Number of CNOTs

                    if real_weight < min_cnots:
                        min_cnots = real_weight
                        best_partition = labels
                possible_partitions.append(best_partition)
                number_of_cnots.append(min_cnots)
            else:
                labels = kmeans(features, k)
                possible_partitions.append(labels)

                real_weight = 0
                for u, v in graph.edges():
                    if labels[u] != labels[v]:
                        a = len(np.where(labels == labels[u])[0])
                        b = len(np.where(labels == labels[v])[0])
                        real_weight += graph[u][v]['weight'] * 2 ** (a + b - 2)
                number_of_cnots.append(real_weight)
            # Convertor from ([0,0,1,1,0]) to [[0,1,4],[2,3]] format

            possible_partitions_qudit_format = []
            for labels in possible_partitions:
                partition_dict = {}
                for index, label in enumerate(labels):
                    if label not in partition_dict:
                        partition_dict[label] = []
                    partition_dict[label].append(index)
                possible_partitions_qudit_format.append(list(partition_dict.values()))
        return possible_partitions_qudit_format, number_of_cnots
