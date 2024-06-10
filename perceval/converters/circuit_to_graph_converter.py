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
# Main processing
# 1. Function Qiskit gate based circuit => list of gates and modes they act on
# 2. Convert the previous list to a weighted graph
# 3. Cut the graph the best way possible
# 4. Plot the graph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit


def gates_and_qubits(qiskit_circuit):
    '''''
    :rtype gates_names: list of strings
    :rtype gates_qubits: list of lists of ints
    '''''
    # will be used in both classes.
    gates_names = []
    gates_qubits = []
    for instr in qiskit_circuit.data:
        # Extract gate name and qubits
        gates_names.append(instr[0].name)
        qubits = instr[1]
        qubit_numbers = [qiskit_circuit.find_bit(q).index for q in qubits]
        gates_qubits.append(qubit_numbers)
    return gates_names, gates_qubits


class CircuitToGraphConverter:
    """''
    Takes a qiskit circuit and converts it into a graph.
    Initially get the list of gates and the qubits on which each gate is acting on.
    With this output or if the user already has the two lists, it converts it into
    a graph where the vertices are the qubits and the edges the gates acting on them,
    the weight of its edges depends on the type of gate.
    :param qiskit_circuit: Quantum circuit to convert.
    :type qiskit_circuit: QuantumCircuit
    '"""

    def __init__(self, qiskit_circuit: QuantumCircuit = None, gates=None, qubits=None):
        '''''
        :type gates: list of strings
        :type qubits: list of lists of ints
        '''''
        if gates is not None and qubits is not None:
            self.gates = gates
            self.qubits = qubits
        elif qiskit_circuit is not None:
            self.gates, self.qubits = gates_and_qubits(qiskit_circuit)
        else:
            raise ValueError("Either a Qiskit circuit or both gates and qubits lists must be provided")

    def graph_generator(self) -> nx.Graph:
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
                    elif self.gates[l] in ['swap', 'iswap']:
                        # Free operations.
                        pass
                    else:
                        if edge in edge_weights:
                            edge_weights[edge] += 2
                        else:
                            edge_weights[edge] = 2

        for edge, weight in edge_weights.items():
            g.add_edge(*edge, weight=weight)
        return g

    @staticmethod
    def plot_graph(g):  # g is the graph created in graph_generator
        pos = nx.spring_layout(g, seed=42)
        nx.draw_networkx_nodes(g, pos, node_size=90, node_color='b')
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, font_size=10, font_color='white', font_family="sans-serif")
        edge_labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels)
        plt.show()

    def graph_k_clustering_and_cnots_needed(self,
                                            compute_with_min_cnots: bool = False) -> tuple[list[list[int]], list[int]]:
        """''
        Computes the laplacian matrix of the graph, compute its eigenvectors sorted by their
        eigenvalues. For all the possible subpartitions of the graph, it will compute the
        kmeans clustering with the respective number of eigenvectors as features.
        Given the random initial state for the kmeans method, the user has the option to compute
        multiple repetitions and choose the partition that gives the smallest number of CNOTs.
        :param compute_with_min_cnots: Whether to compute it 30 times and choose the output yielding to the minimum of CNOTs.
        :type compute_with_min_cnots: bool
        :return: Clustering result and CNOT counts.
        :rtype: tuple[list[list[int]], list[int]]
        '"""
        graph = self.graph_generator()
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
                for _ in range(30):
                    kmeans = KMeans(n_clusters=k)
                    labels = kmeans.fit_predict(features)

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
                kmeans = KMeans(n_clusters=k)
                labels = kmeans.fit_predict(features)
                possible_partitions.append(labels)

                real_weight = 0
                for u, v in graph.edges():
                    if labels[u] != labels[v]:
                        a = len(np.where(labels == labels[u])[0])
                        b = len(np.where(labels == labels[v])[0])
                        real_weight += graph[u][v]['weight'] * 2 ** (a + b - 2)
                number_of_cnots.append(real_weight)
            # Conversor from ([0,0,1,1,0]) to [[0,1,4],[2,3]] format

            possible_partitions_qudit_format = []
            for labels in possible_partitions:
                partition_dict = {}
                for index, label in enumerate(labels):
                    if label not in partition_dict:
                        partition_dict[label] = []
                    partition_dict[label].append(index)
                possible_partitions_qudit_format.append(list(partition_dict.values()))
        return possible_partitions_qudit_format, number_of_cnots
