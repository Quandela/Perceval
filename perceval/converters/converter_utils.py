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

from itertools import combinations

CNOT_NAMES = ['CX', 'CNOT']

def _is_cyclic_util(v: int, visited: list, parent: int, adj_list: list) -> bool:
    visited[v] = True
    for i in adj_list[v]:
        if not visited[i]:
            if _is_cyclic_util(i, visited, v, adj_list):
                return True
        elif parent != i:
            return True
    return False

def _is_cyclic(adj_list: list, cnot_node_count: int) -> bool:
    # returns True if a cyclic pair is found
    visited = [False] * cnot_node_count
    for i in range(cnot_node_count):
        if not visited[i]:
            if _is_cyclic_util(i, visited, -1, adj_list):
                return True
    return False

def _find_max_ralph_pairs(pairs) -> list:
    # finds largest set of acyclic edges -> Ralph CNOTs can be added to these positions
    nodes = set()  # create a set of all positions (mode indices) at which CNOT exists
    for pair in pairs:
        nodes.update(pair)

    node_map = {node: idx for idx, node in enumerate(nodes)}
    cnot_node_count = len(nodes)  # num of cnots

    max_subset = []  # list of pairs of acyclic combinations of modes

    # Check all possible subsets of the pairs list
    for r in range(1, len(pairs) + 1):
        for subset in combinations(pairs, r):  # creating subsets of pairs taking r at a time
            # Build adjacency list for each subset - each element of this list corresponds to
            # each node (or vertex) and has a list of its adjacent nodes (neighbors)
            adj_list = [[] for _ in range(cnot_node_count)]
            for u, v in subset:
                adj_list[node_map[u]].append(node_map[v])
                adj_list[node_map[v]].append(node_map[u])

            # Check if this subset forms a cycle
            if not _is_cyclic(adj_list, cnot_node_count):
                if len(subset) > len(max_subset):
                    max_subset = subset

    return list(max_subset)


def _gate_list_optimized_cnots(gate_info: list) -> list:
    """
    Optimizes the placement of CNOT gates within the converted circuit.
    Extracts relevant information from the gate sequency to decide where
    to insert Ralph or Knill CNOTs

    :param gate_info: list of gate sequences with corresponding gate names and positions
    :return: list of labelled CNOTs (postprocessed or heralded) in sequential order
    """

    cnot_list_in_order = [elem for elem in gate_info if elem[0].upper() in CNOT_NAMES]  # extracts CNOTs in sequence from gate list
    cnot_pos_pairs = [elem[1] for elem in cnot_list_in_order]  # list of CNOT qubit pos pairs in order of appearance

    ralph_pairs_list = _find_max_ralph_pairs(cnot_pos_pairs[::-1])

    # CNOT
    cnot_type_list = []

    # This for loop generates the list for cnot_types - Knill or Ralph
    for pair in cnot_pos_pairs[::-1]:
        if pair in ralph_pairs_list:
            cnot_type_list.append('postprocessed cnot')
            ralph_pairs_list.remove(pair)
        else:
            cnot_type_list.append('heralded cnot')

    cnot_type_list.reverse()  # required as it was created by going through CNOTs in reverse

    cnot_order_named = []  # List of named CNOTs to be im[;emented in converted circuit
    for index, cnot_elem in enumerate(cnot_list_in_order):
        cnot_order_named.append([cnot_type_list[index], cnot_elem[1]])  # cnot_elem[1] -> qubit positions

    return cnot_order_named


def label_cnots_in_gate_sequence(gate_info: list) -> list:
    """
    Processes a list of gate sequence to return an optimized list of gate names.
    During the conversion from a gate-based circuit to a linear optical circuit,
    CNOTs need to be classified as either Ralph or Knill for optimized use of
    resources in the circuit.

    :param gate_info: list of gate sequences with corresponding gate names and positions
    """
    cnot_order_named = _gate_list_optimized_cnots(gate_info)
    # generate gate info with CNOT names
    cnot_counter = 0
    for i, elem in enumerate(gate_info):
        if elem[0].upper() in CNOT_NAMES:
            gate_info[i] = cnot_order_named[cnot_counter]
            cnot_counter += 1

    return [elem[0] for elem in gate_info]  # extract list of gate names
