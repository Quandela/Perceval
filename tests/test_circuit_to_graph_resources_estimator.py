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

import pytest
from perceval.converters.circuit_to_graph_converter import CircuitToGraphConverter
from perceval.converters.resources_estimator import ResourcesEstimator
from qiskit.circuit.random import random_circuit


def test_circuit_to_graph_converter():
    qiskit_circuit = random_circuit(8, 10, max_operands=2)
    converter = CircuitToGraphConverter(qiskit_circuit=qiskit_circuit)

    # Generate the graph
    graph = converter.graph_generator()

    # Plot the graph (optional)
    # converter.plot_graph(graph)

    # Check calculation
    single_calc_result = converter.graph_k_clustering_and_cnots_needed()[0]
    cx_count = converter.graph_k_clustering_and_cnots_needed()[1]
    # Check calculation with minimum CNOTs
    min_cnot_calc_result = converter.graph_k_clustering_and_cnots_needed(compute_with_min_cnots=True)[0]
    min_cx_count = converter.graph_k_clustering_and_cnots_needed(compute_with_min_cnots=True)[1]

    assert single_calc_result is not None
    assert all(c >= 0 for c in cx_count)
    assert min_cnot_calc_result is not None
    assert all(c >= 0 for c in min_cx_count)


def test_resources_estimator():
    qiskit_circuit = random_circuit(8, 10, max_operands=2)  # Generate a random circuit for demonstration
    estimator = ResourcesEstimator(qiskit_circuit)

    optimal_encoding = estimator.encoding
    entangling_gates = estimator.needed_entangling_gates
    needed_modes = estimator.needed_modes
    needed_photons = estimator.needed_photons

    assert isinstance(optimal_encoding, list)
    assert entangling_gates >= 0
    assert needed_modes > 0
    assert needed_photons > 0

    custom_encoding = [[0, 1], [2, 3], [4], [5], [6, 7]]
    estimator_with_encoding = ResourcesEstimator(qiskit_circuit, custom_encoding)

    custom_enc = estimator_with_encoding.encoding
    custom_entangling_gates = estimator_with_encoding.needed_entangling_gates
    custom_needed_modes = estimator_with_encoding.needed_modes
    custom_needed_photons = estimator_with_encoding.needed_photons

    assert custom_enc == custom_encoding
    assert custom_entangling_gates >= 0
    assert custom_needed_modes > 0
    assert custom_needed_photons > 0
