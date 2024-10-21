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
import numpy as np

has_qiskit = True
try:
    from qiskit.circuit.random import random_circuit
except ModuleNotFoundError as e:
    assert e.name == "qiskit"
    pytest.skip("need `qiskit` module", allow_module_level=True)


from perceval.converters import CircuitToGraphConverter
from perceval.converters import ResourcesEstimator
from perceval.utils.qmath import kmeans
from perceval import pdisplay
import matplotlib.pyplot as plt


def test_kmeans():
    # Simple dataset with two clear clusters
    data = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]])
    expected_labels = [0, 0, 1, 1]

    labels = kmeans(data, n_clusters=2, n_init=10)

    assert set(labels) == {0, 1}  # Ensure we have two clusters
    assert (np.array_equal(np.sort(labels[:2]), np.sort(expected_labels[:2])) or
            np.array_equal(np.sort(labels[:2]), np.sort(expected_labels[2:])))  # Check first cluster
    assert (np.array_equal(np.sort(labels[2:]), np.sort(expected_labels[2:])) or
            np.array_equal(np.sort(labels[2:]), np.sort(expected_labels[:2])))  # Check second cluster

    # Test for a larger number of clusters
    data = np.random.rand(100, 2)
    labels = kmeans(data, n_clusters=5, n_init=10)
    assert len(set(labels)) == 5  # Ensure we have 5 clusters


def test_circuit_to_graph_converter():
    qiskit_circuit = random_circuit(8, 10, max_operands=2)
    converter = CircuitToGraphConverter(qiskit_circuit=qiskit_circuit)

    # Check calculation
    single_calc_result, cx_count = converter.graph_k_clustering_and_cnots_needed()
    # Check calculation with minimum CNOTs
    min_cnot_calc_result, min_cx_count = converter.graph_k_clustering_and_cnots_needed(compute_with_min_cnots=True)

    assert single_calc_result is not None
    assert all(c >= 0 for c in cx_count)
    assert min_cnot_calc_result is not None
    assert all(c >= 0 for c in min_cx_count)


def test_resources_estimator():
    qiskit_circuit = random_circuit(8, 10, max_operands=2)  # Generate a random circuit for demonstration
    estimator = ResourcesEstimator(qiskit_circuit)

    assert type(estimator.encoding) == list
    assert estimator.num_entangling_gates_needed > 0
    assert estimator.num_modes_needed > 0
    assert estimator.num_photons_needed > 0

    custom_encoding = [[0, 1], [2, 3], [4], [5], [6, 7]]
    estimator_with_encoding = ResourcesEstimator(qiskit_circuit, custom_encoding)

    assert estimator_with_encoding.encoding == custom_encoding
    assert estimator_with_encoding.num_entangling_gates_needed >= 0
    assert estimator_with_encoding.num_modes_needed > 0
    assert estimator_with_encoding.num_photons_needed > 0
