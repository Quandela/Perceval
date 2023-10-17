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

from perceval.algorithm.abstract_algorithm import AAlgorithm
from perceval.algorithm.tomography._tomography_utils import state_to_dens_matrix, compute_matrix, matrix_basis, \
    matrix_to_vector, vector_to_matrix, decomp
from perceval.components import Processor
from typing import List
import numpy


class ATomography(AAlgorithm):
    def __init__(self, nqubit: int, operator_processor: Processor, heralded_modes: List = [], post_process=False,
                 renormalization=None):
        super().__init__(processor=operator_processor)
        self._nqubit = nqubit
        self._operator_processor = operator_processor
        self._backend = operator_processor.backend  # default - SLOSBackend()
        self._heralded_modes = heralded_modes
        # TODO:ask with Stephen if they are always similar to default in Perceval
        self._post_process = post_process
        self._renormalization = renormalization

    def is_physical(self, input_matrix, eigen_tolerance=10 ** (-6)):
        """
        Verifies if a matrix is trace preserving, hermitian, and completely positive (using the Choi matrix)

        :param input_matrix: chi of a quantum map computed from Quantum Process Tomography
        :param eigen_tolerance: brings a tolerance for the positivity of the eigenvalues of the Choi matrix
        :return: bool and string
        """
        # todo: fix implementation and one in QST for density matrix
        d2 = len(input_matrix)
        nqubit = int(numpy.log2(d2) / 2)
        # check if trace preserving
        b = True
        s = ""
        if not numpy.isclose(numpy.trace(input_matrix), 1):
            b = False
            print("trace :", numpy.trace(input_matrix))
            s += "|trace not 1|"

        # check if hermitian
        for i in range(d2):
            for j in range(i, d2):
                if not numpy.isclose(input_matrix[i][j], numpy.conjugate(input_matrix[j][i])):
                    b = False
                    s += "|not hermitian|"
        # todo: find if density matrix is CP too and needs that check

        if b:
            return True
        return False, s
