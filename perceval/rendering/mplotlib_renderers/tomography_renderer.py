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


import math

import numpy
from perceval.algorithm.tomography.abstract_process_tomography import AProcessTomography
import matplotlib.pyplot as plt

from ..mplotlib_renderers._mplot_utils import _get_sub_figure

def _generate_pauli_captions(nqubit: int):
    from perceval.algorithm.tomography.tomography_utils import _generate_pauli_index
    pauli_indices = _generate_pauli_index(nqubit)
    pauli_names = []
    for subset in pauli_indices:
        pauli_names.append([member.name for member in subset])

    basis = []
    for val in pauli_names:
        basis.append(''.join(val))
    return basis

class TomographyRenderer:
    def __init__(self, render_size, mplot_noshow: bool, mplot_savefig: str):
        self.render_size = render_size
        self.mplot_noshow = mplot_noshow
        self.mplot_savefig = mplot_savefig

    def render(self, qpt: AProcessTomography, precision: float):

        chi_op = qpt.chi_matrix()

        if self.render_size is not None and isinstance(self.render_size, tuple) and len(self.render_size) == 2:
            fig = plt.figure(figsize=self.render_size)
        else:
            fig = plt.figure()
        pauli_captions = _generate_pauli_captions(qpt._nqubit)
        significant_digit = int(math.log10(1 / precision))

        # Real plot
        ax = fig.add_subplot(121, projection='3d')
        ax.set_title("Re[$\\chi$]")
        real_chi = numpy.round(chi_op.real, significant_digit)
        _get_sub_figure(ax, real_chi, pauli_captions)

        # Imag plot
        ax = fig.add_subplot(122, projection='3d')
        ax.set_title("Im[$\\chi$]")
        imag_chi = numpy.round(chi_op.imag, significant_digit)
        _get_sub_figure(ax, imag_chi, pauli_captions)

        if not self.mplot_noshow:
            plt.show()
        if self.mplot_savefig:
            fig.savefig(self.mplot_savefig, bbox_inches="tight", format="svg")
            return ""

        return None
