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


import matplotlib.pyplot as plt
from .._density_matrix_utils import _csr_to_rgb, _csr_to_greyscale, generate_ticks

class DensityMatrixRenderer:
    def __init__(self,
        color: bool = True,
        cmap='hsv',
        mplot_noshow: bool = False,
        mplot_savefig: str = None):
        """
        :meta private:
        :param output_format:
        :param color: whether to display the phase according to some circular cmap
        :param cmap: the cmap to use fpr the phase indication
        """
        self.color = color
        self.cmap = cmap
        self.mplot_noshow = mplot_noshow
        self.mplot_savefig = mplot_savefig

    def render(self, dm):
        """
        :meta private:
        :param dm: density matrix to be displayed
        """

        fig = plt.figure()

        if self.color:
            img = _csr_to_rgb(dm.mat, self.cmap)
            plt.imshow(img)
        else:
            img = _csr_to_greyscale(dm.mat)
            plt.imshow(img, cmap='gray')

        l1, l2 = generate_ticks(dm)

        plt.yticks(l1, l2)
        plt.xticks([])

        if not self.mplot_noshow:
            plt.show()
        if self.mplot_savefig:
            fig.savefig(self.mplot_savefig, bbox_inches="tight", format="svg")
            return ""
