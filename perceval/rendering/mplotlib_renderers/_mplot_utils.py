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

import matplotlib
import os
import platform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy
import networkx as nx


def autoselect_backend():
    try:
        # The next line may raise an exception if the backend needs to be autodetected, because gtk* candidate backends
        # require cairo which is not available on Windows
        matplotlib.rcParams['backend']
    except Exception:  # We cannot guess the exception type we need to catch here: it can come from any Matplotlib
        # backend or third party. We do not have control over this code

        # In order to avoid matplotlib trying to use cairo (which is a dependency of cairocffi retrieved by drawsvg),
        # hint the backend given the execution context, and avoid cairo related backends at all cost!
        in_notebook = False
        in_pycharm_or_spyder = "PYCHARM_HOSTED" in os.environ or 'SPY_PYTHONPATH' in os.environ

        try:
            from IPython import get_ipython
            in_notebook = 'IPKernelApp' in get_ipython().config
        except (ImportError, AttributeError):
            pass

        try:
            if in_pycharm_or_spyder:
                matplotlib.use("module://backend_interagg")
            elif in_notebook:
                matplotlib.use("module://matplotlib_inline.backend_inline")
            elif platform.system() == "Darwin":
                matplotlib.use("MacOSX")
            else:
                import tkinter
                matplotlib.use("TkAgg")
        except Exception:  # We want to catch anything that can happen above
            # Last chance: use "agg" non-interactive backend (which should work "anywhere").
            matplotlib.use("agg")

def _get_sub_figure(ax: Axes3D, array: numpy.array, basis_name: list):
    # Data
    size = array.shape[0]
    x = numpy.array([[i] * size for i in range(size)]).ravel()  # x coordinates of each bar
    y = numpy.array([i for i in range(size)] * size)  # y coordinates of each bar
    z = numpy.zeros(size * size)  # z coordinates of each bar
    dxy = numpy.ones(size * size) * 0.5  # Width/Lenght of each bar
    dz = array.ravel()  # length along z-axis of each bar (height)

    # Colors
    # get range of colorbars so we can normalize
    max_height = numpy.max(dz)
    min_height = numpy.min(dz)
    color_map = plt.get_cmap('viridis_r')
    if max_height != min_height:
        has_only_one_value = False
        # scale each z to [0,1], and get their rgb values
        rgba = [color_map((k - min_height) / max_height) for k in dz]
    else:
        has_only_one_value = True
        rgba = [color_map(0)]


    # Caption
    font_size = 6

    # XY
    ax.set_xticks(numpy.arange(size) + 1)
    ax.set_yticks(numpy.arange(size) + 1)
    ax.tick_params(axis='x', which='major', labelsize=font_size)
    ax.set_xticklabels(basis_name)
    ax.tick_params(axis='y', which='major', labelsize=font_size)
    ax.set_yticklabels(basis_name)

    # Z
    if not has_only_one_value:
        ax.set_zlim(zmin=dz.min(), zmax=dz.max())
    ax.tick_params('z', which='both', labelsize=font_size)
    ax.grid(True, axis='z', which='major', linewidth=2)
    # interval = [v for v in ax.get_zticks() if v > 0][0]
    # ax.zaxis.set_minor_locator(ticker.MultipleLocator(interval/5))

    # Plot
    ax.bar3d(x, y, z, dxy, dxy, dz, color=rgba, alpha=0.7)
    ax.view_init(elev=30, azim=45)
