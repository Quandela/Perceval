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

SEP = ":"
PCVL_PREFIX = f"{SEP}PCVL{SEP}"
ZIP_PREFIX = f"{PCVL_PREFIX}zip{SEP}"

MATRIX_TAG = "Matrix"
CIRCUIT_TAG = "ACircuit"
COMPONENT_TAG = "Component"
EXPERIMENT_TAG = "Experiment"
HERALD_TAG = "Herald"
PORT_TAG = "Port"
BS_TAG = "BasicState"
SV_TAG = "StateVector"
SVD_TAG = "SVDistribution"
BSD_TAG = "BSDistribution"
BSC_TAG = "BSCount"
BSS_TAG = "BSSamples"
NOISE_TAG = "NoiseModel"
POSTSELECT_TAG = "PostSelect"
BS_LAYERED_DETECTOR_TAG = "BSLayeredDetector"
DETECTOR_TAG = "Detector"

VALUE_NOT_SET = 0x0fffffff  # Maximum writable value
