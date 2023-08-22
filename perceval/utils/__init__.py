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

from .matrix import Matrix, MatrixN, MatrixS, matrix_double
from .format import simple_float, simple_complex, format_parameters
from .parameter import Parameter, P, Expression, E
from .mlstr import mlstr
from .statevector import BasicState, StateVector, SVDistribution, BSDistribution, BSCount, BSSamples, \
    tensorproduct, AnnotatedBasicState, allstate_iterator, anonymize_annotations
from .polarization import Polarization, convert_polarized_state, build_spatial_output_states
from .postselect import PostSelect
from ._random import random_seed
from .globals import global_params
from .conversion import samples_to_sample_count, samples_to_probs, sample_count_to_samples, sample_count_to_probs,\
    probs_to_samples, probs_to_sample_count
from .stategenerator import StateGenerator, Encoding
from exqalibur import Annotation  # Used to provide the Annotation class to the perceval root namespace
