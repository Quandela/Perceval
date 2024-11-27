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

from .abstract_component import AComponent
from .abstract_processor import AProcessor, ProcessorType
from .linear_circuit import Circuit, ACircuit
from .generic_interferometer import GenericInterferometer
from .processor import Processor
from .source import Source
from ._pauli import (PauliType, PauliEigenStateType, get_pauli_eigen_state_prep_circ,
                     get_pauli_basis_measurement_circuit, get_pauli_gate, get_pauli_eigenvector_matrix,
                     get_pauli_eigenvectors)
from .tomography_exp_configurer import processor_circuit_configurator
from .comp_utils import decompose_perms
from .port import APort, Port, Herald, PortLocation, get_basic_state_from_ports
from .detector import IDetector, DetectionType, Detector, BSLayeredPPNR, get_detection_type
from .unitary_components import BSConvention, BS, PS, WP, HWP, QWP, PR, Unitary, PERM, PBS, Barrier
from .non_unitary_components import TD, LC
from .component_catalog import Catalog
from ._mode_connector import ModeConnector, UnavailableModeException
from .feed_forward_configurator import AFFConfigurator, FFCircuitProvider, FFConfigurator
catalog = Catalog('perceval.components.core_catalog')
