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

"""Registration of Perceval classes in the archive serialization system."""

import json
from collections.abc import Callable
from typing import TypeVar, Type

from perceval.components import BS, BSLayeredPPNR, Barrier, Circuit, CompiledCircuit, Detector, Experiment, Herald, \
    FFCircuitProvider, FFConfigurator, GenericInterferometer, HWP, LC, PBS, PERM, Port, PR, PS, QWP, TD, Unitary, WP
from perceval.components.compiled_circuit import CompiledCircuitVersion
from perceval.utils import BSCount, BSDistribution, BSSamples, MatrixN, MatrixS, NoiseModel, PostSelect, StateVector, \
    SVDistribution, FockState, AnnotatedFockState, NoisyFockState

from ._circuit_serialization import serialize_circuit, serialize_compiled_circuit, serialize_compiled_circuit_version, \
    serialize_component, serialize_herald, serialize_port
from ._constants import BSC_TAG, BSD_TAG, BSS_TAG, BS_LAYERED_DETECTOR_TAG, COMPILED_CIRCUIT_TAG, \
    COMPILED_CIRCUIT_VERSION_TAG, DETECTOR_TAG, EXPERIMENT_TAG, HERALD_TAG, NOISE_TAG, PORT_TAG, \
    POSTSELECT_TAG, SVD_TAG, SV_TAG, FS_TAG, NFS_TAG, AFS_TAG, MATRIXN_TAG, MATRIXS_TAG
from ._detector_serialization import serialize_bs_layer, serialize_detector
from ._experiment_serialization import serialize_experiment
from ._matrix_serialization import serialize_matrix
from ._state_serialization import serialize_bssamples, serialize_state, serialize_statevector, serialize_svdistribution, \
    serialize_bsdistribution, serialize_bscount
from .deserialize import deserialize_bscount, deserialize_bsdistribution, deserialize_bs_layered_detector, \
    deserialize_bssamples, deserialize_circuit, deserialize_compiled_circuit, deserialize_compiled_circuit_version, \
    deserialize_component, deserialize_detector, deserialize_experiment, deserialize_herald, deserialize_matrix, \
    deserialize_noise_model, deserialize_port, deserialize_postselect, deserialize_state, deserialize_statevector, \
    deserialize_svdistribution
from .library import DescriptorBinary, DescriptorString, Serialization


T = TypeVar("T")
PB_repr = TypeVar("PB_repr")


def _register_binary(
    cls: Type[T],
    tag: str,
    serialize_fn: Callable[[T], PB_repr],
    deserialize_fn: Callable[[bytes], T],
) -> None:
    """Register a class whose canonical representation is a protobuf binary."""
    Serialization.register_class(
        cls,
        class_write_custom=lambda obj, ar: (DescriptorBinary(serialize_fn(obj).SerializeToString()), []),
        class_read_custom=lambda ar, desc, pre_recorder: deserialize_fn(desc.value),
        descriptor_type=DescriptorBinary,
        tag=tag,
    )


def _register_string(
    cls: type,
    tag: str,
    serialize_fn: Callable,
    deserialize_fn: Callable,
) -> None:
    """Register a class whose canonical representation is textual."""
    Serialization.register_class(
        cls,
        class_write_custom=lambda obj, ar: (DescriptorString(serialize_fn(obj)), []),
        class_read_custom=lambda ar, desc, pre_recorder: deserialize_fn(desc.value),
        descriptor_type=DescriptorString,
        tag=tag,
    )


def register_perceval_serializers() -> None:
    """Register every domain class handled by the legacy ``serialize`` API."""
    # Protobuf remains the source representation for these classes.

    # Note: the GenericInterferometer is only serialized as a Circuit, the other members are not serialized
    # I think it's unimportant given they are all private, and the GenericInterferometer is only a way to build a Circuit
    _CIRCUIT_CLASSES = (Circuit, GenericInterferometer)
    _COMPONENT_CLASSES = (FFCircuitProvider, FFConfigurator, LC, TD, BS, Barrier, HWP, PBS, PERM, PR, PS, QWP, Unitary, WP)

    for cls in _CIRCUIT_CLASSES:
        _register_binary(cls, cls.__name__, serialize_circuit, deserialize_circuit)
    for cls in _COMPONENT_CLASSES:
        _register_binary(cls, cls.__name__, serialize_component, deserialize_component)

    _register_binary(Experiment, EXPERIMENT_TAG, serialize_experiment, deserialize_experiment)
    _register_binary(CompiledCircuitVersion, COMPILED_CIRCUIT_VERSION_TAG, serialize_compiled_circuit_version, deserialize_compiled_circuit_version)
    _register_binary(CompiledCircuit, COMPILED_CIRCUIT_TAG, serialize_compiled_circuit, deserialize_compiled_circuit)
    _register_binary(Herald, HERALD_TAG, serialize_herald, deserialize_herald)
    _register_binary(Port, PORT_TAG, serialize_port, deserialize_port)
    _register_binary(MatrixN, MATRIXN_TAG, serialize_matrix, deserialize_matrix)
    _register_binary(MatrixS, MATRIXS_TAG, serialize_matrix, deserialize_matrix)
    _register_binary(BSLayeredPPNR, BS_LAYERED_DETECTOR_TAG, serialize_bs_layer, deserialize_bs_layered_detector)
    _register_binary(Detector, DETECTOR_TAG, serialize_detector, deserialize_detector)

    # Use the str representation for these
    # Register these using new Descriptors to avoid writing size ? Remove outlets |> or {} ?
    _register_string(FockState, FS_TAG, serialize_state, deserialize_state)
    _register_string(NoisyFockState, NFS_TAG, serialize_state, deserialize_state)
    _register_string(AnnotatedFockState, AFS_TAG, serialize_state, deserialize_state)
    _register_string(StateVector, SV_TAG, serialize_statevector, deserialize_statevector)
    _register_string(SVDistribution, SVD_TAG, serialize_svdistribution, deserialize_svdistribution)
    _register_string(BSDistribution, BSD_TAG, serialize_bsdistribution, deserialize_bsdistribution)
    _register_string(BSCount, BSC_TAG, serialize_bscount, deserialize_bscount)
    _register_string(BSSamples, BSS_TAG, serialize_bssamples, deserialize_bssamples)
    _register_string(PostSelect, POSTSELECT_TAG, str, deserialize_postselect)

    Serialization.register_class(NoiseModel,
        [
            "brightness",
            "indistinguishability",
            "g2",
            "g2_distinguishable",
            "transmittance",
            "phase_imprecision",
            "phase_error",
        ],
        tag=NOISE_TAG,
    )
