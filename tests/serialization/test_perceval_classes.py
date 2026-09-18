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
from packaging.version import Version

from perceval.components import (
    BS,
    BSLayeredPPNR,
    Barrier,
    Circuit,
    CompiledCircuit,
    Detector,
    Experiment,
    FFCircuitProvider,
    FFConfigurator,
    GenericInterferometer,
    Herald,
    HWP,
    LC,
    PBS,
    PERM,
    Port,
    PR,
    PS,
    QWP,
    TD,
    Unitary,
    WP,
)
from perceval.components.compiled_circuit import CompiledCircuitVersion
from perceval.serialization import DescriptorBinary, DescriptorString, InputArchive, OutputArchive, Serialization, serialize
from perceval.utils import BSCount, BSDistribution, BSSamples, BasicState, Encoding, Matrix, NoiseModel, PostSelect, \
    StateVector, SVDistribution


def _compiled_circuit_version():
    version = CompiledCircuitVersion()
    version.hardware_version = Version("0.1.0")
    version.carac_version = Version("2.7")
    version.user_input_mapping = [0, 1]
    version.user_output_mapping = [0, 1]
    version.unused_inputs_mapping = []
    version.unused_outputs_mapping = []
    return version


def _objects_and_descriptors():
    state = BasicState("|1,0>")
    state_vector = StateVector(state)
    sv_distribution = SVDistribution()
    sv_distribution.add(state_vector, 1)
    bs_count = BSCount()
    bs_count[state] = 2
    bs_samples = BSSamples()
    bs_samples.append(state)
    compiled_version = _compiled_circuit_version()

    return (
        (TD(1), DescriptorBinary),
        (Circuit(2) // BS.H(), DescriptorBinary),
        (Experiment(), DescriptorBinary),
        (compiled_version, DescriptorBinary),
        (CompiledCircuit("chip", 2, [], compiled_version), DescriptorBinary),
        (Herald(1), DescriptorBinary),
        (Port(Encoding.DUAL_RAIL, "port"), DescriptorBinary),
        (Matrix.eye(2), DescriptorBinary),
        (BSLayeredPPNR(2), DescriptorBinary),
        (Detector.pnr(), DescriptorBinary),
        (state, DescriptorString),
        (state_vector, DescriptorString),
        (sv_distribution, DescriptorString),
        (BSDistribution(state), DescriptorString),
        (bs_count, DescriptorString),
        (bs_samples, DescriptorString),
        (PostSelect("[0] == 1"), DescriptorString),
    )


@pytest.mark.parametrize("obj, descriptor_type", _objects_and_descriptors())
def test_perceval_class_archive_round_trip(obj, descriptor_type):
    archive = OutputArchive()
    Serialization.serialize(obj, archive)

    assert isinstance(archive.memo[archive.roots[0]][1], descriptor_type)

    restored = Serialization.deserialize(
        InputArchive.from_text(archive.to_text())
    )
    assert serialize(restored, compress=False) == serialize(
        obj, compress=False
    )


def _component_objects():
    circuit = Circuit(2) // BS.H()
    return (
        Circuit(2),
        GenericInterferometer(2, lambda _: BS.H()),
        BS(),
        Barrier(2),
        HWP(.1),
        PBS(),
        PERM([1, 0]),
        PR(.1),
        PS(.1),
        QWP(.1),
        Unitary(Matrix.eye(2)),
        WP(.1, .2),
        FFCircuitProvider(1, 0, circuit),
        FFConfigurator(1, 0, circuit, {}),
        LC(.1),
        TD(1),
    )


@pytest.mark.parametrize("component", _component_objects())
def test_component_classes_are_registered_explicitly(component):
    archive = OutputArchive()
    Serialization.serialize(component, archive)

    assert archive.memo[archive.roots[0]][0] == type(component).__name__

    restored = Serialization.deserialize(
        InputArchive.from_text(archive.to_text())
    )
    assert serialize(restored, compress=False) == serialize(
        component, compress=False
    )


def test_unregistered_component_subclass_is_rejected():
    class DerivedCircuit(Circuit):
        pass

    with pytest.raises(RuntimeError, match="unhandled class 'DerivedCircuit'"):
        Serialization.serialize(DerivedCircuit(2), OutputArchive())

def test_noise_model():
    nm = NoiseModel(.5)
    archive = OutputArchive()
    Serialization.serialize(nm, archive)

    assert archive.memo[archive.roots[0]][0] == "NoiseModel"

    restored = Serialization.deserialize(InputArchive.from_text(archive.to_text()))
    assert nm == restored
