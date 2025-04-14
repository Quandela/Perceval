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
from __future__ import annotations

from base64 import b64decode
from os import path
import json
from zlib import decompress

from perceval.components import Circuit, BSLayeredPPNR, Detector, AComponent, Experiment, PortLocation, Port, Herald
from perceval.utils import Matrix, BSDistribution, SVDistribution, BasicState, BSCount, NoiseModel, PostSelect
from perceval.serialization import _matrix_serialization, deserialize_state, _detector_serialization
from ._port_deserialization import deserialize_herald, deserialize_port
from ._constants import (
    SEP,
    PCVL_PREFIX,
    ZIP_PREFIX,
    BS_TAG,
    SV_TAG,
    SVD_TAG,
    BSD_TAG,
    BSC_TAG,
    BSS_TAG,
    MATRIX_TAG,
    CIRCUIT_TAG,
    NOISE_TAG,
    POSTSELECT_TAG,
    BS_LAYERED_DETECTOR_TAG,
    DETECTOR_TAG, COMPONENT_TAG, HERALD_TAG, PORT_TAG, VALUE_NOT_SET, EXPERIMENT_TAG,
)
from ._state_serialization import deserialize_statevector, deserialize_bssamples
from . import _component_deserialization as _cd
from . import _schema_circuit_pb2 as pb


def deserialize_float(floatstring):
    return float(floatstring)


def deserialize_matrix(pb_mat: str | pb.Matrix) -> Matrix:
    if not isinstance(pb_mat, pb.Matrix):
        pb_binary_repr = pb_mat
        pb_mat = pb.Matrix()
        if isinstance(pb_binary_repr, bytes):
            pb_mat.ParseFromString(pb_binary_repr)
        else:
            pb_mat.ParseFromString(b64decode(pb_binary_repr))
    return _matrix_serialization.deserialize_pb_matrix(pb_mat)


def matrix_from_file(filepath: str) -> Matrix:
    """
    Deserialize a matrix from a binary file
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_matrix(f.read())


def deserialize_circuit(pb_circ: str | bytes | pb.Circuit, known_params: dict = None) -> Circuit:
    if not isinstance(pb_circ, pb.Circuit):
        pb_binary_repr = pb_circ
        pb_circ = pb.Circuit()
        if isinstance(pb_binary_repr, bytes):
            pb_circ.ParseFromString(pb_binary_repr)
        else:
            pb_circ.ParseFromString(b64decode(pb_binary_repr))
    builder = CircuitBuilder(pb_circ.n_mode, pb_circ.name, known_params)
    for pb_c in pb_circ.components:
        builder.add(pb_c)
    return builder.retrieve()


def circuit_from_file(filepath: str) -> Circuit:
    """
    Deserialize a circuit from a binary file
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'rb') as f:
        return deserialize_circuit(f.read())


def deserialize_component(pb_c: pb.Component, known_params: dict = None) -> AComponent:
    if not isinstance(pb_c, pb.Component):
        pb_binary_repr = pb_c
        pb_c = pb.Component()
        if isinstance(pb_binary_repr, bytes):
            pb_c.ParseFromString(pb_binary_repr)
        else:
            pb_c.ParseFromString(b64decode(pb_binary_repr))

    return CircuitBuilder.deserialize(pb_c, known_params)


def deserialize_experiment(pb_e: pb.Experiment, known_params: dict = None) -> Experiment:
    if not isinstance(pb_e, pb.Experiment):
        pb_binary_repr = pb_e
        pb_e = pb.Experiment()
        if isinstance(pb_binary_repr, bytes):
            pb_e.ParseFromString(pb_binary_repr)
        else:
            pb_e.ParseFromString(b64decode(pb_binary_repr))

    builder = ExperimentBuilder(pb_e, known_params)
    return builder.resolve()


def deserialize_svdistribution(serial_svd) -> SVDistribution:
    assert serial_svd[0] == '{' and serial_svd[-1] == '}', "Invalid serialized SVDistribution"
    if len(serial_svd) == 2:
        return SVDistribution()
    svd = SVDistribution()
    for s in serial_svd[1:-1].split(";"):
        k, v = s.split("=")
        svd[deserialize_statevector(k)] = float(v)
    return svd


def deserialize_bsdistribution(serial_bsd) -> BSDistribution:
    assert serial_bsd[0] == '{' and serial_bsd[-1] == '}', "Invalid serialized BSDistribution"
    if len(serial_bsd) == 2:
        return BSDistribution()
    bsd = BSDistribution()
    for s in serial_bsd[1:-1].split(";"):
        k, v = s.split("=")
        bsd[deserialize_state(k)] = float(v)
    return bsd


def deserialize_bscount(serial_bsc) -> BSCount:
    assert serial_bsc[0] == '{' and serial_bsc[-1] == '}', "Invalid serialized BSCount"
    if len(serial_bsc) == 2:
        return BSCount()
    bsc = BSCount()
    for s in serial_bsc[1:-1].split(";"):
        k, v = s.split("=")
        bsc[deserialize_state(k)] = int(v)
    return bsc


def deserialize_noise_model(serial_nm: str) -> NoiseModel:
    return NoiseModel(**json.loads(serial_nm))


def deserialize_postselect(serial_ps: str) -> PostSelect:
    return PostSelect(serial_ps)


def deserialize_bs_layered_detector(pb_detect: str | bytes | pb.BSLayeredPPNR) -> BSLayeredPPNR:
    if not isinstance(pb_detect, pb.BSLayeredPPNR):
        pb_binary_repr = pb_detect
        pb_detect = pb.BSLayeredPPNR()
        if isinstance(pb_binary_repr, bytes):
            pb_detect.ParseFromString(pb_binary_repr)
        else:
            pb_detect.ParseFromString(b64decode(pb_binary_repr))
    return _detector_serialization.deserialize_bs_layer(pb_detect)


def deserialize_detector(pb_detect: str | bytes | pb.Detector) -> Detector:
    if not isinstance(pb_detect, pb.Detector):
        pb_binary_repr = pb_detect
        pb_detect = pb.Detector()
        if isinstance(pb_binary_repr, bytes):
            pb_detect.ParseFromString(pb_binary_repr)
        else:
            pb_detect.ParseFromString(b64decode(pb_binary_repr))
    return _detector_serialization.deserialize_detector(pb_detect)


# Known deserializer functions
DESERIALIZER = {
    BS_TAG: BasicState,
    SV_TAG: deserialize_statevector,
    SVD_TAG: deserialize_svdistribution,
    BSD_TAG: deserialize_bsdistribution,
    BSC_TAG: deserialize_bscount,
    BSS_TAG: deserialize_bssamples,
    MATRIX_TAG: deserialize_matrix,
    CIRCUIT_TAG: deserialize_circuit,
    NOISE_TAG: deserialize_noise_model,
    POSTSELECT_TAG: deserialize_postselect,
    DETECTOR_TAG: deserialize_detector,
    BS_LAYERED_DETECTOR_TAG: deserialize_bs_layered_detector,
    COMPONENT_TAG: deserialize_component,
    HERALD_TAG: deserialize_herald,
    PORT_TAG: deserialize_port,
    EXPERIMENT_TAG: deserialize_experiment,
}


def deserialize(obj):
    if isinstance(obj, bytes):
        raise TypeError("Generic deserialize function does not handle binary representation. "
                        "Use specialized functions (e.g. deserialize_circuit) instead.")
    if isinstance(obj, dict):
        r = {}
        for k, v in obj.items():
            r[deserialize(k)] = deserialize(v)
    elif isinstance(obj, list):
        r = []
        for k in obj:
            r.append(deserialize(k))
    elif isinstance(obj, str) and obj.startswith(PCVL_PREFIX):
        if obj.startswith(ZIP_PREFIX):
            # if zip found -> update obj
            # STEPS: remove prefix -> decode b64 encoding -> decompress -> decode utf-8 (byte-> str)
            obj = b64decode(obj[len(ZIP_PREFIX):])
            obj = decompress(obj).decode('utf-8')

        lp = len(PCVL_PREFIX)
        p = obj[lp:].find(SEP)
        class_obj = obj[lp:p+lp]
        serial_obj = obj[p+lp+1:]

        def serializer_not_implemented(_: str):
            raise NotImplementedError(f"Not serializer found for {class_obj}")

        deserialize_func = DESERIALIZER.get(class_obj, serializer_not_implemented)
        r = deserialize_func(serial_obj)
    else:
        r = obj
    return r


def deserialize_file(filepath: str):
    """
    Agnosticly deserialize any supported type from a text file.
    """
    if not path.isfile(filepath):
        raise FileNotFoundError(f'No file at path {filepath}')
    with open(filepath, 'r') as f:
        return deserialize(json.loads(f.read()))


class CircuitBuilder:
    deserialize_fn = {
        'circuit': deserialize_circuit,
        'beam_splitter': _cd.deserialize_bs,
        'phase_shifter': _cd.deserialize_ps,
        'permutation': _cd.deserialize_perm,
        'unitary': _cd.deserialize_unitary,
        'wave_plate': _cd.deserialize_wp,
        'quarter_wave_plate': _cd.deserialize_qwp,
        'half_wave_plate': _cd.deserialize_hwp,
        'time_delay': _cd.deserialize_dt,
        'polarization_rotator': _cd.deserialize_pr,
        'polarized_beam_splitter': _cd.deserialize_pbs,
        'loss_channel': _cd.deserialize_lc,
    }

    def __init__(self, m: int, name: str, params: dict):
        if not name:
            name = None
        self._circuit = Circuit(m=m, name=name)
        self._params = params or dict()

    def add(self, serial_comp):
        component = self.deserialize(serial_comp, self._params)
        self._circuit.add(serial_comp.starting_mode, component, merge=False)

    def retrieve(self):
        return self._circuit

    @staticmethod
    def deserialize(serial_comp, params):
        component = None
        t = serial_comp.WhichOneof('type')
        serial_sub_comp = getattr(serial_comp, t)
        # find the correct deserialization function and use it
        if t in CircuitBuilder.deserialize_fn:
            func = CircuitBuilder.deserialize_fn[t]
            component = func(serial_sub_comp, params)
        elif t == "barrier":
            component = _cd.deserialize_barrier(serial_comp.n_mode,
                                                serial_sub_comp)  # Special case: need an additional info

        if component is None:
            raise NotImplementedError(f'Component could not be deserialized (type = {t})')

        return component


class ExperimentBuilder:
    deserialize_port_fn = {
        "port": deserialize_port,
        "herald": deserialize_herald,
    }

    deserialize_detector_fn = {
        "detector": deserialize_detector,
        "ppnr": deserialize_bs_layered_detector,
    }

    def __init__(self, pb_e: pb.Experiment, params: dict):
        self._pb_e = pb_e
        self._params = params or dict()

    def deserialize_ports(self, experiment, serialized_port_map, location: PortLocation):
        for i, serial_port in sorted(serialized_port_map.items()):  # Sorted needed for the heralds autogenerated names
            t = serial_port.WhichOneof('type')
            serial_sub_comp = getattr(serial_port, t)
            # find the correct deserialization function and use it
            func = self.deserialize_port_fn[t]
            port = func(serial_sub_comp)
            if isinstance(port, Port):
                experiment.add_port(i, port, location=location)
            elif isinstance(port, Herald):
                experiment.add_herald(i, port.expected, port.user_given_name, location)

    def resolve(self):
        name = self._pb_e.name
        m = self._pb_e.n_mode

        if not name:
            name = None
        if m == 0:
            m = None
        experiment = Experiment(m_circuit=m, name=name)

        noise_model_str = self._pb_e.noise_model
        if noise_model_str:
            experiment.noise = deserialize(noise_model_str)

        # Includes the heralded modes --> needed before adding the heralds
        input_state_str = self._pb_e.input_state
        if input_state_str:
            input_state = deserialize(input_state_str)
            if not isinstance(input_state, BasicState) or not input_state.has_polarization:
                experiment.with_input(input_state)
            else:
                experiment.with_polarized_input(input_state)

        min_photons_filter = self._pb_e.min_photons_filter
        if min_photons_filter != VALUE_NOT_SET:
            experiment.min_detected_photons_filter(min_photons_filter)

        # Needed before adding the heralds
        for serial_comp in self._pb_e.components:
            component = CircuitBuilder.deserialize(serial_comp, self._params)
            experiment.add(serial_comp.starting_mode, component)

        # TODO: Detector deserialization needed before component deserialization if there are feed-forward configurators
        #  However, we can't add photonic components after adding a detector (PCVL-799)
        for i, serial_detector in self._pb_e.detectors.items():
            t = serial_detector.WhichOneof('type')
            serial_sub_comp = getattr(serial_detector, t)
            # find the correct deserialization function and use it
            func = self.deserialize_detector_fn[t]
            detector = func(serial_sub_comp)
            experiment.add(i, detector)

        self.deserialize_ports(experiment, self._pb_e.input_ports, PortLocation.INPUT)
        self.deserialize_ports(experiment, self._pb_e.output_ports, PortLocation.OUTPUT)

        # Blocks adding new components --> Needed after adding the components
        post_select_str = self._pb_e.post_select
        if post_select_str:
            experiment.set_postselection(deserialize(post_select_str))

        return experiment
