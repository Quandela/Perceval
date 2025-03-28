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

from multipledispatch import dispatch

from perceval.components import Experiment, Herald, Port, APort, IDetector, BSLayeredPPNR, Detector, AComponent
from perceval.serialization import _schema_circuit_pb2 as pb
from perceval.serialization import serialize
from perceval.serialization._circuit_serialization import serialize_port, serialize_herald, ComponentSerializer
from perceval.serialization._constants import VALUE_NOT_SET
from perceval.serialization._detector_serialization import serialize_detector, serialize_bs_layer


class ExperimentSerializer:

    def __init__(self):
        self._serialized = None

    def serialize(self, experiment: Experiment):
        self._serialized = pb.Experiment()

        if experiment.input_state is not None:
            self._serialized.input_state = serialize.serialize(experiment.input_state)

        self._serialized.name = experiment.name

        if experiment.noise is not None:
            self._serialized.noise_model = serialize.serialize(experiment.noise)

        if experiment.post_select_fn is not None:
            self._serialized.post_select = serialize.serialize(experiment.post_select_fn)

        self._serialized.n_mode = experiment.circuit_size

        if experiment.min_photons_filter:
            self._serialized.min_photons_filter = experiment.min_photons_filter
        else:
            self._serialized.min_photons_filter = VALUE_NOT_SET

        self._serialize_ports(experiment._in_ports, experiment._out_ports)
        self._serialize_detectors(experiment.detectors)
        self._serialize_components(experiment.components)

        return self._serialized

    @staticmethod
    @dispatch(Herald)
    def _serialize_port(port):
        pb_port = pb.APort()
        pb_port.herald.CopyFrom(serialize_herald(port))
        return pb_port

    @staticmethod
    @dispatch(Port)
    def _serialize_port(port):
        pb_port = pb.APort()
        pb_port.port.CopyFrom(serialize_port(port))
        return pb_port

    def _serialize_ports(self, in_ports: dict[APort, list[int]], out_ports: dict[APort, list[int]]):
        for port, modes in in_ports.items():
            pb_port = self._serialize_port(port)
            self._serialized.input_ports[modes[0]].CopyFrom(pb_port)

        for port, modes in out_ports.items():
            pb_port = self._serialize_port(port)
            self._serialized.output_ports[modes[0]].CopyFrom(pb_port)

    @staticmethod
    @dispatch(BSLayeredPPNR)
    def _serialize_detector(detector):
        pb_detector = pb.IDetector()
        pb_detector.ppnr.CopyFrom(serialize_bs_layer(detector))
        return pb_detector

    @staticmethod
    @dispatch(Detector)
    def _serialize_detector(detector):
        pb_detector = pb.IDetector()
        pb_detector.detector.CopyFrom(serialize_detector(detector))
        return pb_detector

    def _serialize_detectors(self, detectors: list[IDetector]):
        for i, detector in enumerate(detectors):
            if detector is not None:
                pb_detector = self._serialize_detector(detector)
                self._serialized.detectors[i].CopyFrom(pb_detector)

    def _serialize_components(self, components: list[tuple[tuple, AComponent]]):
        comp_serializer = ComponentSerializer()
        for r, c in components:
            if not isinstance(c, IDetector):
                self._serialized.components.extend([comp_serializer.serialize(r[0], c)])


def serialize_experiment(experiment: Experiment):
    return ExperimentSerializer().serialize(experiment)
