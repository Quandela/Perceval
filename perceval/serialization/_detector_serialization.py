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
from perceval.serialization import _schema_circuit_pb2 as pb
from perceval.components import BSLayeredPPNR, Detector


def serialize_bs_layer(detector: BSLayeredPPNR):
    pb_d = pb.BSLayeredPPNR()
    pb_d.name = detector.name
    pb_d.bs_layers = detector._layers
    pb_d.reflectivity = detector._r
    return pb_d


def deserialize_bs_layer(pb_d: pb.BSLayeredPPNR) -> BSLayeredPPNR:
    detector = BSLayeredPPNR(pb_d.bs_layers, pb_d.reflectivity)
    detector.name = pb_d.name
    return detector


def serialize_detector(detector: Detector):
    pb_d = pb.Detector()
    pb_d.name = detector.name
    if detector._wires is not None:
        pb_d.n_wires = detector._wires
    if detector.max_detections is not None:
        pb_d.max_detections = detector.max_detections
    return pb_d


def deserialize_detector(pb_d: pb.Detector) -> Detector:
    n_wires = pb_d.n_wires or None
    max_detections = pb_d.max_detections or None
    detector = Detector(n_wires, max_detections)
    detector.name = pb_d.name
    return detector
