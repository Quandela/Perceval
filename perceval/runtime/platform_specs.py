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

from typing import Any

from ..utils import FockState, deprecated, ProcessorType
from ..utils.logging import channel, get_logger
from ..components import Experiment, ACircuit, Detector

DEFAULT_MIN_VERSION = "0.10.0"


class PlatformSpecs(dict):
    """
    This class represents the specs of any RemoteProcessor.
    It guarantees that some common fields exist by giving some default values.

    Common fields are accessible through properties, and should be accessed through them.
    If a given common field is not filled by the processor, it will return None or a default value of the correct type.
    If a given RemoteProcessor specs contain a field unknown to this class, it can still be accessed through the dict item syntax

    >>> rp = RemoteProcessor(...)
    >>> rs = rp.specs  # This is a PlatformSpecs object
    >>> pdisplay(rs.architecture)
    >>> print(rs["this_platform_specific_spec"])

    List of included fields (not extensive):
        - architecture
        - available_commands
        - constraints
        - min_client_version
        - pcvl_version
        - software_versions
        - parameters
        - type
    """

    def __getitem__(self, item):
        if hasattr(self, item):
            get_logger().warn(f"Getting {item} from a RemoteProcessor specs should be done using `specs.{item}`"
                              "as it is a common spec key", channel.user)
            return getattr(self, item)
        return self._getitem(item)

    def _getitem(self, item):
        # Avoids the custom __getitem__, and writing super() everywhere
        return super().__getitem__(item)

    @property
    def architecture(self) -> Experiment | None:
        """
        The architecture is an Experiment representing the hardware implementation of the platform, including

            * input state with maximum number of photons for each mode
            * optical components
            * detectors if they are imperfect

        :return: The experiment representing the physical hardware of the RemoteProcessor,
            or None if the RemoteProcessor isn't linked to a hardware chip,
        """
        if "architecture" in self:
            return self._getitem("architecture")
        if "specific_circuit" in self:
            return self._make_architecture()

        return None

    @architecture.setter
    def architecture(self, value: Experiment):
        assert isinstance(value, Experiment)
        self["architecture"] = value

    def _make_architecture(self):
        m = self._getitem("specific_circuit").m  # Avoid deprecation warnings
        e = Experiment(m)
        for r, c in self._getitem("specific_circuit"):
            e.add(r, c)
        if self.get("detector", None) == "threshold":
            threshold = Detector.threshold()
            for i in range(m):
                e.add(i, threshold)
        connected_modes = self.get("connected_input_modes", None)
        if connected_modes:
            input_state = [0] * m
            for i in connected_modes:
                input_state[i] += 1
            e.with_input(FockState(input_state))
        return e

    @property
    @deprecated(reason="Inspect the circuit from the architecture", version="1.2.0")
    def specific_circuit(self) -> ACircuit | None:
        """
        :return: the hardware implemented circuit of the platform, or None it isn't linked to a hardware chip,
         or not representable as a unitary Circuit
        """
        if "specific_circuit" in self:
            return self._getitem("specific_circuit")
        if "architecture" in self:
            try:  # Might fail if non-unitary
                return self._getitem("architecture").unitary_circuit()
            except RuntimeError:
                return None

        return None

    @specific_circuit.setter
    @deprecated(reason="Set the components in the architecture", version="1.2.0")
    def specific_circuit(self, value: ACircuit):
        assert isinstance(value, ACircuit)
        self["specific_circuit"] = value

    @property
    @deprecated(reason="Inspect the input state from the architecture", version="1.2.0")
    def connected_input_modes(self) -> list[int]:
        """
        :return: A list containing the mode numbers where a photon can be sent to the chip on the platform hardware.
            The list may be empty if there is no hardware
        """
        return self.get("connected_input_modes", [])

    @connected_input_modes.setter
    @deprecated(reason="Set the input state in the architecture", version="1.2.0")
    def connected_input_modes(self, value: list[int]):
        assert isinstance(value, list)
        assert all(isinstance(val, int) for val in value)
        self["connected_input_modes"] = value

    @property
    @deprecated(reason="Inspect the detectors or detection type from the architecture", version="1.2.0")
    def detector(self) -> str | None:
        """
        :return: The type of the detection, as a str, or None if there is no hardware
        """
        if "detector" in self:
            return self._getitem("detector")
        arch = self.architecture
        if arch is not None:
            return arch.detection_type.name
        return None

    @detector.setter
    @deprecated(reason="Set the detectors in the architecture", version="1.2.0")
    def detector(self, value: str):
        assert isinstance(value, str)
        self["detector"] = value

    @property
    def available_commands(self) -> list[str]:
        """
        :return: the list of command names available for this platform
        """
        return self.get("available_commands", [])

    @available_commands.setter
    def available_commands(self, value: list[str]):
        assert isinstance(value, list)
        assert all(isinstance(val, str) for val in value)
        self["available_commands"] = value

    @property
    def constraints(self) -> dict[str, Any]:
        """
        :return: A dictionary detailing the constraints of the platform (min/max number of modes/photons, ...).
            May be empty if there are no constraints
        """
        return self.get("constraints", {})

    @constraints.setter
    def constraints(self, value: dict[str, Any]):
        assert isinstance(value, dict)
        assert all(isinstance(val, str) for val in value)
        self["constraints"] = value

    @property
    def min_client_version(self) -> str:
        """
        :return: The minimum version of perceval that the user must have for the call to this platform to resolve correctly
        """
        return self.get("min_client_version", DEFAULT_MIN_VERSION)

    @min_client_version.setter
    def min_client_version(self, value: str):
        assert isinstance(value, str)  # Should we use version.Version ?
        self["min_client_version"] = value

    @property
    def pcvl_version(self) -> str:
        """
        :return: The version of perceval on the platform
        """
        if "pcvl_version" in self:
            return self._getitem("pcvl_version")
        if "software_versions" in self:
            return self.software_versions.get("perceval-quandela", DEFAULT_MIN_VERSION)
        return DEFAULT_MIN_VERSION

    @pcvl_version.setter
    def pcvl_version(self, value: str):
        assert isinstance(value, str)
        self["pcvl_version"] = value

    @property
    def description(self) -> str:
        """
        :return: A short description of the platform
        """
        return self.get("description", "")

    @description.setter
    def description(self, value: str):
        assert isinstance(value, str)
        self["description"] = value

    @property
    def parameters(self) -> dict[str, str]:
        """
        :return: A dictionary containing the possible parameters of the platform.
            * The key must be given to the platform using the :code:`set_parameters()` method of the RemoteProcessor.
            * The value is a description of what the parameter does.
        """
        return self.get("parameters", {})

    @parameters.setter
    def parameters(self, value: dict[str, str]):
        assert isinstance(value, dict)
        assert all(isinstance(val, str) for val in value)
        assert all(isinstance(val, str) for val in value.values())
        self["parameters"] = value

    @property
    def pre_shots_compatibility(self) -> bool:
        # Internal use - do not comment this
        return self.get("pre_shots_compatibility", False)

    @pre_shots_compatibility.setter
    def pre_shots_compatibility(self, value: bool):
        assert isinstance(value, bool)
        self["pre_shots_compatibility"] = value

    @property
    def software_versions(self) -> dict[str, str]:
        """
        :return: The main software versions of the platform (ex: perceval, exqalibur...)
        """
        if "software_versions" in self:
            return self._getitem("software_versions")
        return {'perceval-quandela': self.pcvl_version}

    @software_versions.setter
    def software_versions(self, value: dict[str, str]):
        assert isinstance(value, dict)
        assert all(isinstance(val, str) for val in value)
        assert all(isinstance(val, str) for val in value.values())
        self["software_versions"] = value

    @property
    def type(self) -> ProcessorType:
        if "type" in self:
            self_type = self._getitem("type")
            if isinstance(self_type, ProcessorType):
                return self_type
            return ProcessorType.SIMULATOR if self_type == "simulator" else ProcessorType.PHYSICAL
        return ProcessorType.SIMULATOR

    @type.setter
    def type(self, value: ProcessorType):
        assert isinstance(value, ProcessorType)
        # Store as a str so we can serialize it easily
        self["type"] = "simulator" if value == ProcessorType.SIMULATOR else "qpu"
