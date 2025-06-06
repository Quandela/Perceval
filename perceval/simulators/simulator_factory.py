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

from .feed_forward_simulator import FFSimulator
from .simulator_interface import ISimulator
from .simulator import Simulator
from .delay_simulator import DelaySimulator
from .loss_simulator import LossSimulator
from .polarization_simulator import PolarizationSimulator
from ._simulator_utils import _unitary_components_to_circuit
from perceval.components import ACircuit, TD, LC, Processor, Experiment, AFFConfigurator
from perceval.backends import ABackend, SLOSBackend, BACKEND_LIST


class SimulatorFactory:
    """
    Using the SimulatorFactory is an easy and integrated way of instantiating the correct layers of simulation for a
    given circuit. The factory will adapt to the component needs, in terms of simulation, and chain the correct
    simulator calls.
    """

    @staticmethod
    def build(circuit: ACircuit | Processor | list,
              backend: ABackend | str = None,
              **kwargs) -> ISimulator:
        """
        :param circuit: The optical circuit to build the simulation layers around.
            The circuit can be a unitary circuit (Circuit object), a list containing positioned unitary components + LC
            + TD, or a Processor object.
        :param backend: (Optional) Any probampli capable backend instance or name. If no backend is passed, then the
            processor backend name is used if the first parameter's type is Processor. Ultimately, the fallback is a
            SLOS backend instantiated without any configuration (i.e. no mask)
        :return: A simulator object with the input circuit set
        """
        sim_polarization = False
        sim_delay = False
        sim_losses = False
        sim_feed_forward = False
        convert_to_circuit = False
        min_detected_photons = None
        post_select = None
        heralds = None
        source = None
        noise = None
        m = None

        if isinstance(circuit, Processor):
            if backend is None:
                # If no backend was chosen, the backend type set in the Processor is used
                backend = circuit.backend
            source = circuit.source
            circuit = circuit.experiment

        if not isinstance(circuit, ACircuit):
            convert_to_circuit = True
            if isinstance(circuit, Experiment):
                m = circuit.circuit_size
                min_detected_photons = circuit.min_photons_filter
                post_select = circuit.post_select_fn
                heralds = circuit.heralds
                noise = circuit.noise
                if circuit.is_unitary:
                    circuit = circuit.unitary_circuit(use_phase_noise=True)
                else:
                    circuit = circuit.components

            if not isinstance(circuit, ACircuit):
                for _, cp in circuit:
                    if not sim_losses and isinstance(cp, LC):
                        sim_losses = True
                        convert_to_circuit = False
                    if not sim_delay and isinstance(cp, TD):
                        sim_delay = True
                        convert_to_circuit = False
                    if not sim_polarization and isinstance(cp, ACircuit):
                        sim_polarization = cp.requires_polarization
                    if not sim_feed_forward and isinstance(cp, AFFConfigurator):
                        sim_feed_forward = True
                        convert_to_circuit = False

        if isinstance(circuit, ACircuit):
            sim_polarization = circuit.requires_polarization


        if backend is None:
            backend = SLOSBackend()  # The default is SLOS
        if isinstance(backend, str):
            if backend in BACKEND_LIST:
                backend = BACKEND_LIST[backend](**kwargs)  # Create an instance of the backend
            else:
                raise ValueError(f"Backend '{backend}' not supported")

        # Building the simulator layers
        if sim_feed_forward:
            assert not sim_delay, "Cannot simulate time delays in a feed-forward simulation"
            assert not sim_polarization, "Cannot simulate polarization in a feed-forward simulation"
            simulator = FFSimulator(backend)
            simulator.set_noise(noise)
            simulator.set_source(source)

        else:
            simulator = Simulator(backend)
            if sim_polarization:
                simulator = PolarizationSimulator(simulator)
            if sim_delay:
                simulator = DelaySimulator(simulator)
            if sim_losses:
                simulator = LossSimulator(simulator)

        simulator.set_selection(min_detected_photons_filter=min_detected_photons,
                                postselect=post_select, heralds=heralds)

        if convert_to_circuit:
            circuit = _unitary_components_to_circuit(circuit, m)
        # m can only be retrieved from a Processor (and useful only in that case)
        simulator.set_circuit(circuit, m)
        return simulator
