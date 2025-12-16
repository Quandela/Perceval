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
import copy
from typing import Any

from perceval.components import AComponent, Barrier, PERM, IDetector, Herald, PortLocation, Source, Experiment
from perceval.utils import (NoiseModel, BasicState, FockState, BSDistribution, SVDistribution, StateVector,
                            partial_progress_callable, get_logger, deprecated)
from perceval.utils.logging import channel
from perceval.components.feed_forward_configurator import AFFConfigurator
from perceval.backends import AStrongSimulationBackend, IFFBackend
from perceval.runtime import Processor

from .simulator_interface import ISimulator


class FFSimulator(ISimulator):

    def __init__(self, backend: AStrongSimulationBackend):
        super().__init__()
        self._precision = None

        self._components = None
        self._backend = backend

        self._noise_model = None

    def compute_physical_logical_perf(self, value: bool):
        # TODO: decide what happens to this
        if value:
            get_logger().warn("Only the global performance can be computed for a feed-forward simulator.")

    def set_circuit(self, circuit: Processor | Experiment | list[tuple[tuple, AComponent]], m=None):
        if isinstance(circuit, (Processor, Experiment)):
            self._components = circuit.components
            min_detected_photons = circuit._min_detected_photons_filter
            post_select = circuit.post_select_fn
            heralds = circuit.heralds
            self.set_selection(min_detected_photons, post_select, heralds)
        else:
            self._components = circuit

    def set_noise(self, nm: NoiseModel):
        self._noise_model = nm

    @deprecated("Version 1.1 - Source is no longer used")
    def set_source(self, source: Source):
        pass

    def _probs_svd(self,
                   input_state: SVDistribution | tuple[Source, BasicState],
                   detectors: list[IDetector] = None,
                   progress_callback: callable = None) -> tuple[BSDistribution, float]:

        # 1: Find all the FFConfigurators that can be simulated without measuring more modes
        considered_config, measured_modes, unsafe_modes = self._find_next_simulation_layer()

        # 2: Launch a simulation with the default circuits
        components = self._components.copy()
        default_circuits = []
        for i, config in considered_config:
            r = self._components[i][0]
            circ_r = config.config_modes(r)
            default_circuits.append(config.default_circuit)

            # Use only the first mode as circuits can have different sizes
            components[i] = (circ_r[0], config.default_circuit)

        # We can't reject any state at this moment since we need all possible measured states
        # Except for heralds on safe modes (i.e. not subject to feed-forward anywhere)
        new_heralds = {r: v for r, v in self._heralds.items() if r not in unsafe_modes} if self._heralds is not None else None
        # TODO: in theory, if we can split the Postselect keeping only the safe modes,
        #  it can be even faster by removing more impossible measures (thus not simulating them)

        # Estimation of possible measures: n for each measured mode
        if isinstance(input_state, tuple):
            n = input_state[1].n
            m = input_state[1].m
            if self._noise_model is not None and self._noise_model.g2 > 0:
                n *= 2
        else:
            n = input_state.n_max
            m = input_state.m
        intermediate_progress = 1 / n ** len(measured_modes)
        prog_cb = partial_progress_callable(progress_callback, max_val=intermediate_progress)

        default_res = self._simulate(input_state, components, m, detectors, prog_cb, new_heralds=new_heralds)
        default_norm_factor = default_res["global_perf"]
        default_res = default_res["results"]

        # 3: deduce all measurable states and launch one simulation for each of them
        res = BSDistribution()
        simulated_measures = set()

        global_perf = 0  # = P(logic_filter and n >= photon_filter)
        prog_cb = partial_progress_callable(progress_callback, min_val=intermediate_progress)

        for j, (state, prob) in enumerate(default_res.items()):
            sub_circuits = [config.configure(state[slice(self._components[i][0][0], self._components[i][0][-1] + 1)])
                            for i, config in considered_config]

            if all(c1 == default_circuit for c1, default_circuit in zip(sub_circuits, default_circuits)):
                # Default case, no need for further computation

                # There is a problem here: if the same circuit is instanced twice, the == operator returns False
                # This can be a problem if we copy the default circuit or instantiate a new one

                if self._post_process_state(state):
                    prob *= default_norm_factor
                    res[self._remove_heralds(state[:m])] = prob
                    global_perf += prob

                if prog_cb is not None:
                    prog_cb((j + 1) / len(default_res), "Computing feed-forwarded circuit")
                continue

            measured_state = tuple(state[i] for i in measured_modes)

            if measured_state not in simulated_measures:

                for sub_i, (i, config) in enumerate(considered_config):
                    r = self._components[i][0]
                    # Does not use the list of modes as circuits can have different sizes
                    components[i] = (config.config_modes(r)[0], sub_circuits[sub_i])

                new_heralds = {i: state[i] for i in measured_modes if i not in self._heralds}
                new_prog_cb = partial_progress_callable(prog_cb, j / len(default_res), (j + 1) / len(default_res))

                sub_res = self._simulate(input_state, components, m, detectors, new_prog_cb,
                                         filter_states=True, new_heralds=new_heralds)

                norm_factor = sub_res["global_perf"]

                # The remaining states are only the ones with n >= filter and mask
                global_perf += norm_factor

                for st, p in sub_res["results"].items():
                    # No need for post_process here: results are already post-processed by the sub simulator
                    res[self._remove_heralds(st[:m])] = p * norm_factor

                simulated_measures.add(measured_state)

        if len(res):
            res.normalize()
        return res, global_perf

    def _find_next_simulation_layer(self) -> tuple[list[tuple[int, AFFConfigurator]], list[int], set[int]]:
        """
        :return: The list containing the tuples with the index in the component list
         of the configuration independent FFConfigurators and their instances,
         the list of the associated measured modes,
         and the list of modes that are touched at anytime by feed-forward configurators (including after the layer)
        """
        # We can add a configurator as long as the measured mode don't come from a configurable circuit
        feed_forwarded_modes: set[int] = set()
        measured_modes = set()
        res = []
        lock_res = False

        for i, (r, c) in enumerate(self._components):
            if isinstance(c, AFFConfigurator):
                if not lock_res and any(r0 in feed_forwarded_modes for r0 in r):
                    lock_res = True

                feed_forwarded_modes.update(c.config_modes(r))
                if not lock_res:
                    res.append((i, c))
                    measured_modes.update(r)

            elif isinstance(c, Barrier):
                continue

            elif isinstance(c, PERM):
                to_remove = []
                to_add = []
                perm_vector = c.perm_vector
                for new_mode in r:
                    if new_mode in feed_forwarded_modes:
                        to_remove.append(new_mode)
                        to_add.append(perm_vector[new_mode - r[0]] + r[0])
                feed_forwarded_modes.difference_update(to_remove)
                feed_forwarded_modes.update(to_add)

            elif any(new_mode in feed_forwarded_modes for new_mode in r):
                feed_forwarded_modes.update(r)

        return res, list(measured_modes), feed_forwarded_modes

    def _get_sim_params(self,
                       input_state: SVDistribution | tuple[Source, BasicState],
                       components: list[tuple[tuple, AComponent | Processor | Experiment]],
                       m: int,
                       detectors: list[IDetector] = None,
                       filter_states: bool = False,
                       new_heralds: dict[int, int] = None):
        """Initialize a new simulator with the given components and heralds.
        Heralds that are already in this simulator are still considered.

        :param input_state: The input state used for the simulation
        :param components: A list of components that will be added in the simulation. Can themselves be processors.
        :param filter_states: Whether the states should be filtered in the sub-simulation.
        :param new_heralds: The list of heralds that should be added, containing the position and the value

        :return: A configured simulator to run, the input state to give it, and the detectors to use
        """

        if detectors is None:
            detectors = m * [None]
        proc = Processor(self._backend, m)
        if self._noise_model is not None:
            proc.noise = self._noise_model

        for r, c in components:
            proc.add(r, c)

        # Now the Processor has only the heralds that were possibly added by adding Processors as input, all at the end
        if isinstance(input_state, SVDistribution):
            heralded_dist = proc.generate_noisy_heralds()
            if len(heralded_dist):
                input_state = input_state * heralded_dist  # Must not change the original object
        else:
            proc.with_input(input_state[1])
            input_state = (input_state[0], proc.input_state)

        sum_new_heralds = 0
        if new_heralds is not None:
            for r, v in new_heralds.items():
                proc.add_port(r, Herald(v), PortLocation.OUTPUT)
                sum_new_heralds += v

        if filter_states:

            if self._heralds is not None:
                for r, v in self._heralds.items():
                    proc.add_port(r, Herald(v), PortLocation.OUTPUT)

            if self._postselect.has_condition:
                if proc.post_select_fn is not None:
                    postselect = copy.copy(self._postselect)
                    postselect.merge(proc.post_select_fn)
                else:
                    postselect = self._postselect
                proc.set_postselection(postselect)

            # We need to retrieve the new heralds as they are actually counting user photons
            proc.min_detected_photons_filter(self._min_detected_photons_filter - sum_new_heralds)

        else:
            # In that case, the new heralds are user-defined heralds that we can safely simulate.
            # We can't filter more due to potential losses that would depend on the FF parts of the circuit
            proc.min_detected_photons_filter(0)

        from .simulator_factory import SimulatorFactory  # Avoids a circular import

        sim = SimulatorFactory.build(proc)
        if self._precision is not None:
            sim.set_precision(self._precision)
        sim.set_silent(True)
        return sim, input_state, detectors + proc.detectors[m:]

    def _simulate(self, input_state: SVDistribution | tuple[Source, BasicState],
                  components: list[tuple[tuple, AComponent | Processor]],
                  m: int,
                  detectors: list[IDetector],
                  prog_cb=None,
                  filter_states: bool = False,
                  new_heralds: dict[int, int] = None) \
            -> dict[str, Any]:
        sim, input_state, detectors = self._get_sim_params(input_state, components, m, detectors, filter_states, new_heralds)
        return sim.probs_svd(input_state, detectors, prog_cb)

    def _post_process_state(self, bs: BasicState) -> bool:
        """Returns True if the state checks all requirements of the simulator"""
        if bs.n < self.min_detected_photons_filter:
            return False

        for m, v in self._heralds.items():
            if bs[m] != v:
                return False
        return self._postselect(bs)

    def _remove_heralds(self, state: BasicState) -> BasicState:
        if not self._keep_heralds:
            return state.remove_modes(list(self._heralds.keys()))
        return state

    def probs(self, input_state: BasicState) -> BSDistribution:
        """
        Compute the probability distribution from a BasicState input

        :param input_state: A basic state describing the input to simulate

        :return: A BSDistribution
        """
        if isinstance(self._backend, IFFBackend) and self._backend.can_simulate_feed_forward(self._components, input_state):
            m = input_state.m
            get_logger().info("Perform a direct feed-forward simulation", channel.general)
            sim = self._get_sim_params(SVDistribution(input_state), [], m, filter_states = True)[0]
            sim.keep_heralds(self._keep_heralds)
            self._backend.set_feed_forward(self._components, m)
            return sim.probs(input_state)

        return self._probs_svd(SVDistribution(input_state))[0]

    def probs_svd(self,
                  input_dist: SVDistribution,
                  detectors: list[IDetector] = None,
                  progress_callback: callable = None):
        """
        Compute the probability distribution from a SVDistribution input and as well as performance scores

        :param input_dist: A state vector distribution describing the input to simulate
        :param detectors: An optional list of detectors
        :param progress_callback: A function with the signature `func(progress: float, message: str)`

        :return: A dictionary of the form { "results": BSDistribution, "global_perf": float }

            * results is the post-selected output state distribution
            * global_perf is the probability that a state is post-selected
        """
        if isinstance(self._backend, IFFBackend) and self._backend.can_simulate_feed_forward(self._components, input_dist, detectors):
            if isinstance(input_dist, tuple):
                m = input_dist[1].m
            else:
                m = input_dist.m
            get_logger().info("Perform a direct feed-forward simulation", channel.general)
            sim, input_dist, detectors  = self._get_sim_params(input_dist, [], m, detectors, filter_states=True)
            sim.compute_physical_logical_perf(self._compute_physical_logical_perf)
            sim.keep_heralds(self._keep_heralds)
            self._backend.set_feed_forward(self._components, m)
            return sim.probs_svd(input_dist, detectors, progress_callback)

        res = self._probs_svd(input_dist, detectors, progress_callback)
        return {'results': res[0],
                'global_perf': res[1]}

    def evolve(self, input_state: FockState | StateVector) -> StateVector:
        if not isinstance(self._backend, IFFBackend) or not self._backend.can_simulate_feed_forward(self._components, input_state):
            raise RuntimeError("Cannot perform state evolution with feed-forward")

        m = input_state.m
        sim = self._get_sim_params(SVDistribution(input_state), [], m, filter_states = True)[0]
        sim.keep_heralds(self._keep_heralds)
        self._backend.set_feed_forward(self._components, m)
        return sim.evolve(input_state)

    def set_precision(self, precision: float):
        self._precision = precision
