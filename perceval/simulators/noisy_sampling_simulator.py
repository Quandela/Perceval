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

import math
import time
import sys

from perceval.backends import ASamplingBackend
from perceval.components import ACircuit, IDetector, get_detection_type, DetectionType, check_heralds_detectors
from perceval.utils import BasicState, BSDistribution, BSCount, BSSamples, SVDistribution, PostSelect, \
    samples_to_sample_count
from perceval.utils.logging import get_logger, channel, deprecated
from ._simulate_detectors import simulate_detectors_sample


class SamplesProvider:

    def __init__(self, sampling_backend: ASamplingBackend):
        self._backend = sampling_backend
        self._pools = {}
        self._weights = BSCount()
        self._sample_coeff = 1.1
        self._min_samples = 100  # to be sampled at once
        self._max_samples = 2000  # to be sampled at once
        self.sleep_between_batches = 0.2

    def prepare(self, noisy_input: BSDistribution, n_samples: int, progress_callback: callable = None):
        for noisy_s, prob in noisy_input.items():
            ns = min(math.ceil(prob * n_samples), self._max_samples)
            for bs in noisy_s.separate_state(keep_annotations=False):
                self._weights.add(bs, ns)

        get_logger().debug(f"Prepare {len(self._weights)} pools of a total of {self._weights.total()} samples",
                     channel.general)
        for input_state, count in self._weights.items():
            if input_state.n == 0:
                self._pools[input_state] = [input_state]*count
            else:
                self._backend.set_input_state(input_state)
                self._pools[input_state] = self._backend.samples(count)
            self._weights[input_state] = math.ceil(0.1 * self._weights[input_state])

            if progress_callback:
                cancel_request = progress_callback(0, 'prepare')
                time.sleep(self.sleep_between_batches)  # else callback method doesn't have time to be called
                if cancel_request is not None and cancel_request.get('cancel_requested', False):
                    break

    def _compute_samples(self, fock_state: BasicState):
        if fock_state not in self._pools:
            self._pools[fock_state] = []
            self._weights[fock_state] = self._min_samples

        n_samples = self._weights[fock_state]
        get_logger().debug(f"Simulate {n_samples} more {fock_state.n}-photon samples", channel.general)
        self._backend.set_input_state(fock_state)
        self._pools[fock_state] += self._backend.samples(n_samples)
        self._weights[fock_state] = max(int(self._weights[fock_state] * self._sample_coeff), 16)

    def sample_from(self, input_state: BasicState) -> BasicState:
        if input_state not in self._pools or len(self._pools[input_state]) == 0:
            self._compute_samples(input_state)
        return self._pools[input_state].pop()


class NoisySamplingSimulator:
    """
    Simulates a sampling, using a perfect sampling algorithm. It is used to take advantage of a parallel sampling
    algorithm, by computing multiple output states at once, while taking noise and post-processing (heralds,
    post-selection, detector characteristics) into account

    :param sampling_backend: Instance of a sampling-capable back-end
    """

    def __init__(self, sampling_backend: ASamplingBackend):
        self._backend = sampling_backend
        self._min_detected_photons_filter = 0
        self._postselect: PostSelect = PostSelect()
        self._heralds: dict = {}
        self._keep_heralds = True
        self.sleep_between_batches = 0.2  # sleep duration (in s) between two batches of samples
        self._detectors = None

    def set_detectors(self, detector_list: list[IDetector]):
        """
        :param detector_list: A list of detectors to simulate
        """
        self._detectors = detector_list

    def keep_heralds(self, value: bool):
        """
        Tells the simulator to keep or discard ancillary modes in output states

        :param value: True to keep ancillaries/heralded modes, False to discard them (default is keep).
        """
        self._keep_heralds = value

    def set_selection(self,
                      min_detected_photons_filter: int = None,
                      postselect: PostSelect = None,
                      heralds: dict = None,
                      min_detected_photon_filter: int = None):  # TODO: remove for PCVL-786
        """Set multiple selection filters at once to remove unwanted states from computed output distribution

        :param min_detected_photons_filter: minimum number of detected photons in the output distribution
        :param postselect: a post-selection function
        :param heralds: expected detections (heralds). Only corresponding states will be selected, others are filtered
                        out. Mapping of heralds. For instance `{5: 0, 6: 1}` means 0 photon is expected on mode 5 and 1
                        on mode 6.
        """
        if min_detected_photon_filter is not None:  # TODO: remove for PCVL-786
            get_logger().warn(
                'DeprecationWarning: Call with deprecated argument "min_detected_photon_filter", please use "min_detected_photons_filter" instead')
            min_detected_photons_filter = min_detected_photon_filter
        if min_detected_photons_filter is not None:
            self._min_detected_photons_filter = min_detected_photons_filter
        if postselect is not None:
            self._postselect = postselect
        if heralds is not None:
            self._heralds = heralds

    def _state_selected(self, state: BasicState) -> bool:
        """
        Computes if the state is selected given heralds and post selection function
        """
        for m, v in self._heralds.items():
            if state[m] != v:
                return False
        if self._postselect is not None:
            return self._postselect(state)
        return True

    def set_circuit(self, circuit: ACircuit):
        """
        Set the circuit to simulate the sampling on

        :param circuit: A unitary circuit
        """
        self._backend.set_circuit(circuit)

    # TODO: remove for PCVL-786
    @ deprecated(version="0.11.1", reason="Use set_min_detected_photons_filter instead")
    def set_min_detected_photon_filter(self, value: int):
        """
        Set the physical detection filter. Any output state with less than this threshold gets discarded.

        :param value: Minimal photon count in output states of interest.
        """
        self._min_detected_photons_filter = value

    def set_min_detected_photons_filter(self, value: int):
        """
        Set the physical detection filter. Any output state with less than this threshold gets discarded.

        :param value: Minimal photon count in output states of interest.
        """
        self._min_detected_photons_filter = value

    def _perfect_sampling_no_selection(
            self,
            input_state: BasicState,
            n_samples: int,
            progress_callback: callable = None) -> dict:
        self._backend.set_input_state(input_state)
        samples_acquired = 0
        results = BSSamples()
        while samples_acquired < n_samples:
            loop_sample_count = min(1000, n_samples - samples_acquired)

            results += self._backend.samples(loop_sample_count)
            samples_acquired += loop_sample_count

            if progress_callback:
                cancel_request = progress_callback(samples_acquired / n_samples, 'sampling')
                time.sleep(self.sleep_between_batches)  # else callback method doesn't have time to be called
                if cancel_request is not None and cancel_request.get('cancel_requested', False):
                    break

        return {
            "results": results,
            "physical_perf": 1,
            "logical_perf": 1
        }

    def _noisy_sampling(
            self,
            noisy_input: BSDistribution,
            provider: SamplesProvider,
            max_samples: int,
            max_shots: int,
            detection_type: DetectionType,
            progress_callback: callable = None) -> dict:

        output = BSSamples()
        idx = 0
        not_selected_physical = 0
        not_selected = 0
        selected_inputs = []
        shots = 0
        while len(output) < max_samples and (max_shots is None or shots < max_shots):
            if idx == len(selected_inputs):
                idx = 0
                selected_inputs = noisy_input.sample(max_samples, non_null=False)
            selected_bs = selected_inputs[idx]
            idx += 1

            # Sampling
            if selected_bs.has_annotations:  # In case of annotations, input must be separately sampled, then recombined
                bs_list = selected_bs.separate_state(keep_annotations=False)
                sampled_components = []
                for bs in bs_list:
                    sampled_components.append(provider.sample_from(bs))
                sampled_state = sampled_components.pop()
                for component in sampled_components:
                    sampled_state = sampled_state.merge(component)
            else:
                sampled_state = provider.sample_from(selected_bs)

            if self._detectors:
                sampled_state = simulate_detectors_sample(sampled_state, self._detectors, detection_type)

            # Post-processing
            shots += 1
            if sampled_state.n < self._min_detected_photons_filter:
                not_selected_physical += 1
                continue
            if self._state_selected(sampled_state):
                if self._heralds and not self._keep_heralds:  # Remove ancillary modes
                    sampled_state = sampled_state.remove_modes(list(self._heralds.keys()))
                output.append(sampled_state)
            else:
                not_selected += 1

            # Progress handling
            if progress_callback:
                exec_request = progress_callback(len(output) / max_samples, "sampling")
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    break

        # Performance estimate
        selected = len(output)
        logical_perf = 0
        physical_perf = 0
        if selected > 0:
            physical_perf = (selected + not_selected) / (selected + not_selected + not_selected_physical)
            logical_perf = selected / (selected + not_selected)
        return {'results': output, 'physical_perf': physical_perf, 'logical_perf': logical_perf}

    def _check_input_svd(self, svd: SVDistribution) -> tuple[float, float]:
        """
        Check the mixed input state for its validity in the sampling case (no superposed states allowed) and compute
        both Zero Photon Probability (zpp) and MAX Probability of input states containing enough photons (max_p).
        zpp is used to compute samples/shots ratio.
        max_p is used to compute a threshold to ignore non-probable input states.
        """
        if self._detectors:
            assert len(self._detectors) == svd.m,\
                f"State length ({svd.m}) and detector count ({len(self._detectors)}) do not match"
        zpp = 0
        max_p = 0
        for sv, p in svd.items():
            if len(sv) > 1:
                raise ValueError(f"Noisy sampling does not support superposed states, got {sv}")

            n_photons = next(iter(sv.n))  # Number of photons in the (non superposed) state vector
            if n_photons == 0:
                zpp += p
            if n_photons >= self._min_detected_photons_filter:
                max_p = max(max_p, p)
        return zpp, max_p

    def _preprocess_input_state(self, svd: SVDistribution, max_p: float, n_threshold: int
                                ) -> tuple[BSDistribution, float]:
        """
        Rework the input distribution to get rid of improbable states. Compute a first value for physical performance
        """
        p_threshold = max_p / n_threshold
        new_input = BSDistribution()
        physical_perf = 1
        for sv, p in svd.items():
            n_photons = next(iter(sv.n))
            if n_photons < self._min_detected_photons_filter:
                physical_perf -= p
            elif p >= p_threshold:
                new_input[sv[0]] = p
        new_input.normalize()
        get_logger().debug(
            f"Reduced input SVD from {len(svd)} to {len(new_input)} elements using {p_threshold} threshold",
            channel.general)
        return new_input, physical_perf

    def samples(self,
                svd: SVDistribution,
                max_samples: int,
                max_shots: int = None,
                progress_callback: callable = None) -> dict:
        """
        Run a noisy sampling simulation and retrieve the results

        :param svd: The noisy input, expressed as a mixed state
        :param max_samples: Max expected samples of interest in the results
        :param max_shots: Shots limit before the sampling ends (you might get fewer samples than expected)
        :param progress_callback: A progress callback
        :return: A dictionary of the form { "results": BSSamples, "physical_perf": float, "logical_perf": float }
        * results is the post-selected output state distribution
        * physical_perf is the performance computed from the detected photon filter
        * logical_perf is the performance computed from the post-selection
        """
        if not check_heralds_detectors(self._heralds, self._detectors):
            return {"results": BSSamples(), "physical_perf": 1, "logical_perf": 0}

        zpp, max_p = self._check_input_svd(svd)

        # Choose a consistent samples limit
        prepare_samples = max_samples
        if max_shots is not None:
            prepare_samples = min(max_samples, max_shots)
        if prepare_samples == 0:
            return {"results": BSSamples(), "physical_perf": 1, "logical_perf": 1}

        # The best case scenario is a perfect sampling => use the "highway" code
        det_type = get_detection_type(self._detectors)
        if not self._heralds and not self._postselect.has_condition and len(svd) == 1 and det_type == DetectionType.PNR:
            only_input = next(iter(svd))[0]
            if not only_input.has_annotations:
                get_logger().debug("Perfect sampling: use the fast '_perfect_samples_no_selection' call",
                                   channel.general)
                return self._perfect_sampling_no_selection(only_input, prepare_samples, progress_callback)

        new_input, pre_physical_perf = self._preprocess_input_state(svd, max_p, prepare_samples)

        # Prepare pools of pre-computed samples
        provider = SamplesProvider(self._backend)
        provider.sleep_between_batches = self.sleep_between_batches
        provider.prepare(new_input, prepare_samples, progress_callback)

        res = self._noisy_sampling(new_input, provider, max_samples, max_shots, det_type, progress_callback)
        res['physical_perf'] *= pre_physical_perf
        self.log_resources(sys._getframe().f_code.co_name, {
            'n': svd.n_max, 'max_samples': max_samples, 'max_shots': max_shots})
        return res

    def sample_count(self,
                     svd: SVDistribution,
                     max_samples: int,
                     max_shots: int = None,
                     progress_callback: callable = None) -> dict:
        sampling = self.samples(svd, max_samples, max_shots, progress_callback)
        sampling['results'] = samples_to_sample_count(sampling['results'])
        return sampling

    def log_resources(self, method: str, extra_parameters: dict):
        """Log resources of the noisy sampling simulator

        :param method: name of the method used
        :param extra_parameters: extra parameters to log

            Extra parameter can be:

                - max_samples
                - max_shots
        """
        extra_parameters = {key: value for key, value in extra_parameters.items() if value is not None}
        my_dict = {
            'layer': 'NoisySamplingSimulator',
            'backend': self._backend.name,
            'm': self._backend._circuit.m,
            'method': method
        }
        if extra_parameters:
            my_dict.update(extra_parameters)
        get_logger().log_resources(my_dict)
