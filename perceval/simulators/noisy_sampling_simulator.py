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
from typing import Callable, Dict

from perceval.backends import ASamplingBackend
from perceval.components import ACircuit
from perceval.utils import BasicState, BSCount, BSSamples, SVDistribution


class SamplesProvider:

    def __init__(self, sampling_backend: ASamplingBackend):
        self._backend = sampling_backend
        self._pools = {}
        self._weights = BSCount()
        self._sample_coeff = 1.1
        self._min_samples = 100

    def prepare(self, noisy_input: SVDistribution, n_samples: int):
        # print("SamplesProvider - prepare")
        for sv, prob in noisy_input.items():
            # ns = int(max(prob * self._sample_coeff * n_samples, self._min_samples))
            ns = int(math.floor(prob * n_samples))
            for bs in sv[0].separate_state(keep_annotations=False):
                # print(f" ** {bs} component : add {ns}")
                self._weights.add(bs, ns)

        for input_state, count in self._weights.items():
            # print(f"                - {count} samples from {input_state}")
            if input_state.n == 0:
                self._pools[input_state] = [input_state]*count
            else:
                self._backend.set_input_state(input_state)
                self._pools[input_state] = self._backend.samples(count)
            self._weights[input_state] = math.ceil(0.1 * self._weights[input_state])
        # print("WEIGHTS AFTER PREPARE")
        # print(self._weights)

    def _compute_samples(self, fock_state: BasicState):
        if fock_state not in self._pools:
            self._pools[fock_state] = []
            self._weights[fock_state] = self._min_samples

        # print(f"SamplesProvider - compute {self._weights[fock_state]} additional samples for {fock_state}")
        self._backend.set_input_state(fock_state)
        self._pools[fock_state] += self._backend.samples(self._weights[fock_state])
        self._weights[fock_state] = int(self._weights[fock_state] * self._sample_coeff)

    def sample_from(self, input_state: BasicState) -> BasicState:
        # print(f"SamplesProvider - pop 1 sample from {input_state}")
        if input_state not in self._pools or len(self._pools[input_state]) == 0:
            self._compute_samples(input_state)
        return self._pools[input_state].pop()


class NoisySamplingSimulator:

    def __init__(self, sampling_backend: ASamplingBackend):
        self._backend = sampling_backend
        self._min_detected_photon_filter = 0

    def set_circuit(self, circuit: ACircuit):
        self._backend.set_circuit(circuit)

    def set_min_detected_photon_filter(self, value: int):
        self._min_detected_photon_filter = value

    def samples(self,
                svd: SVDistribution,
                max_samples: int,
                max_shots: int = None,
                progress_callback: Callable = None) -> Dict:
        new_svd = SVDistribution()
        physical_perf = 1
        zpp = 0
        for sv, p in svd.items():
            if len(sv) > 1:
                raise ValueError(f"Noisy sampling does not support superposed states, got {sv}")

            n_photons = next(iter(sv.n))
            if n_photons == 0:
                zpp += p
            if n_photons < self._min_detected_photon_filter or max_samples * p < 10:
                # print(f"remove {sv}")
                physical_perf -= p
            else:
                new_svd[sv] = p

        if max_shots is not None:
            max_shots = round(max_shots*(1 - zpp))

        provider = SamplesProvider(self._backend)
        provider.prepare(new_svd, max_samples)

        output = BSSamples()
        idx = 0
        not_selected_physical = 0
        not_selected = 0
        selected_inputs = []
        shots = 0
        while len(output) < max_samples and (max_shots is None or shots < max_shots):
            if idx == len(selected_inputs):
                idx = 0
                selected_inputs = new_svd.sample(max_samples, non_null=False)
            selected_bs = selected_inputs[idx][0]
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
            output.append(sampled_state) ## TODO remove this and fix post-processing below
            # Post-processing
            shots += 1
            # if not self._state_selected_physical(sampled_state):
            #     not_selected_physical += 1
            #     continue
            # if self._state_selected(sampled_state):
            #     output.append(self.postprocess_output(sampled_state))
            # else:
            #     not_selected += 1

            # Progress handling
            if progress_callback:
                exec_request = progress_callback(len(output)/max_samples, "sampling")
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    break
        for in_state, out_list in provider._pools.items():
            if len(out_list):
                print(f"for {in_state}, {len(out_list)} samples remaining")
        return {"results": output, "physical_perf": physical_perf}
