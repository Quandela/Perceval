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

from ._simulator_utils import _to_bsd, _inject_annotation, _merge_sv, _annot_state_mapping
from .simulator_interface import ISimulator
from perceval.components import ACircuit
from perceval.utils import BasicState, BSDistribution, StateVector, SVDistribution, PostSelect, global_params, \
    DensityMatrix, post_select_distribution, post_select_statevector
from perceval.backends import AProbAmpliBackend
from perceval.utils.density_matrix_utils import extract_upper_triangle

from copy import copy
from multipledispatch import dispatch
from numbers import Number
from typing import Callable, Set, Union, Optional, List
from scipy.sparse import csc_array, csr_array


class Simulator(ISimulator):
    """
    A simulator class relying on a probability amplitude capable backend to simulate the output of a unitary
    non-polarized circuit given an BasicState, StateVector or SVDistribution input. The simulator is able to evolve
    or simulate the sampling a states with annotated photons.

    :param backend: A probability amplitude capable backend object
    """

    def __init__(self, backend: AProbAmpliBackend):
        self._backend = backend
        self._invalidate_cache()
        self._min_detected_photons: int = 0
        self._postselect: PostSelect = PostSelect()
        self._heralds: dict = {}
        self._logical_perf: float = 1
        self._physical_perf: float = 1
        self._rel_precision: float = 1e-6  # Precision relative to the highest probability of interest in probs_svd
        self._keep_heralds = True

    @property
    def precision(self):
        return self._rel_precision

    @precision.setter
    def precision(self, value: float):
        assert isinstance(value, Number) and value >= 0., "Precision must be a positive number"
        self._rel_precision = value

    def set_precision(self, precision: float):
        self.precision = precision

    def set_min_detected_photon_filter(self, value: int):
        """
        Set a minimum number of detected photons in the output distribution

        :param value: The minimum photon count
        """
        self._min_detected_photons = value

    def keep_heralds(self, value: bool):
        """
        Tells the simulator to keep or discard ancillary modes in output states

        :param value: True to keep ancillaries/heralded modes, False to discard them (default is keep).
        """
        self._keep_heralds = value

    def set_selection(self, min_detected_photon_filter: int = None,
                      postselect: PostSelect = None,
                      heralds: dict = None):
        """Set multiple selection filters at once to remove unwanted states from computed output distribution

        :param min_detected_photon_filter: minimum number of detected photons in the output distribution
        :param postselect: a post-selection function
        :param heralds: expected detections (heralds). Only corresponding states will be selected, others are filtered
                        out. Mapping of heralds. For instance `{5: 0, 6: 1}` means 0 photon is expected on mode 5 and 1
                        on mode 6.
        """
        if min_detected_photon_filter is not None:
            self._min_detected_photons = min_detected_photon_filter
        if postselect is not None:
            self._postselect = postselect
        if heralds is not None:
            self._heralds = heralds

    @property
    def logical_perf(self):
        return self._logical_perf

    def set_postselection(self, postselect: PostSelect):
        """Set a post-selection function

        :param postselect: a PostSelect object
        """
        self._postselect = postselect

    def clear_postselection(self):
        """Clear the post-selection function"""
        self._postselect = PostSelect()

    def clear_heralds(self):
        self._heralds = {}

    def set_circuit(self, circuit: ACircuit):
        """Set a circuit for simulation.

        :param circuit: a unitary circuit without polarized components
        """
        self._invalidate_cache()
        self._backend.set_circuit(circuit)

    @dispatch(BasicState, BasicState)
    def prob_amplitude(self, input_state: BasicState, output_state: BasicState) -> complex:
        """Compute the probability amplitude of an output fock state versus an input fock state.

        :param input_state: A fock state with or without photon annotations
        :param output_state: A fock state with or without photon annotations. If the input state holds annotations,
            the output state must hold the same ones, otherwise the computed probability amplitude is 0.

            >>> simulator.set_circuit(Circuit(1))  # One mode identity
            >>> simulator.prob_amplitude(BasicState('|{_:0}>'), BasicState('|{_:1}>'))
            0
            >>> simulator.prob_amplitude(BasicState('|{_:0}>'), BasicState('|{_:0}>'))
            1

        :return: The complex probability amplitude
        """
        if input_state.n == 0:
            return complex(1) if output_state.n == 0 else complex(0)
        input_map = _annot_state_mapping(input_state)
        output_map = _annot_state_mapping(output_state)
        if len(input_map) != len(output_map):
            return complex(0)
        probampli = 1
        for annot, in_s in input_map.items():
            if annot not in output_map:
                return complex(0)
            self._backend.set_input_state(in_s)
            probampli *= self._backend.prob_amplitude(output_map[annot])
        return probampli

    @dispatch(StateVector, BasicState)
    def prob_amplitude(self, input_state: StateVector, output_state: BasicState) -> complex:
        result = complex(0)
        for state, pa in input_state:
            result += self.prob_amplitude(state, output_state) * pa
        return result

    @dispatch(BasicState, BasicState)
    def probability(self, input_state: BasicState, output_state: BasicState) -> float:
        """Compute the probability of an output fock state versus an input fock state, simulating a measure.

        :param input_state: A fock state with or without photon annotations
        :param output_state: A fock state, annotations are ignored
        :return: The probability (float between 0 and 1)
        """
        if input_state.n == 0:
            return 1 if output_state.n == 0 else 0
        input_list = input_state.separate_state(keep_annotations=False)
        result = 0
        for p_output_state in output_state.partition(
                [input_state.n for input_state in input_list]):
            prob = 1
            for i_state, o_state in zip(input_list, p_output_state):
                self._backend.set_input_state(i_state)
                prob *= self._backend.probability(o_state)
            result += prob
        return result

    @dispatch(StateVector, BasicState)
    def probability(self, input_state: StateVector, output_state: BasicState) -> float:
        output_state.clear_annotations()
        sv_out = self.evolve(input_state)  # This is not as optimized as it could be
        result = 0
        for state, pa in sv_out:
            state.clear_annotations()
            if state == output_state:
                result += abs(pa) ** 2
        return result

    def _invalidate_cache(self):
        self._evolve = {}
        self.DEBUG_evolve_count = 0
        self.DEBUG_merge_count = 0

    def _evolve_cache(self, input_list: Set[BasicState]):
        for state in input_list:
            if state not in self._evolve:
                self._backend.set_input_state(state)
                self._evolve[state] = self._backend.evolve()
                self.DEBUG_evolve_count += 1

    def _merge_probability_dist(self, input_list) -> BSDistribution:
        results = BSDistribution()
        for input_state in input_list:
            results = BSDistribution.tensor_product(results, _to_bsd(self._evolve[input_state]), merge_modes=True)
            self.DEBUG_merge_count += 1
        return results

    @dispatch(BasicState)
    def probs(self, input_state: BasicState) -> BSDistribution:
        """
        Compute the probability distribution from a state input
        :param input_state: The input fock state or state vector
        :return: The post-selected output state distribution (BSDistribution)
        """
        input_list = input_state.separate_state(keep_annotations=False)
        self._evolve_cache(set(input_list))
        result = self._merge_probability_dist(input_list)
        result, self._logical_perf = post_select_distribution(
            result, self._postselect, self._heralds, self._keep_heralds)
        return result

    @dispatch(StateVector)
    def probs(self, input_state: StateVector) -> BSDistribution:
        if len(input_state) == 1:
            return self.probs(input_state[0])
        return _to_bsd(self.evolve(input_state))

    def _probs_svd_generic(self, input_dist, p_threshold, progress_callback: Optional[Callable] = None):
        decomposed_input = []
        """decomposed input:
        From a SVD = {
            pa_11*bs_11 + ... + pa_n1*bs_n1: p1,
            pa_12*bs_12 + ... + pa_n2*bs_n2: p2,
            ...
            pa_1k*bs_1k + ... + pa_nk*bs_nk: pk
        }
        the following data structure is built:
        [
            (p1, [
                    (pa_11, {annot_11*: bs_11*,..}),
                    ...
                    (pa_n1, {annot_n1*: bs_n1*,..})
                 ]
            ),
            ...
            (pk, [
                    (pa_1k, {annot_1k*: bs_1k*,..}),
                    ...
                    (pa_nk, {annot_nk*: bs_nk*,..})
                 ]
            )
        ]
        where {annot_xy*: bs_xy*,..} is a mapping between an annotation and a pure basic state"""
        for sv, prob in input_dist.items():
            if max(sv.n) >= self._min_detected_photons:
                decomposed_input.append((prob, [(pa, _annot_state_mapping(st)) for st, pa in sv]))
            else:
                self._physical_perf -= prob
        input_set = set([state for s in decomposed_input for t in s[1] for state in t[1].values()])
        self._evolve_cache(input_set)

        """Reconstruct output probability distribution"""
        res = BSDistribution()
        for idx, (prob0, sv_data) in enumerate(decomposed_input):
            """First, recombine evolved state vectors given a single input"""
            result_sv = StateVector()
            for probampli, instate_list in sv_data:
                prob_sv = abs(probampli)**2
                evolved_in_s = StateVector()
                for annot, in_s in instate_list.items():
                    cached_res = _inject_annotation(self._evolve[in_s], annot)
                    evolved_in_s = _merge_sv(evolved_in_s, cached_res, prob_threshold=p_threshold / (10 * prob_sv * prob0))
                    if len(evolved_in_s) == 0:
                        break
                    self.DEBUG_merge_count += 1
                if evolved_in_s:
                    result_sv += probampli*evolved_in_s

            """Then, add the resulting distribution for a single input to the global distribution"""
            for bs, p in _to_bsd(result_sv).items():
                if bs.n >= self._min_detected_photons:
                    res[bs] += p * prob0
                else:
                    self._physical_perf -= p * prob0

            if progress_callback:
                exec_request = progress_callback((idx + 1) / len(decomposed_input), 'probs')
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    raise RuntimeError("Cancel requested")
        res.normalize()
        return res

    def _probs_svd_fast(self, input_dist, p_threshold, progress_callback: Optional[Callable] = None):
        decomposed_input = []
        """decomposed input:
           From a SVD = {
               bs_1: p1,
               bs_2: p2,
               ...
               bs_k: pk
           }
           the following data structure is built:
           [
               (p1, [bs_1,]),
               ...
               (pk, [bs_k,])
           ]
           where [bs_x,] is the list of the un-annotated separated basic state (result of bs_x.separate_state())
        """
        for sv, prob in input_dist.items():
            if max(sv.n) >= self._min_detected_photons:
                decomposed_input.append(
                    (prob, sv[0].separate_state(keep_annotations=False))
                )
            else:
                self._physical_perf -= prob

        """Create a cache with strong simulation of all unique input"""
        cache = {}
        input_set = set([state for s in decomposed_input for state in s[1]])
        len_input_set = len(input_set)
        for idx, state in enumerate(input_set):
            self._backend.set_input_state(state)
            cache[state] = self._backend.prob_distribution()
            if progress_callback and idx % 10 == 0:
                progress = (idx + 1) / len_input_set * 0.5  # From 0. to 0.5
                exec_request = progress_callback(progress, 'compute probability distributions')
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    raise RuntimeError("Cancel requested")

        """Reconstruct output probability distribution"""
        res = BSDistribution()
        for idx, (prob0, bs_data) in enumerate(decomposed_input):
            """First, recombine evolved state vectors given a single input"""
            probs_in_s = BSDistribution()
            for in_s in bs_data:
                probs_in_s = BSDistribution.tensor_product(probs_in_s, cache[in_s],
                                                           merge_modes=True,
                                                           prob_threshold=p_threshold / (10*prob0))
                if len(probs_in_s) == 0:
                    break
                self.DEBUG_merge_count += 1

            """Then, add the resulting distribution to the global distribution"""
            if probs_in_s:
                for bs, p in probs_in_s.items():
                    if bs.n >= self._min_detected_photons:
                        res[bs] += p * prob0
                    else:
                        self._physical_perf -= p * prob0

            if progress_callback and idx % 20 == 0:
                progress = (idx + 1) / len(decomposed_input) * 0.5 + 0.5  # From 0.5 to 1
                exec_request = progress_callback(progress, 'recombine distributions')
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    raise RuntimeError("Cancel requested")
        res.normalize()
        return res

    def probs_svd(self, input_dist: SVDistribution, progress_callback: Optional[Callable] = None):
        """
        Compute the probability distribution from a SVDistribution input and as well as performance scores

        :param input_dist: A state vector distribution describing the input to simulate
        :param progress_callback: A function with the signature `func(progress: float, message: str)`

        :return: A dictionary of the form { "results": BSDistribution, "physical_perf": float, "logical_perf": float }

            * results is the post-selected output state distribution
            * physical_perf is the performance computed from the detected photon filter
            * logical_perf is the performance computed from the post-selection
        """

        """Trim input SVD given _rel_precision threshold"""
        max_p = 0
        has_superposed_states = False
        for sv, p in input_dist.items():
            if max(sv.n) >= self._min_detected_photons:
                max_p = max(p, max_p)
            if len(sv) > 1:
                has_superposed_states = True
        p_threshold = max(global_params['min_p'], max_p * self._rel_precision)
        svd = SVDistribution({state: pr for state, pr in input_dist.items() if pr > p_threshold})
        if has_superposed_states:
            res = self._probs_svd_generic(svd, p_threshold, progress_callback)
        else:
            res = self._probs_svd_fast(svd, p_threshold, progress_callback)

        res, self._logical_perf = post_select_distribution(res, self._postselect, self._heralds, self._keep_heralds)
        return {'results': res,
                'physical_perf': self._physical_perf,
                'logical_perf': self._logical_perf}

    def probs_density_matrix(self, dm: DensityMatrix) -> dict:
        """
        gives the output probability distribution, after evolving some density matrix through the simulator
        :param dm: the input DensityMatrix
        """
        if not isinstance(dm, DensityMatrix):
            raise TypeError(f"dm must be a DensityMatrix object, {type(dm)} was given")

        input_list = self._get_density_matrix_input_list(dm)
        u_evolve = self._construct_evolve_operator(input_list, dm)

        # Here I change to csr format to be able to iterate on the rows
        u_evolve_in_row = csr_array(u_evolve)
        res_bsd = BSDistribution()

        for row_idx, fs in enumerate(dm.inverse_index):

            vec = u_evolve_in_row[[row_idx]]
            prob = abs((vec @ dm.mat @ vec.conj().T)[0, 0])
            if fs.n >= self._min_detected_photons:
                res_bsd[fs] += prob
            else:
                self._physical_perf -= prob

        res_bsd, logical_perf_coeff = post_select_distribution(
            res_bsd, self._postselect, self._heralds, self._keep_heralds)
        return {'results': res_bsd,
                'physical_perf': self._physical_perf,
                'logical_perf': self._logical_perf * logical_perf_coeff}

    def evolve(self, input_state: Union[BasicState, StateVector]) -> StateVector:
        """
        Evolve a state through the circuit

        :param input_state: The input fock state or state vector
        :return: The output state vector
        """
        if not isinstance(input_state, StateVector):
            input_state = StateVector(input_state)

        # Decay input to a list of basic states without annotations and evolve each of them
        decomposed_input = [(pa, st.separate_state(keep_annotations=True)) for st, pa in input_state]
        input_list = [copy(state) for t in decomposed_input for state in t[1]]
        for state in input_list:
            state.clear_annotations()
        self._evolve_cache(set(input_list))

        result_sv = StateVector()
        for probampli, instate_list in decomposed_input:
            reslist = []
            for in_s in instate_list:
                if in_s.n == 0:
                    reslist.append(in_s)
                    continue
                annotation = in_s.get_photon_annotation(0)
                in_s.clear_annotations()
                reslist.append(_inject_annotation(self._evolve[in_s], annotation))

            # Recombine results for one basic state input
            evolved_in_s = reslist.pop(0)
            for sv in reslist:
                evolved_in_s = _merge_sv(evolved_in_s, sv)
                self.DEBUG_merge_count += 1
            result_sv += evolved_in_s * probampli
        result_sv, _ = post_select_statevector(result_sv, self._postselect, self._heralds, self._keep_heralds)
        return result_sv

    def evolve_svd(self,
                   svd: Union[SVDistribution, StateVector, BasicState],
                   progress_callback: Optional[Callable] = None) -> dict:
        """
        Compute the SVDistribution evolved through a Linear Optical circuit

        :param svd: The input StateVector distribution
        :param progress_callback: A function with the signature `func(progress: float, message: str)`
        :return: A dictionary of the form { "results": SVDistribution, "physical_perf": float, "logical_perf": float }

            * results is the post-selected output SVDistribution
            * physical_perf is the performance computed from the detected photon filter
            * logical_perf is the performance computed from the post-selection
        """
        if not isinstance(svd, SVDistribution):
            return SVDistribution(self.evolve(svd))

        # If it's actually an SVD
        intermediary_logical_perf = 1
        new_svd = SVDistribution()
        for idx, (sv, p) in enumerate(svd.items()):
            if min(sv.n) >= self._min_detected_photons:
                new_sv = self.evolve(sv)
                intermediary_logical_perf -= p*self._logical_perf
                if new_sv.m != 0:
                    new_svd[new_sv] += p
            else:
                self._physical_perf -= p
            if progress_callback:
                exec_request = progress_callback((idx + 1) / len(svd), 'evolve_svd')
                if exec_request is not None and 'cancel_requested' in exec_request and exec_request['cancel_requested']:
                    raise RuntimeError("Cancel requested")
        self._logical_perf = intermediary_logical_perf
        new_svd.normalize()
        return {'results': new_svd,
                'physical_perf': self._physical_perf,
                'logical_perf': self._logical_perf}

    def evolve_density_matrix(self, dm: DensityMatrix) -> DensityMatrix:
        """
        Compute the DensityMatrix evolved from "dm" through a Linear optical circuit

        :param dm: The density Matrix to evolve
        :return: The evolved DensityMatrix
        """
        if not isinstance(dm, DensityMatrix):
            raise TypeError(f"dm must be of DensityMatrix type, {type(dm)} was given")

        # Establishing the set of FockState to evolve
        input_list = self._get_density_matrix_input_list(dm)

        u_evolve = self._construct_evolve_operator(input_list, dm)

        inter_matrix = u_evolve @ extract_upper_triangle(dm.mat) @ u_evolve.T.conj()
        out_matrix = inter_matrix + inter_matrix.T.conjugate(copy=False)

        return DensityMatrix(out_matrix, index=dm.index, check_hermitian=False)

    def _construct_evolve_operator(self, input_list: List[BasicState], dm: DensityMatrix) -> csc_array:
        """
            construct the evolution operator needed to perform evolve_density_matrix.
            Stores it in a csc sparse_matrix
        """

        u_evolve_data = []
        u_evolve_indices = []
        u_evolve_indptr = [0]
        nnz_count = 0
        for i, fs in enumerate(dm.inverse_index):
            if fs in input_list:
                output_sv = self.evolve(fs)
                for state, amplitude in output_sv:
                    u_evolve_data.append(amplitude)
                    u_evolve_indices.append(dm.index[state])
                    nnz_count += 1
            u_evolve_indptr.append(nnz_count)

        # Here we use csc array, because it is constructed row by row
        u_evolve = csc_array((u_evolve_data,
                              u_evolve_indices,
                              u_evolve_indptr),
                             shape=dm.shape)
        return u_evolve

    @staticmethod
    def _get_density_matrix_input_list(dm: DensityMatrix) -> list:
        """
        get the list of Fockstates on which a DensityMatrix is embedded
        """
        input_list = []
        for k in range(dm.size):
            if dm.mat[k, k] != 0:
                input_list.append(dm.inverse_index[k])
        return input_list
