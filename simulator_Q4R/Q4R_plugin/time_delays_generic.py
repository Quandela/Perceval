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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from collections import defaultdict

import perceval.components.base_components as comp
from perceval import Circuit, BasicState, StateVector, ACircuit, Backend, SVDistribution
from perceval.utils.algorithms.simplification import simplify

from typing import Union, Type, List

import quandelibc as qc


def expand_TD(circuit, depth, pre_computing=False):
    r"""
    :param circuit: A circuit with time delay
    :param depth: The depth to which the circuit has to be expanded
    :param pre_computing: If set to True, intermediary matrices will be computed. Useful for Stepper (?)
    """
    m = circuit.m
    U = []

    cur_U = []
    for r, c in circuit:
        if not isinstance(c, comp.TD):
            cur_U.append([r, c])

        else:
            t = c.get_variables()["t"]
            assert isinstance(t, int), "Time delays do not support non integer delay for now"

            r0 = r[0]

            if r0 != m - 1:
                # Add a fake permutation to put the TD at the last mode
                r = list(range(r0, m))
                perm_list = [m - r0 - 1] + list(range(1, m - r0 - 1)) + [0]
                cur_U.append([r, comp.PERM(perm_list)])

            for _ in range(t):
                U.append(cur_U.copy())
                cur_U = []

            if r0 != m - 1:
                # Nullify the fake permutation
                cur_U.append([r, comp.PERM(perm_list)])

    U.append(cur_U.copy())

    new_m = depth * m + len(U) - 1
    new_circ = Circuit(new_m)

    if pre_computing:
        for i, cur_U in enumerate(U):
            # First, we add all non TD components
            U_circ = Circuit(m)
            for r, c in cur_U:
                U_circ = U_circ // (r[0], c)

            U_circ = comp.Unitary(U_circ.compute_unitary(use_symbolic=False))

            U[i] = [[list(range(U_circ.m)), U_circ]]

    for d in range(depth):
        for i, cur_U in enumerate(U):
            # First, we add all non TD components
            for r, c in cur_U:
                new_circ = new_circ // (r[0] + d * m, c)

            # Then we permute to mimic TD
            if i != len(U) - 1:
                r0 = (d + 1) * m - 1
                perm_list = [new_m - i - r0 - 1] + list(range(1, new_m - i - r0 - 1)) + [0]
                new_circ = new_circ // (r0, comp.PERM(perm_list))

    return new_circ


def create_input(state, depth, TD_number: Union[int, ACircuit]):
    if isinstance(TD_number, ACircuit):
        TD_number = count_TD(TD_number)
    out = state ** depth
    out *= BasicState([0] * TD_number)
    return out


def count_TD(circuit):
    return sum([c.get_variables()["t"] if isinstance(c, comp.TD) else 0 for _, c in circuit])


def TD_allstateprob_iterator(simulator, input_state: Union[BasicState, StateVector], circuit):
    # Give the prob output for the steady regime
    ns = input_state.n
    if not isinstance(ns, list):
        ns = [ns]
    m = circuit.m
    TD_number = count_TD(circuit)
    depth = TD_number + 1
    real_input = create_input(input_state, depth, TD_number)
    real_circuit = simplify(expand_TD(circuit, depth, True))
    interest_m = depth * m

    sim = simulator(real_circuit)
    real_out_sv = sim.evolve(real_input)

    for k in range(max(ns) * (1 + TD_number) + 1):
        for out_state in qc.FSArray(m, k):
            # Simply get all possible output states
            out_prob = 0
            out_state = BasicState(out_state)

            for real_out in real_out_sv:
                if list(real_out[(depth - 1) * m: interest_m]) == list(out_state):
                    real_out = BasicState(real_out)
                    out_prob += abs(real_out_sv[real_out]) ** 2

            yield out_state, out_prob


def update_cond_prob(res, subdict, state, keep_all):
    # First, normalise subdict
    s = sum(subdict.values())
    if s == 0:
        # It is impossible to reach the prior state
        return

    factor = 1.0 / s

    new_subdict = {key: val * factor for key, val in subdict.items() if
                   val != 0}  # Normalize and remove impossible cases

    if not keep_all:
        res[state] = new_subdict
    else:
        m = next(iter(new_subdict)).m
        res[tuple(state[i * m: (i + 1) * m] for i in range(state.m // m))] = new_subdict


def conditional_probs(simulator: Type[Backend], input_state: BasicState, circuit: ACircuit,
                      distance: int, anterior_state: List[BasicState] = None, keep_all: bool = False, threshold=False):
    r"""
    :param simulator: a backend without any circuit
    :param input_state: a state defining on which modes the source will be replicated
    :param circuit: A circuit with time delays
    :param distance: Define the time distance between the correlations.
    :param anterior_state: If it's a BasicState list, returns the correlation only for these states.
     By default, gives the correlation for all the states
    :param keep_all: If True, the anterior state will be all the states between distance and the t0_state excluded,
     in a tuple form.
    :param threshold: If True, states are reduced into a threshold state
    :return: dict of {anterior_state(s) : {t0_state : conditional_prob}}.
    """
    # Give the conditional prob output for the steady regime
    m = circuit.m
    TD_number = count_TD(circuit)
    depth = TD_number + (distance + 1 if TD_number else 1)
    real_input = create_input(input_state, depth, TD_number)
    real_circuit = simplify(expand_TD(circuit, depth, True))
    interest_m = (depth - distance) * m if not keep_all else (depth - 1) * m
    sim = simulator(real_circuit)
    real_out_sv = sim.evolve(real_input)

    res = defaultdict(lambda: defaultdict(lambda: 0))

    for state, prob_ampli in real_out_sv.items():
        prob = abs(prob_ampli) ** 2
        if prob != 0:
            last_state = state[(depth - 1) * m: depth * m]
            first_state = state[(depth - distance - 1) * m: interest_m]

            if anterior_state is not None and first_state not in anterior_state:
                continue  # Does not work because first_state is never in anterior_state

            if threshold:
                first_state = threshold_detection(first_state)
                last_state = threshold_detection(last_state)

            res[separate(first_state, m, keep_all)][last_state] += prob

    # Renormalisation
    for first_state in res:
        normalise_dict(res[first_state])

    return res


def is_independent(simulator, input_state, circuit, distance, epsilon=1e-7):
    independent_probs = {state: prob for state, prob in TD_allstateprob_iterator(simulator, input_state, circuit)}
    distant_probs = conditional_probs(simulator, input_state, circuit, distance)

    for sub_dict in distant_probs.values():
        if not epsilon_identical(independent_probs, sub_dict, epsilon):
            return False

    return True


def epsilon_identical(dict_1, dict_2, epsilon):
    for key in dict_1:
        if key in dict_2:
            if abs(dict_1[key] - dict_2[key]) > epsilon:
                return False

        elif abs(dict_1[key]) > epsilon:
            return False

    for key in dict_2:
        if key not in dict_1:
            if abs(dict_2[key]) > epsilon:
                return False

    return True


def threshold_detection(state):
    l = [1 if state[i] else 0 for i in range(state.m)]
    return BasicState(l)


def dict_threshold_detection(input_dist):
    svd = SVDistribution()

    for state, prob in input_dist.items():
        svd[StateVector(threshold_detection(state))] += prob

    return svd


def normalise_dict(subdict):
    # First, normalise subdict
    s = sum(subdict.values())
    if s == 0:
        # It is impossible to reach the prior state
        return

    factor = 1.0 / s

    new_subdict = {key: val * factor for key, val in subdict.items() if
                   val != 0}  # Normalize and remove impossible cases

    subdict.clear()
    subdict.update(new_subdict)


def separate(state, m, keep_all):
    if not keep_all:
        return state
    return tuple(state[i * m: (i+1) * m] for i in range(state.m // m))
