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

from .time_delays_generic import expand_TD, count_TD, separate, normalise_dict, threshold_detection

from perceval import BasicState, Processor, Source, ACircuit, Backend
from perceval.utils.algorithms.simplification import simplify

import quandelibc as qc

from typing import Type

from collections import defaultdict


def create_processor_input(input_state, depth, source):
    m = input_state.m
    dist = dict()
    for i in range(depth):
        for j in range(m):
            if input_state[j]:
                dist[i * m + j] = source

    return dist


def TD_allstateprob_iterator_processor(simulator: Type[Backend], input_state: BasicState, circuit, source):
    # Give the prob output for the steady regime
    assert max(list(
        input_state)) == 1, "processor can not be used with input states having more than one photon on the same mode"
    n = input_state.n * (1 if source.purity == 1 else 2)
    m = circuit.m
    TD_number = count_TD(circuit)
    depth = TD_number + 1
    real_circuit = simplify(expand_TD(circuit, depth, True))
    source_distribution = create_processor_input(input_state, depth, source)
    interest_m = depth * m

    p = Processor(source_distribution, real_circuit)
    _, real_out_svd = p.run(simulator)

    for k in range(n * (1 + TD_number) + 1):
        for out_state in qc.FSArray(m, k):
            # Simply get all possible output states
            out_prob = 0
            out_state = BasicState(out_state)

            for real_out in real_out_svd:
                real_out_bs = real_out[0]  # real_out is actually a StateVector
                if list(real_out_bs[(depth - 1) * m: interest_m]) == list(out_state):
                    out_prob += real_out_svd[real_out]

            yield out_state, out_prob


def conditional_probs_processor(simulator: Type[Backend], input_state: BasicState, circuit: ACircuit,
                                distance: int, source: Source, anterior_state=None, keep_all=False, threshold=False):
    r"""
    :param simulator: a backend class with which the computation will be done
    :param input_state: a state defining on which modes the source will be replicated
    :param circuit: A circuit with time delays
    :param distance: Define the time distance between the correlations.
    :param anterior_state: If it's a BasicState list, returns the correlation only for these states.
     By default, gives the correlation for all the states (computational cost is the same anyway)
    :param keep_all: If True, the anterior state will be all the states between distance and the t0_state excluded,
     in a tuple form.
    :param source: The source to be replicated.
    :param threshold: If True, states are reduced into a threshold state
    :return: dict of {anterior_state(s) : {t0_state : conditional_prob}}.
    """
    # Give the conditional prob output for the steady regime
    assert max(list(
        input_state)) <= 1, "processor can not be used with input states having more than one photon on the same mode"
    m = circuit.m
    TD_number = count_TD(circuit)
    depth = TD_number + (distance + 1 if TD_number else 1)
    source_distribution = create_processor_input(input_state, depth, source)
    real_circuit = simplify(expand_TD(circuit, depth, True))
    interest_m = depth * m

    p = Processor(source_distribution, real_circuit)
    _, real_out_svd = p.run(simulator)

    res = defaultdict(lambda: defaultdict(lambda: 0))

    for state, prob in real_out_svd.items():
        state_bs = state[0]
        last_state = state_bs[(depth - 1) * m: depth * m]
        first_state = state_bs[(depth - distance - 1) * m: interest_m]

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
