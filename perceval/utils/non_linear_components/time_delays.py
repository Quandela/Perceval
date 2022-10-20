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
from enum import Enum

import perceval.components.unitary_components as comp
import perceval.components.non_unitary_components as ncomp
from perceval.components import Circuit, ACircuit


class compType(Enum):
    linear = 0
    non_linear = 1
    td = 2


def expand_TD(circuit, depth, m, TD_number, pre_computing=False):
    r"""
    :param circuit: A flat component list [r, c] with time delays
    :param depth: The depth to which the circuit has to be expanded
    :param pre_computing: If set to True, intermediary matrices will be computed. Useful for Stepper (?)
    """
    U = []

    cur_U = []
    for r, c in circuit:
        if isinstance(c, ACircuit):
            cur_U.append([r, c])

        elif not isinstance(c, ncomp.TD):
            U.append([compType.linear, cur_U.copy()])
            cur_U = []
            U.append([compType.non_linear, [[r, c]]])

        else:
            t = int(c.get_variables()["t"])

            r0 = r[0]

            if r0 != m - 1:
                # Add a fake permutation to put the TD at the last mode
                r = list(range(r0, m))
                perm_list = [m - r0 - 1] + list(range(1, m - r0 - 1)) + [0]
                cur_U.append([r, comp.PERM(perm_list)])

            for _ in range(t):
                U.append([compType.linear, cur_U.copy()])
                U.append([compType.td, []])
                cur_U = []

            if r0 != m - 1:
                # Nullify the fake permutation
                cur_U.append([r, comp.PERM(perm_list)])

    U.append([compType.linear, cur_U.copy()])

    if pre_computing:
        for i, type_and_cur_U in enumerate(U):
            # First, we add all linear components
            if type_and_cur_U[0] == compType.linear:
                cur_U = type_and_cur_U[1]
                U_circ = Circuit(m)
                for r, c in cur_U:
                    U_circ.add(r, c)

                U_circ = comp.Unitary(U_circ.compute_unitary(use_symbolic=False))

                U[i] = [compType.linear, [[list(range(m)), U_circ]]]

    new_m = depth * m + TD_number
    new_circ = []

    for d in range(depth):
        i_td = 0
        for i, type_and_cur_U in enumerate(U):
            ctype, cur_U = type_and_cur_U
            if ctype != compType.td:
                for r, c in cur_U:
                    new_circ.append((r[0] + d * m, c))

            # Then we permute to mimic TD
            else:
                r0 = (d + 1) * m - 1
                perm_list = [new_m - i_td - r0 - 1] + list(range(1, new_m - i_td - r0 - 1)) + [0]
                i_td += 1
                new_circ.append((r0, comp.PERM(perm_list)))

    return new_circ, new_m


# def create_input(state, depth, TD_number: Union[int, ACircuit]):
#     if isinstance(TD_number, ACircuit):
#         TD_number = count_TD(TD_number)
#     out = state ** depth
#     out *= BasicState([0] * TD_number)
#     return out


def count_TD(circuit):
    return int(sum([c.get_variables()["t"] if isinstance(c, ncomp.TD) else 0 for _, c in circuit]))


def count_independant_TD(circuit, m):
    count = 0
    count_per_mode = [[0, {i}] for i in range(m)]
    for r, c in circuit:
        if isinstance(c, ncomp.TD):
            t = c.get_variables()["t"]
            assert not isinstance(t, str), "Time parameter %s not set" % t
            if t != int(t):
                raise NotImplementedError("Non integer time delays not implemented")
            t = int(t)
            r = r[0]
            if not len(count_per_mode[r][1]) == 1:
                cur_count = max(count_per_mode[i][0] for i in count_per_mode[r][1])
                count += cur_count
                for mode in count_per_mode[r][1]:
                    count_per_mode[mode][1] = {mode}
                for i in range(m):
                    count_per_mode[i][0] = max(0, count_per_mode[i][0] - cur_count)

            count_per_mode[r][0] += t

        elif len(r) > 1:
            set_r = set(r)
            for mode in r:
                count_per_mode[mode][1] |= set_r

    count += max(count_per_mode[i][0] for i in range(m))
    return count


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

'''
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

    Warning: this function is not up to date and may not work
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
'''
