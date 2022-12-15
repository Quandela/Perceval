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
from typing import Union

import numpy as np
import perceval.components.unitary_components as comp
from perceval.components import ACircuit, Circuit


def simplify(circuit: Union[list, ACircuit], m: int=None, display: bool = False) -> Union[list, Circuit]:
    r"""
    Tries to simplify a circuit when simplifications are possible

    :param circuit: the circuit to simplify
    :param display: Slightly changes the behaviour of the func, should give a better visualisation. Use False for computing
    :return: The simplified circuit
    """
    final_circuit_comp = []

    if isinstance(circuit, ACircuit):
        m = circuit.m
    else:
        assert m is not None, "m must be specified"

    for r, c in circuit:
        if isinstance(r, int):
            r = tuple(r + i for i in range(c.m))
        final_circuit_comp.append([r, c])
        final_circuit_comp = _simplify_comp(final_circuit_comp, m, display)

    if isinstance(circuit, Circuit):
        res = Circuit(m)
        for r, c in final_circuit_comp:
            res.add(r, c)
        return res

    return final_circuit_comp


def _simplify_comp(components, m, display):
    # Simplify the circuit according to the last added component
    [_, c] = components[-1]

    if isinstance(c, comp.PERM):
        return _simplify_perm(components, m, display)
    if isinstance(c, comp.PS):
        return _simplify_PS(components, m, display)

    else:
        return components


# Permutation simplifications


###################################################################
# These functions act directly on permutation list; could be useful outside the simplification
def extend_perm(r, perm_list, m):
    M_r = r[-1] + 1

    new_perm = list(range(r[0])) + [perm_list[i] + r[0] for i in range(len(perm_list))] + list(range(M_r, m))

    return list(range(m)), new_perm


def perm_compose(left_r, left_perm, right_r, right_perm):
    max_r = max(left_r[-1] + 1, right_r[-1] + 1)

    # Resize the perm lists, so they begin at mode 0 and end at the same mode
    new_r, left_perm = extend_perm(left_r, left_perm, max_r)
    right_perm = extend_perm(right_r, right_perm, max_r)[1]

    # Now compose the perms
    new_perm = [right_perm[left_perm[i]] for i in range(len(right_perm))]

    return list(range(max_r)), new_perm


def reduce_perm(r, perm):
    n = len(perm)
    for i in range(n):
        if perm[i] != i:
            break

    for j in range(n - 1, -1, -1):
        if perm[j] != j:
            break

    perm = [perm[k] - i for k in range(i, j + 1)]

    return r[i: j + 1], perm


def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    # Conversion to python list of integers, necessary for comp.PERM creation
    return [inv[i].item() for i in range(len(inv))]


###################################################################################################


def _update_adjacent(adjacent_modes, r):
    min_r = r[0]
    i = 0
    while i < len(adjacent_modes):  # Needed as the size of adjacent_modes will change
        modes = adjacent_modes[i]
        if min_r in modes:
            adjacent_modes[i] = list(set(modes).union(set(r)))
        else:
            for mode in modes:
                if mode in r:
                    adjacent_modes.pop(i)
                    i -= 1
                    break
        i += 1


def _generate_compatible_perm(perm_list, adjacent_modes):
    # Assumes the perm_list corresponds to the permutation at the right
    m = len(perm_list)

    reverse = [-1] * m
    first_step = []
    second_step = []
    third_step = []

    for modes in adjacent_modes:
        if len(modes) > 1:
            m_mode = modes[0]
            M_mode = modes[-1]
            out = [perm_list[mode] for mode in modes]
            min_out = min(out)
            max_out = max(out)
            if max_out - min_out == M_mode - m_mode:
                # The modes are kept adjacent by the permutation, keep it for the second phase
                second_step.append(modes)

            else:
                # Some modes diverge
                first_step.append(modes)

        else:
            third_step.append(modes[0])

    first_step.sort(key=lambda modes: min(perm_list[modes[i]] for i in range(len(modes))))
    second_step.sort(key=lambda modes: min(perm_list[modes[i]] for i in range(len(modes))))
    third_step.sort(key=lambda mode: perm_list[mode])
    third_step.reverse()  # Visit the lonely modes in the reverse order to maximize the number of crosses

    for step in (first_step, second_step):
        for modes in step:
            out = [perm_list[mode] for mode in modes]
            reverse = _update_perm(reverse, min(out), modes)

    reverse2 = reverse.copy()
    # Now fill the holes with single modes (adjacent_modes is ordered so modes will not be too much moved)
    for mode in third_step:
        reverse2 = _update_perm(reverse2, perm_list[mode], [mode])

    # Check if no permutation occurs, it's always worth to try something
    if reverse2 == list(range(m)):
        # We try to shuffle the lonely modes
        third_step.reverse()
        for mode in third_step:
            reverse = _update_perm(reverse, perm_list[mode], [mode])
    else:
        reverse = reverse2

    # Now, we have to find the permutation such that the created permutation with this one is perm_list
    right_perm = perm_compose(tuple(range(m)), reverse, tuple(range(m)), perm_list)[1]

    # return left_perm, right_perm
    return reverse, invert_permutation(right_perm)


def _search_empty_space(perm, n, init):
    # Find the nearest non-attributed space of len n in the permutation
    target = n * [-1]

    for i in range(len(perm)):
        if i + init + n <= len(perm) and perm[init + i: init + i + n] == target:
            return i, n

        if init - i >= 0 and perm[init - i: init - i + n] == target:
            return -i, n

    # If there is no available space, seek for a smaller space; perm will be changed
    return _search_empty_space(perm, n - 1, init)


def _update_perm(perm, init, modes):
    m = len(perm)
    i, n = _search_empty_space(perm, len(modes), init)
    slice_min = init + i
    slice_max = init + i + n
    j_right = 0
    j_left = 1
    while len(modes) - n:
        # There is not enough space, some modes must be moved
        cur_index = slice_max + j_right
        if cur_index < m and perm[cur_index] == -1:
            perm[slice_max + 1: cur_index + 1] = perm[slice_max: cur_index]
            perm[slice_max] = -1
            slice_max += 1
            n += 1
        else:
            j_right += 1

        if not len(modes) - n:
            break

        cur_index = slice_min - j_left
        if cur_index >= 0 and perm[cur_index] == -1:
            perm[cur_index: slice_min - 1] = perm[cur_index + 1: slice_min]
            perm[slice_min - 1] = -1
            slice_min -= 1
            n += 1
        else:
            j_left += 1

    perm[slice_min: slice_max] = modes

    return perm


def _move_comp(in_components, perm):
    # Perm is the inverse of the right left permutation
    new_in_comp = []

    for r, c in in_components:
        mode = perm[r[0]]
        new_r = [mode + i for i in range(len(r))]

        new_in_comp.append([new_r, c])

    return new_in_comp


def _evaluate_perm(left_perm_list, right_perm_list, display):
    if display:
        # Useful in case of display
        s = 0

        for i in range(len(left_perm_list)):
            if i != left_perm_list[i]:
                s += abs(left_perm_list[i] - i)
                s += 1

        return s + len(left_perm_list) + len(right_perm_list)

    else:
        # Useful in term of computation for the Stepper
        return len(left_perm_list) + len(right_perm_list)


def _simplify_perm(components, m, display):
    [r, c] = components.pop()

    end_components = components
    # Check several permutations
    found_other_perm = False

    for i in range(len(components) - 1, -1, -1):

        [old_r, old_c] = components[i]
        if isinstance(old_c, comp.PERM):
            found_other_perm = True

            adjacent_modes = [[j] for j in range(m)]
            in_components = []
            for j in range(i + 1, len(components)):
                mid_r, mid_c = components[j]
                _update_adjacent(adjacent_modes, mid_r)
                in_components.append([mid_r, mid_c])

            in_components.reverse()
            break

    if found_other_perm and i == len(components) - 1:  # The permutations are successive

        [left_r, left_c] = end_components.pop(-1)
        left_perm = left_c.perm_vector
        perm = c.perm_vector
        new_r, new_c_perm = perm_compose(left_r, left_perm, r, perm)
        new_r, new_c_perm = reduce_perm(new_r, new_c_perm)

        if len(new_r):
            end_components.append([new_r, comp.PERM(new_c_perm)])

    elif found_other_perm and len(adjacent_modes) > 1:  # Non-successive permutations and things to do

        # Simulates an unraveling on a smaller circuit with only the permutations
        # First, we extend our permutations to the entire circuit
        extended_r, c_list = extend_perm(r, c.perm_vector, m)
        old_c_list = extend_perm(old_r, old_c.perm_vector, m)[1]

        # Then we generate permutations that are compatible with our dependent modes
        left_right_perm, left_left_perm = _generate_compatible_perm(invert_permutation(old_c_list), adjacent_modes)

        # Now we can unravel the middle compatible permutation
        right_perm = perm_compose(extended_r, left_right_perm, extended_r, c_list)[1]
        right_r, right_perm = reduce_perm(extended_r, right_perm)

        # Score evaluation
        left_r, left_perm = reduce_perm(extended_r, left_left_perm)
        old_score = _evaluate_perm(reduce_perm(extended_r, old_c_list)[1], reduce_perm(extended_r, c_list)[1], display)
        new_score = _evaluate_perm(left_perm, right_perm, display)

        if old_score > new_score:
            # We now have to move the components
            new_in_comp = _move_comp(in_components, invert_permutation(left_right_perm))

            end_components = components[:i]
            if len(left_r):
                end_components.append([left_r, comp.PERM(left_perm)])

            end_components += new_in_comp
            if len(right_r):
                end_components.append([right_r, comp.PERM(right_perm)])

        else:
            r, c = reduce_perm(extended_r, c_list)
            end_components.append([r, comp.PERM(c)])

    else:  # It's the only permutation or it can't be simplified with the previous one
        r, c = reduce_perm(r, c.perm_vector)
        if len(r):
            end_components.append([r, comp.PERM(c)])

    return end_components


# Phase shifter simplification
def _simplify_PS(components, m, display):
    # For now, assume all value are numeric
    [r, c] = components.pop()
    r0 = r[0]

    found_PS = False

    phi = c.get_variables()["phi"]

    if not isinstance(phi, str):
        for i in range(len(components) - 1, -1, -1):

            [old_r, old_c] = components[i]
            if isinstance(old_c, comp.PS) and r0 == old_r[0]:
                found_PS = True

                old_phi = old_c.get_variables()["phi"]
                if not isinstance(old_phi, str):
                    new_phi = phi + old_phi
                    if new_phi % (2 * np.pi) != 0 or display:
                        new_c = comp.PS(new_phi)

                        components[i] = [old_r, new_c]

                    else:
                        components.pop(i)

                    break

            elif isinstance(old_c, comp.PERM):
                perm_list = extend_perm(old_r, old_c.perm_vector, m)[1]
                r0 = invert_permutation(perm_list)[r0]

            elif r0 in old_r:
                break

        if not found_PS and (c.get_variables()["phi"] % (2 * np.pi) or display):
            components.append([r, c])

    else:
        components.append([r, c])

    return components
