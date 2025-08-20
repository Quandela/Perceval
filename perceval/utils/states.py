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

from collections import defaultdict

from exqalibur import Annotation
from multipledispatch import dispatch
from typing import Generator, Union, final

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias  # Only used with python 3.9

import exqalibur as xq


FockState: TypeAlias = xq.FockState
NoisyFockState: TypeAlias = xq.NoisyFockState
AnnotatedFockState: TypeAlias = xq.AnnotatedFockState
BSCount: TypeAlias = xq.BSCount
BSSamples: TypeAlias = xq.BSSamples
StateVector: TypeAlias = xq.StateVector
BSDistribution: TypeAlias = xq.BSDistribution
SVDistribution: TypeAlias = xq.SVDistribution

class BasicStateMeta(type):
    _state_classes = {FockState, NoisyFockState, AnnotatedFockState}

    def __instancecheck__(cls, inst):
        """Implement isinstance(inst, cls)."""
        return cls.__subclasscheck__(type(inst))

    def __subclasscheck__(cls, sub):
        """Implement issubclass(sub, cls)."""
        return any(c in cls._state_classes for c in sub.mro()) or sub is cls

State: TypeAlias = Union[FockState, NoisyFockState, AnnotatedFockState]

@final
class BasicState(metaclass=BasicStateMeta):

    @dispatch(type, FockState)
    def __new__(cls, fs: FockState):
        return FockState(fs)

    @dispatch(type, NoisyFockState)
    def __new__(cls, fs: NoisyFockState):
        return NoisyFockState(fs)

    @dispatch(type, AnnotatedFockState)
    def __new__(cls, fs: AnnotatedFockState):
        return AnnotatedFockState(fs)

    @dispatch(type)
    def __new__(cls):
        return FockState()

    @dispatch(type, (list, tuple))
    def __new__(cls, photons: list[int]):
        return FockState(photons)

    @dispatch(type, int)
    def __new__(cls, m: int):
        return FockState(m)

    @dispatch(type, (list, tuple), (list, tuple))
    def __new__(cls, photons: list[int], noise: list[int]):
        return NoisyFockState(FockState(photons), noise)

    @dispatch(type, str)
    def __new__(cls, source: str):
        parsed_state = xq.ParsedBasicState(source)
        if parsed_state.is_noisy():
            return NoisyFockState(parsed_state)
        elif parsed_state.is_annotated():
            return AnnotatedFockState(parsed_state)
        return FockState(parsed_state)

    # All the following methods are here to make the linter happy
    @property
    def m(self) -> int:
        """
        :return: The number of modes
        """
        return 0

    @property
    def n(self) -> int:
        """
        :return: The number of photons
        """
        return 0

    def __len__(self) -> int:
        return 0

    def __eq__(self, other) -> bool:
        return False

    def __ne__(self, other) -> bool:
        return True

    def __getitem__(self, item):
        return 0

    def __mul__(self, other: State | int | float | complex) -> State | StateVector | BSDistribution:
        return StateVector()

    def __rmul__(self, other: State | int | float | complex) -> State | StateVector | BSDistribution:
        return StateVector()

    def __pow__(self, power) -> State:
        return FockState()

    def __iter__(self) -> Generator[int, None, None]:
        yield 0

    def merge(self, other: BasicState) -> State:
        """
        :param other: a BasicState with the same number of modes than self.
        :return: A new state for which the photons in one mode are the photons in this mode in ``self`` and in ``other``.
         the type of the state is automatically inferred from ``self`` and ``other`` so that it can contain all the information.
         Rules: FockState photons are converted to {0} for a NoisyFockState.
                NoisyFockState photons are converted to {_:`noise_tag`} for a AnnotatedFockState.
        """

    def __add__(self, other) -> State | StateVector | BSDistribution:
        return StateVector()

    def __radd__(self, other) -> State | StateVector | BSDistribution:
        return StateVector()

    def __sub__(self, other) -> State | StateVector | BSDistribution:
        return StateVector()

    def __rsub__(self, other) -> State | StateVector | BSDistribution:
        return StateVector()

    @property
    def has_polarization(self) -> bool:
        return False

    @property
    def has_annotations(self) -> bool:
        return False

    def separate_state(self) -> list[FockState]:
        """
        :return: A list of states where each state represents a collection of indistinguishable photons from the original state
        """

    def split_state(self) -> dict[int, FockState]:
        """
        :return: A dict of states where each state represents a collection of indistinguishable photons from the original state,
         associated with the noise tag they were defined with.
        """

    def partition(self, photon_nb: list[int]) -> list[list[FockState]]:
        """
        :param photon_nb: a list of photon numbers. The sum of this list must be equal to self.n
        :return: A list containing all lists of states such that the merge of all these states is self,
         where each state of the sublists has the number of photons specified in ``photon_nb``.
        """

    def clear_annotations(self) -> FockState:
        """
        :return: A new state where the photons are on the same modes as self, with no noise or annotations.
        """

    def get_mode_annotations(self, mode: int) -> list[Annotation] | list[int]:
        """
        :param mode: The mode to return annotations for.
        :return: A list of len self[mode] containing the annotations of mode ``mode``.
         They can be Annotation for AnnotatedFockState, or integers for NoisyFockState.
        """

    def get_photon_annotation(self, photon: int) -> Annotation | int:
        """
        :param photon: The photon to look at (ordered by ascending mode number)
        :return: The Annotation of the photon if self is a AnnotatedFockState, or the noise tag if self is a NoisyFockState.
        """

    def inject_annotation(self, annot: Annotation | int) -> AnnotatedFockState | NoisyFockState:
        """
        :param annot: The annotation to inject. If this is an Annotation, the result will be an AnnotatedFockState.
          If this is an integer, the result will be an NoisyFockState.
        :return: A new BasicState where all photons now have the given annotation.
        """

    def mode2photon(self, mode: int) -> int:
        """
        :param mode: The mode of the photon.
        :return: The index of the first photon in the given mode (when photons are ordered mode by mode),
         or -1 if the mode is empty.
        """

    def photon2mode(self, n: int) -> int:
        """
        :param n: The photon index.
        :return: The mode of the nth photon, where modes are in ascending order.
        """

    def prodnfact(self) -> float:
        """
        :return: The product self[0]! self[1]! ... self[m-1]!
        """

    def remove_modes(self, modes: list[int]) -> State:
        """
        :param modes: The list of modes to remove.
        :return: A new state with only the modes that are not in ``modes``
        """

    def set_slice(self, other: State, start: int, end: int) -> State:
        """
        :param other: The state to replace part of ``self`` with. Must have :math:`end - start` modes
        :param start: The first mode to replace.
        :param end: The last mode to replace (excluded).
        :return: A new state, promoted to the smallest possible type given ``self`` and ``other``,
         where the section between start and end is ``other``, and the remaining comes from ``self``.
        """

    def threshold_detection(self, nb: int = 1):
        """
        :param nb: Maximum number of photons per mode.
        :return: a new FockState where the photon count of mode m is min(nb, self[m])
        """


def allstate_array(input_state: BasicState, mask: xq.FSMask = None) -> xq.FSArray:
    m = input_state.m
    n = input_state.n
    if mask is not None:
        output_array = xq.FSArray(m, n, mask)
    else:
        output_array = xq.FSArray(m, n)

    output_array.generate()

    return output_array


def allstate_iterator(input_state: BasicState | StateVector, mask: xq.FSMask = None) -> Generator[xq.FockState]:
    """Iterator on all possible output states compatible with mask generating StateVector

    :param input_state: a given input state vector
    :param mask: an optional mask
    :return: a single state in the Fock space of the input state. When all the space is covered, the iteration ends.
    """
    m = input_state.m
    ns = input_state.n
    ns = [ns] if isinstance(ns, int) else list(ns)

    for n in ns:
        if mask is not None:
            output_array = xq.FSArray(m, n, mask)
        else:
            output_array = xq.FSArray(m, n)
        for output_state in output_array:
            yield output_state


def max_photon_state_iterator(m: int, n_max: int):
    """Iterator on all possible output state on m modes with at most n_max photons

    :param m: number of modes
    :param n_max: maximum number of photons
    :return: a single state containing from 0 to n_max photons. When all the space is covered, the iteration ends.
    """
    for n in range(n_max+1):
        output_array = xq.FSArray(m, n)
        for output_state in output_array:
            yield output_state


@dispatch(StateVector, annot_tag=str)
def anonymize_annotations(sv: StateVector, annot_tag: str = "a") -> StateVector:
    # This algo anonymizes annotations but is not enough to have superposed states exactly the same given the storage
    # order of BasicStates inside the StateVector
    m = sv.m
    annot_map = {}
    result = StateVector()
    for bs, pa in sv:
        if isinstance(bs, FockState):
            result += StateVector(bs) * pa
        else:
            s = [""] * m
            for i in range(bs.n):
                mode = bs.photon2mode(i)
                annot = bs.get_photon_annotation(i)
                if annot_map.get(str(annot)) is None:
                    if isinstance(bs, NoisyFockState):
                        annot_map[str(annot)] = f"{{{len(annot_map)}}}"
                    else:
                        annot_map[str(annot)] = f"{{{annot_tag}:{len(annot_map)}}}"
                s[mode] += annot_map[str(annot)]
            if isinstance(bs, NoisyFockState):
                result += StateVector(NoisyFockState("|" + ",".join([v and v or "0" for v in s]) + ">")) * pa
            else:
                result += StateVector(AnnotatedFockState("|" + ",".join([v and v or "0" for v in s]) + ">")) * pa
    result.normalize()
    return result


@dispatch(SVDistribution, annot_tag=str)
def anonymize_annotations(svd: SVDistribution, annot_tag: str = "a") -> SVDistribution:
    sv_dist = defaultdict(lambda: 0)
    for k, p in svd.items():
        state = anonymize_annotations(k, annot_tag=annot_tag)
        sv_dist[state] += p
    return SVDistribution({k: v for k, v in sorted(sv_dist.items(), key=lambda x: -x[1])})


@dispatch(BSDistribution, int)
def filter_distribution_photon_count(bsd: BSDistribution, min_photons_filter: int) -> tuple[BSDistribution, float]:
    """
    Filter the states of a BSDistribution to keep only those having state.n >= min_photons_filter

    :param bsd: the BSDistribution to filter out
    :param min_photons_filter: the minimum number of photons required to keep a state
    :return: a tuple containing the normalized filtered BSDistribution and the probability that the state is kept
    """
    if min_photons_filter == 0:
        return bsd, 1

    res = BSDistribution({state: prob for state, prob in bsd.items() if state.n >= min_photons_filter})
    perf = sum(res.values())

    if len(res):
        res.normalize()
    return res, perf


@dispatch(SVDistribution, int)
def filter_distribution_photon_count(svd: SVDistribution, min_photons_filter: int) -> tuple[BSDistribution, float]:
    """
    Filter the states of a BSDistribution to keep only those having state.n >= min_photons_filter

    :param svd: the SVDistribution to filter out
    :param min_photons_filter: the minimum number of photons required to keep a state
    :return: a tuple containing the normalized filtered BSDistribution and the probability that the state is kept
    """
    if min_photons_filter == 0:
        return svd, 1

    res = SVDistribution({state_v: prob for state_v, prob in svd.items() for state in state_v.keys() if state.n >= min_photons_filter})
    perf = sum(res.values())

    if len(res):
        res.normalize()
    return res, perf
