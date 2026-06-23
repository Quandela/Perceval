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

from __future__ import annotations  # Python 3.11 : Replace using Self typing

import copy
from typing import Iterator, SupportsFloat
from collections.abc import Mapping, Sequence
from .globals import global_params

import exqalibur

class BSDistribution():
    """
    Basic state distribution holding measured states (i.e. perfect Fock states), of the same size (number of modes).

    The ``BSDistribution`` can be build via any of the following parameters:

    :param fs: (optional) build from a single state which gets a probability of 1.
    :param bsd: (optional) build from an existing dictionary or distribution. Keys have to be perfect Fock states.
    :param fsa: (optional) a `FSArray` requiring a same size probability vector
    :param probs: (optional) the probability vector working with parameter `fsa`
    """

    def __init__(self, *args):
        """
        Overloaded function.

        1. __init__(self: BSDistribution) -> None

        empty distribution

        2. __init__(self: BSDistribution, fs: FockState) -> None

        constructor from existing fockstate

        3. __init__(self: BSDistribution, bsd: Mapping[FockState, SupportsFloat]) -> None

        constructor from dict of BasicStates

        4. __init__(self: BSDistribution, bsd: BSDistribution) -> None

        constructor from existing BSD

        5. __init__(self: BSDistribution, fsa: FSArray, probs: Sequence[SupportsFloat]) -> None

        constructor from FsArray and list of probabilities
        """
        try:
            self._function = None
            self._distribution = None
            match len(args):
                case 0:
                    self._function = exqalibur.FSFunction()
                    return
                case 1:
                    if isinstance(args[0], exqalibur.FockState):
                        self._distribution = exqalibur.FSDistribution(args[0])
                        return
                    elif isinstance(args[0], Mapping):
                        self._function = exqalibur.FSFunction(args[0])
                        self._function.trim(global_params["min_p"])
                        return
                    elif isinstance(args[0], BSDistribution):
                        self._function = args[0]._function
                        self._distribution = args[0]._distribution
                        return
                    elif isinstance(args[0], exqalibur.FSFunction):
                        self._function = args[0]
                        self._function.trim(global_params["min_p"])
                        return
                    elif isinstance(args[0], exqalibur.FSDistribution):
                        self._distribution = args[0]
                        self._distribution.trim(global_params["min_p"])
                        return
                case 2:
                    if isinstance(args[0], exqalibur.FSArray) and isinstance(args[1], Sequence):
                        self._function = exqalibur.FSFunction(args[0], args[1])
                        self._function.trim(global_params["min_p"])
                        return
        except TypeError:
            pass
        raise TypeError(
            """__init__(): incompatible constructor arguments. The following argument types are supported:
            1. BSDistribution()
            2. BSDistribution(fs: exqalibur.FockState)
            3. BSDistribution(bsd: collections.abc.Mapping[exqalibur.FockState, typing.SupportsFloat])
            4. BSDistribution(bsd: BSDistribution)
            5. BSDistribution(fsa: exqalibur.FSArray, probs: collections.abc.Sequence[typing.SupportsFloat])""")

    @property
    def _normalized(self) -> bool:
        assert((self._function is None) != (self._distribution is None))
        return self._function is None

    @property
    def _container(self) -> exqalibur.FSFunction | exqalibur.FSDistribution:
        if self._normalized:
            return self._distribution
        else:
            return self._function

    @_container.setter
    def _container(self, value: exqalibur.FSFunction | exqalibur.FSDistribution) -> None:
        if isinstance(value, exqalibur.FSFunction):
            self._value = value
            self._distribution = None
        elif isinstance(value, exqalibur.FSDistribution):
            self._value = None
            self._distribution = value
        else:
            raise TypeError(f"{type(value).__name__} is neither FSFunction nor FSDistribution")

    def normalize(self) -> None:
        """
        Normalize the distribution in place:

        It discards all negative values, then divide by the sum of all coefficients

        An error is raised if the resulting distribution would be empty
        """
        if self._normalized:
            return
        self._distribution = self._function.move_to_distribution()
        self._distribution.trim(global_params["min_p"])
        self._function = None

    def _unnormalize(self) -> None:
        if not self._normalized:
            return
        self._function = self._distribution.move_to_function()
        self._distribution = None

    def __copy__(self) -> BSDistribution:
        return copy.deepcopy(self)

    def __getitem__(self, key: exqalibur.FockState) -> float:
        return self._container.__getitem__(key)

    def __setitem__(self, key: exqalibur.FockState, value: float) -> None:
        self._unnormalize()
        self._function.__setitem__(key, value)

    def __contains__(self, key: exqalibur.FockState) -> bool:
        return self._container.__contains__(key)

    def __delitem__(self, key: exqalibur.FockState) -> None:
        self._unnormalize()
        self._function.__delitem__(key)

    def items(self) -> Iterator[tuple[exqalibur.FockState, float]]:
        """
        Iterate over tuples of (Fock states, probability) contained in the distribution
        """
        return self._container.items()

    def keys(self) -> Iterator[exqalibur.FockState]:
        """
        Iterate over Fock states contained in the distribution
        """
        return self._container.keys()

    def values(self) -> Iterator[float]:
        """
        Iterate over the probabilities contained in the distribution
        """
        return self._container.values()

    def get(self, key: exqalibur.FockState, default: float) -> float:
        """
        Retrieve the probability for a given state, with a default value if the state doesn't exist in the distribution.

        :param fs: State to search
        :param default: Default probability value (defaults to None)
        :return: The state probability if found, the default value otherwise
        """
        return self[key] if key in self else default

    def add(self, key: exqalibur.FockState, value: float) -> None:
        """
        Increment the probability of a given state. If the state doesn't exist beforehand, use the given probability. Probabilities that are too low (1e-16) are discarded.

        :param fs: Fock state
        :param value: Probability
        """
        if abs(value) < global_params["min_p"]:
            return
        self[key] += value

    def __eq__(self, other: BSDistribution) -> bool:
        return self._container.__eq__(other._container)

    def __ne__(self, other: BSDistribution) -> bool:
        return self._container.__ne__(other._container)

    def __len__(self) -> int:
        return self._container.__len__()

    def __iter__(self) -> Iterator[exqalibur.FockState]:
        return self._container.__iter__()

    def __repr__(self) -> str:
        return self._container.__repr__()

    def __str__(self) -> str:
        return self._container.__str__()

    @property
    def m(self) -> int:
        """
        :return: The number of modes of all states in the distribution
        """
        return self._container.m

    def __add__(self, arg :BSDistribution) -> BSDistribution:
        if isinstance(arg, BSDistribution):
            result = copy.copy(self)
            result._unnormalize()
            arg._unnormalize()
            result._function += arg._function
            return result
        raise NotImplemented()

    def __iadd__(self, arg: BSDistribution) -> BSDistribution:
        if isinstance(arg, BSDistribution):
            self._unnormalize()
            arg._unnormalize()
            self._function += arg._function
            return self
        raise NotImplemented()

    def __neg__(self):
        result = copy.copy(self)
        result._unnormalize()
        result._function *= -1
        return result

    def __sub__(self, arg: BSDistribution) -> BSDistribution:
        if isinstance(arg, BSDistribution):
            result = copy.copy(self)
            result._unnormalize()
            arg._unnormalize()
            result._function -= arg._function
            return result
        raise NotImplemented()

    def __isub__(self, arg: BSDistribution) -> BSDistribution:
        if isinstance(arg, BSDistribution):
            self._unnormalize()
            arg._unnormalize()
            self._function -= arg._function
            return self
        raise NotImplemented()

    def __mul__(self, arg: exqalibur.FockState | BSDistribution | float) -> BSDistribution:
        if isinstance(arg, exqalibur.FockState):
            return BSDistribution(self._container * arg)
        elif isinstance(arg, BSDistribution):
            return BSDistribution(self._container * arg._container)
        elif isinstance(arg, SupportsFloat):
            result = copy.copy(self)
            result._unnormalize()
            result._function *= arg
            return result
        raise NotImplemented()

    def __rmul__(self, arg: exqalibur.FockState | float) -> BSDistribution:
        if isinstance(arg, exqalibur.FockState):
            return BSDistribution(arg * self._container)
        elif isinstance(arg, SupportsFloat):
            result = copy.copy(self)
            result._unnormalize()
            result._function *= arg
            return result
        raise NotImplemented()

    def __imul__(self, arg: exqalibur.FockState | BSDistribution | float) -> BSDistribution:
        if isinstance(arg, exqalibur.FockState):
            self._container = self._container * arg
            return self
        elif isinstance(arg, BSDistribution):
            self._container = self._container * arg._container
            return self
        elif isinstance(arg, SupportsFloat):
            self._unnormalize()
            self._function *= arg
            return self
        raise NotImplemented()

    def __truediv__(self, arg: float) -> BSDistribution:
        if isinstance(arg, SupportsFloat):
            result = copy.copy(self)
            result /= arg
            return result
        raise NotImplemented()

    def __itruediv__(self, arg: float) -> BSDistribution:
        if isinstance(arg, SupportsFloat):
            self._unnormalize()
            self._function /= arg
            return self
        raise NotImplemented()

    def __pow__(self, other: BSDistribution) -> BSDistribution:
        return BSDistribution(self._container.__pow__(other))

    def sample(self, count: int, non_null: bool = True) -> exqalibur.BSSamples:
        """
        Generate an ordered list of samples from the distribution.

        :param count: Number of expected samples
        :param non_null: If ``True`` avoids returning in void state (i.e. state containing 0 photon). Defaults to ``True``.
        :return: A list of samples following the probability distribution
        """
        self.normalize()
        return self._distribution.sample(count, non_null)

    def group_modes_simplification(self, group_size: int) -> BSDistribution:
        """
        Group modes by merging their contents in shorter states within the whole distribution.

        This call can be used to perform coarse grain comparison between two very large distributions

        :param group_size: Size of mode groups to consider (e.g. if 2, `|1,1,3,4>` gives `|2,7>`)
        :return: The resulting distribution
        """
        self.normalize()
        return BSDistribution(self._distribution.group_modes_simplification(group_size))

    def photon_threshold_simplification(self, photon_threshold: int) -> BSDistribution:
        """
        Applies a maximum photon per mode threshold to all states in the distribution.

        :param photon_threshold: Max number of photons allowed per mode. Any bigger value will be changed to ``photon_threshold``
        :return: The thresholded distribution
        """
        self.normalize()
        return BSDistribution(self._distribution.photon_threshold_simplification(photon_threshold))

    @staticmethod
    def list_tensor_product(distributions: Sequence[BSDistribution], merge_modes: bool = False, prob_threshold: SupportsFloat = 0.0) -> exqalibur.BSDistribution:
        """
        Compute a series of tensor product between distributions

        :param distributions: List of distributions
        :param merge_modes: If ``True``, resulting states will merge their modes (both distribution must contain states of the same size).
                            Apply a standard tensor product otherwise (defaults to ``False``)
        :param prob_threshold: Threshold under which probabilities are discarded during the tensor product (defaults to ``0.``, i.e. no probability is discarded).
        :return: The result of the tensor product
        """
        if all(distribution._normalized for distribution in distributions):
            result = exqalibur.FSDistribution.list_tensor_product([bsd._distribution for bsd in distributions], merge_modes, prob_threshold)
            return BSDistribution(result)
        else:
            for distribution in distributions:
                distribution._unnormalize()
            result = exqalibur.FSFunction.list_tensor_product([bsd._function for bsd in distributions], merge_modes, prob_threshold)
            # TODO renormalize ?
            return BSDistribution(result)

    @staticmethod
    def tensor_product(bsd1: BSDistribution, bsd2: BSDistribution, merge_modes: bool = False, prob_threshold: SupportsFloat = 0.0) -> exqalibur.BSDistribution:
        """
        Compute the tensor product of two distributions

        :param bsd1: Left hand-side distribution
        :param bsd2: Right hand-side distribution
        :param merge_modes: If ``True``, resulting states will merge their modes (both distribution must contain states of the same size).
                        Apply a standard tensor product otherwise (defaults to ``False``)
        :param prob_threshold: Threshold under which probabilities are discarded during the tensor product (defaults to ``0.``, i.e. no probability is discarded).
        :return: The result of the tensor product
        """
        if bsd1._normalized and bsd2._normalized:
            result = exqalibur.FSDistribution.tensor_product(bsd1._distribution, bsd2._distribution, merge_modes, prob_threshold)
            return BSDistribution(result)
        bsd1._unnormalize()
        bsd2._unnormalize()
        result = exqalibur.FSFunction.tensor_product(bsd1._function, bsd2._function, merge_modes, prob_threshold)
        # TODO renormalize ?
        return BSDistribution(result)
