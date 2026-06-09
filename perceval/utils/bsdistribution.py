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

# import collections
import copy
# from copy import deepcopy
from typing import SupportsFloat
from collections.abc import Mapping, Sequence

import exqalibur

from multipledispatch import dispatch

class BSDistribution():
    def __init__(self, *args):
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
                        self._function.trim(1e-16)
                        return
                    elif isinstance(args[0], BSDistribution):
                        self._function = args[0]._function
                        self._distribution = args[0]._distribution
                        return
                    elif isinstance(args[0], exqalibur.FSFunction):
                        self._function = args[0]
                        self._function.trim(1e-16)
                        return
                    elif isinstance(args[0], exqalibur.FSDistribution):
                        self._distribution = args[0]
                        self._distribution.trim(1e-16)
                        return
                case 2:
                    if isinstance(args[0], exqalibur.FSArray) and isinstance(args[1], Sequence):
                        self._function = exqalibur.FSFunction(args[0], args[1])
                        self._function.trim(1e-16)
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
    def _normalized(self):
        assert((self._function is None) != (self._distribution is None))
        return self._function is None

    @property
    def _container(self):
        if self._normalized:
            return self._distribution
        else:
            return self._function

    def normalize(self) -> None:
        if self._normalized:
            return
        self._distribution = self._function.move_to_distribution()
        self._function = None

    def _unnormalize(self) -> None:
        if not self._normalized:
            return
        self._function = self._distribution.move_to_function()
        self._distribution = None

    def __copy__(self):
        return copy.deepcopy(self)

    def __getitem__(self, key: exqalibur.exqalibur.FockState):
        return self._container.__getitem__(key)

    def __setitem__(self, key: exqalibur.exqalibur.FockState, value: float):
        if value < 1e-16:
            return
        self._unnormalize()
        return self._function.__setitem__(key, value)

    def __contains__(self, key: exqalibur.exqalibur.FockState) -> bool:
        return self._container.__contains__(key)

    def __delitem__(self, key: exqalibur.exqalibur.FockState):
        self._unnormalize()
        self._function.__delitem__(key)

    def items(self):
        return self._container.items()

    def keys(self):
        return self._container.keys()

    def values(self):
        return self._container.values()

    ## non-sense!
    def get(self, key: exqalibur.exqalibur.FockState, default: float):
        if key in self:
            return self.__getitem__(key)
        else:
            return default

    ## non-sense!
    def add(self, key: exqalibur.exqalibur.FockState, value: float):
        if value < 1e-16:
            return
        return self.__setitem__(key, self.__getitem__(key) + value)


    def __eq__(self, other):
        return self._container.__eq__(other._container)

    def __ne__(self, other):
        return self._container.__ne__(other._container)

    def __len__(self):
        return self._container.__len__()

    def __iter__(self):
        return self._container.__iter__()

    def __repr__(self):
        return self._container.__repr__()

    def __str__(self):
        return self._container.__str__()


    @property
    def m(self):
        return self._container.m

    def __add__(self, arg):
        if isinstance(arg, BSDistribution):
            result = copy(self)
            result._unnormalize()
            arg._unnormalize()
            result._function += arg._function
            return result
        raise NotImplemented()

    def __iadd__(self, arg):
        if isinstance(arg, BSDistribution):
            self._unnormalize()
            arg._unnormalize()
            self._function += arg._function
            return self
        raise NotImplemented()

    def __sub__(self, arg):
        if isinstance(arg, BSDistribution):
            result = copy(self)
            result._unnormalize()
            arg._unnormalize()
            result._function -= arg._function
            return result
        raise NotImplemented()

    def __isub__(self, arg):
        if isinstance(arg, BSDistribution):
            self._unnormalize()
            arg._unnormalize()
            self._function -= arg._function
            return self
        raise NotImplemented()

    def __mul__(self, arg):
        if isinstance(arg, exqalibur.FockState):
            if self._normalized:
                return BSDistribution(self._distribution * arg)
            else:
                return BSDistribution(self._function * arg)
        elif isinstance(arg, BSDistribution):
            if self._normalized and arg._normalized:
                return BSDistribution(self._distribution * arg._distribution)
            else:
                return BSDistribution(self._container * arg._container)
        elif isinstance(arg, SupportsFloat):
            result = copy(self)
            result._unnormalize()
            result._function *= arg
            return result
        raise NotImplemented()

    def __rmul__(self, arg):
        if isinstance(arg, exqalibur.FockState):
            if self._normalized:
                return BSDistribution(arg * self._distribution)
            else:
                return BSDistribution(BSDistribution(), arg * self._function)
        elif isinstance(arg, SupportsFloat):
            result = copy(self)
            result._unnormalize()
            result._function *= arg
            return result
        raise NotImplemented()

    def __imul__(self, arg):
        if isinstance(arg, exqalibur.FockState):
            if self._normalized:
                self._distribution = self._distribution * arg
            else:
                self._function *= arg
            return self
        elif isinstance(arg, BSDistribution):
            if self._normalized and arg._normalized:
                self._distribution = BSDistribution(self._distribution * arg._distribution)
            else:
                self._function = BSDistribution(self._container * arg._container)
            return self
        elif isinstance(arg, SupportsFloat):
            self._unnormalize()
            self._function *= arg
            return self
        raise NotImplemented()

    def __div__(self, arg):
        if isinstance(arg, SupportsFloat):
            result = copy(self)
            result._unnormalize()
            result /= arg
            return result
        raise NotImplemented()

    def __idiv__(self, arg):
        if isinstance(arg, SupportsFloat):
            self._unnormalize()
            self._function /= arg
            return self
        raise NotImplemented()

    def __pow__(self, other):
        if self._normalized:
            return BSDistribution(self._distribution.__pow__(other))
        else:
            return BSDistribution(self._function.__pow__(other))

    def sample(self, count, non_null = True):
        self.normalize()
        return self._distribution.sample(count, non_null)

    def group_modes_simplification(self, group_size):
        self.normalize()
        return BSDistribution(self._distribution.group_modes_simplification(group_size))

    def photon_threshold_simplification(self, photon_threshold):
        self.normalize()
        return BSDistribution(self._distribution.photon_threshold_simplification(photon_threshold))


    def list_tensor_product(distributions: Sequence[BSDistribution], merge_modes: bool = False, prob_threshold: SupportsFloat = 0.0) -> exqalibur.exqalibur.BSDistribution:
        if all(distribution._normalized for distribution in distributions):
            result = exqalibur.FSDistribution.list_tensor_product([bsd._distribution for bsd in distributions], merge_modes, prob_threshold)
            return BSDistribution(result)
        else:
            for distribution in distributions:
                distribution._unnormalize()
            result = exqalibur.FSFunction.list_tensor_product([bsd._function for bsd in distributions], merge_modes, prob_threshold)
            # TODO renormalize ?
            return BSDistribution(result)

    def tensor_product(bsd1: BSDistribution, bsd2: BSDistribution, merge_modes: bool = False, prob_threshold: SupportsFloat = 0.0) -> exqalibur.exqalibur.BSDistribution:
        if bsd1._normalized and bsd2._normalized:
            result = exqalibur.FSDistribution.tensor_product(bsd1._distribution, bsd2._distribution, merge_modes, prob_threshold)
            return BSDistribution(result)
        bsd1._unnormalize()
        bsd2._unnormalize()
        result = exqalibur.FSFunction.tensor_product(bsd1._function, bsd2._function, merge_modes, prob_threshold)
        # TODO renormalize ?
        return BSDistribution(result)
