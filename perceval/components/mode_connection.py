
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

from typing import Dict, Callable, Type, Literal, Union

from perceval.components.abstract_component import AComponent


class UnavailableModeException(Exception):
    def __init__(self, mode: Union[int, list[int]], reason: str = None):
        because = ''
        if reason:
            because = f' because: {reason}'
        super().__init__(f"Mode(s) {mode} not available{because}")


class InvalidMappingException(Exception):
    def __init__(self, mapping: dict, reason: str = None):
        because = ''
        if reason:
            because = f' because: {reason}'
        super().__init__(f"Invalid mapping ({mapping}){because}")


class ModeConnectionResolver:
    def __init__(self, left_processor, right_obj):
        self._lp = left_processor
        self._ro = right_obj  # Can either be a component or a processor
        self._r_is_component = isinstance(right_obj, AComponent)
        self._map = {}
        self._l_port_names = None
        self._r_port_names = None
        if self._r_is_component:
            self._n_modes_to_connect = right_obj.m
        else:
            self._n_modes_to_connect = right_obj.mode_of_interest_count

    def _mapping_type_checks(self):
        assert isinstance(self._map, dict), f"Mapping should be a Python dictionnary, got {type(self._map)}"
        for k, v in self._map.items():
            assert isinstance(k, int) or isinstance(k, str), f"Mapping keys supported types are str and int, found {k}"
            if self._r_is_component:
                assert isinstance(v, int) or isinstance(v, list), \
                    "Mapping values must all be integers when the right object is not a processor"
            else:
                assert isinstance(v, int) or isinstance(v, str) or isinstance(v, list), \
                    f"Mapping values supported types are str and int, found {v}"

    def resolve(self, mapping):
        """
        Resolves mode mapping and checks if the mapping is consistent.

        :param mapping: can be an integer or a dictionnary.
         Case int:
            Creates a dictionnary { mapping: 0, mapping+1: 1, ..., mapping+n: n }
         Case dict:
            keys and values can either be integers or strings. If strings, it expects port names of the same size.

        TODO describe consistency checks
        """
        # Handle int input case
        if isinstance(mapping, int):
            self._map = {}
            r_list = list(range(self._n_modes_to_connect))
            if not self._r_is_component:
                r_list = list(range(self._ro.m))
                r_list = [x for x in r_list if x not in list(self._ro.heralds.keys())]
            for i in range(self._n_modes_to_connect):
                self._map[mapping + i] = r_list[i]
            self._check_consistency()
            return self._map

        self._map = mapping
        self._mapping_type_checks()
        result = {}
        for k, v in mapping.items():
            if isinstance(k, int) and isinstance(v, int):
                result[k] = v
            elif isinstance(k, str):  # Mapping between port name and index
                l_idx = self._resolve_port_left(k)
                r_idx = []
                if l_idx is None:
                    raise InvalidMappingException(mapping, f"port '{k}' was not found in processor")
                if isinstance(v, int) and len(l_idx) == 1:
                    result[l_idx[0]] = v
                elif isinstance(v, list):
                    r_idx = v
                else:  # str
                    r_idx = self._resolve_port_right(v)
                    if r_idx is None:
                        raise InvalidMappingException(mapping, f"port '{v}' was not found in processor")
                if len(l_idx) != len(r_idx):
                    raise InvalidMappingException(mapping, f"Unable to resolve '{k}: {v}' - imbalanced ports")
                for i in range(len(l_idx)):
                    result[l_idx[i]] = r_idx[i]
        self._map = result
        self._check_consistency()
        return self._map

    def _check_consistency(self):
        if len(self._map) != self._n_modes_to_connect:
            raise InvalidMappingException(self._map)
        max_out = max(self._map.keys())
        min_out = min(self._map.keys())
        if max_out > self._lp.mode_of_interest_count - 1:
            raise UnavailableModeException(max_out)
        if min_out < 0:
            raise UnavailableModeException(min_out)
        for m_out, m_in in self._map.items():
            if not self._lp.is_mode_connectible(m_out):
                raise UnavailableModeException(m_out)
        m_in = self._map.values()
        if len(m_in) != len(list(dict.fromkeys(m_in))):  # suppress duplicates and check length
            raise InvalidMappingException(self._map)

    def _resolve_port_left(self, name: str):
        if self._l_port_names is None:
            self._l_port_names = self._lp.out_port_names
        count = self._l_port_names.count(name)
        if count == 0:
            return None
        pos = self._l_port_names.index(name)
        res = []
        for i in range(count):
            res.append(pos)
            pos += 1
        return res

    def _resolve_port_right(self, name: str):
        assert not self._r_is_component, "Port names are only available on processors"
        if self._r_port_names is None:
            self._r_port_names = self._ro.in_port_names
        count = self._r_port_names.count(name)
        if count == 0:
            return None
        pos = self._r_port_names.index(name)
        res = []
        for i in range(count):
            res.append(pos)
            pos += 1
        return res
