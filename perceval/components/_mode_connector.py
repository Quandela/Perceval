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

from typing import Dict, List, Union
import warnings

from .abstract_component import AComponent
from .unitary_components import PERM


class UnavailableModeException(Exception):
    def __init__(self, mode: Union[int, List[int]], reason: str = None):
        because = ''
        if reason:
            because = f' because: {reason}'
        super().__init__(f"Mode(s) {mode} not available{because}")


class InvalidMappingException(Exception):
    def __init__(self, mapping: Union[Dict, List], reason: str = None):
        because = ''
        if reason:
            because = f' because: {reason}'
        super().__init__(f"Invalid mapping ({mapping}){because}")


class ModeConnector:
    """
    Resolves a mapping supporting multiple syntaxes in order to connect two objects.
    The left object must be a Processor
    The right object can be a Processor, a (linear or non-linear) component
    """

    def __init__(self, left_processor, right_obj, mapping):
        self._lp = left_processor
        self._ro = right_obj  # Can either be a component or a processor
        self._r_is_component = isinstance(right_obj, AComponent)  # False means it is a Processor
        self._map = mapping
        self._l_port_names = None
        self._r_port_names = None
        self._n_modes_to_connect = right_obj.m

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

    def _get_ordered_rmodes(self):
        """
        Returns ordered mode of interest index (i.e. ignores heralded modes in case the right object is a Processor)
        """
        if self._r_is_component:
            return list(range(self._n_modes_to_connect))
        r_list = list(range(self._ro.circuit_size))
        return [x for x in r_list if x not in self._ro.heralds.keys()]

    def resolve(self):
        """
        Resolves mode mapping and checks if the mapping is consistent.

        :param mapping: can be an integer, a list or a dictionnary.
         Case int:
            Creates a dictionnary { mapping: 0, mapping+1: 1, ..., mapping+n: n }
         Case list:
            Creates a dictionnary { mapping[0]: 0, mapping[1]: 1, ..., mapping[n]: n}
         Case dict:
            keys and values can either be integers or strings. If strings, it expects port names of the same size.

        Consistency checks:
        - The input map key and value types are checked.
        - Each key/value pair size should match. For instance, 'data': [1,2,3] will fail if 'data' port length is 2
        - The constructed mapping must be the right size (= length of modes to connect of right object)
        - Resolved indexes must not be negative
        - All left output modes used in the mapping must be connectible
        - Duplicates are checked
        """
        # Handle int input case
        if isinstance(self._map, int):
            map_begin = self._map
            self._map = {}
            r_list = self._get_ordered_rmodes()
            for i in range(self._n_modes_to_connect):
                self._map[map_begin + i] = r_list[i]
            self._check_consistency()
            return self._map

        # Handle list input case
        if isinstance(self._map, list):
            map_keys = self._map
            map_values = self._get_ordered_rmodes()
            if len(map_keys) != len(map_values):
                raise InvalidMappingException(map_keys, f"input list size is expected to be {len(map_values)}")
            self._map = {k: v for k, v in zip(map_keys, map_values)}
            self._check_consistency()
            return self._map

        # Handle dict input case
        self._mapping_type_checks()
        result = {}
        for k, v in self._map.items():
            if isinstance(k, int) and isinstance(v, int):
                result[k] = v
            elif isinstance(k, str):  # Mapping between port name and index
                l_idx = self._resolve_port_left(k)
                r_idx = []
                if l_idx is None:
                    raise InvalidMappingException(self._map, f"port '{k}' was not found in processor")
                if isinstance(v, int) and len(l_idx) == 1:
                    result[l_idx[0]] = v
                elif isinstance(v, list):
                    r_idx = v
                else:  # str
                    r_idx = self._resolve_port_right(v)
                    if r_idx is None:
                        raise InvalidMappingException(self._map, f"port '{v}' was not found in processor")
                if len(l_idx) != len(r_idx):
                    raise InvalidMappingException(self._map, f"Unable to resolve '{k}: {v}' - imbalanced ports")
                for i in range(len(l_idx)):
                    result[l_idx[i]] = r_idx[i]
        self._map = result
        self._check_consistency()
        return self._map

    def _check_consistency(self):
        """
        Checks the mapping consistency
        """
        if len(self._map) != self._n_modes_to_connect:
            raise InvalidMappingException(self._map)
        min_out = min(self._map.keys())
        if min_out < 0:
            raise UnavailableModeException(min_out)
        for m_out, m_in in self._map.items():
            if not self._lp.is_mode_connectible(m_out):
                raise UnavailableModeException(m_out)
        m_in = self._map.values()
        if len(m_in) != len(list(dict.fromkeys(m_in))):  # suppress duplicates and check length
            raise InvalidMappingException(self._map)

    def _resolve_port_left(self, name: str):
        """
        Resolves mode indexes from an output port name of the left processor
        """
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
        """
        Resolves mode indexes from an input port name of the right object (which has to be a processor)
        """
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

    def add_heralded_modes(self, mapping):
        """
        Add heralded mode mapping to an existing mapping
        """
        if self._r_is_component:
            warnings.warn("Right object is not a processor, thus doesn't contain heralded modes")
            return 0
        other_herald_pos = list(self._ro.heralds.keys())
        new_mode_index = self._lp.circuit_size
        for pos in other_herald_pos:
            mapping[new_mode_index] = pos
            new_mode_index += 1
        return new_mode_index-self._lp.circuit_size

    @staticmethod
    def generate_permutation(mode_mapping: Dict[int, int]):
        """
        Generate a PERM component given an already resolved mode mapping
        Returns a tuple containing:
        - The mode range occupied by the PERM component
        - The PERM component or None if no PERM is needed (no swap in the mapping)
        """
        m_keys = list(mode_mapping.keys())
        min_m = min(m_keys)
        max_m = max(m_keys)
        missing_modes = [x for x in list(range(min_m, max_m + 1)) if x not in m_keys]
        for mm in missing_modes:
            mode_mapping[mm] = max(mode_mapping.values()) + 1
        perm_modes = list(range(min_m, min_m + len(mode_mapping)))
        perm_vect = [mode_mapping[i] for i in sorted(mode_mapping.keys())]
        if perm_vect == list(range(len(perm_modes))):
            return perm_modes, None  # No need for a permutation, modes are already sorted
        return perm_modes, PERM(perm_vect)
