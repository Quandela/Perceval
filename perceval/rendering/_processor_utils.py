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

from perceval.components import Circuit, PERM, Processor


def _flatten(composite, starting_mode=0):
    component_list = []
    for m_range, comp in composite._components:
        if isinstance(comp, Circuit):
            sub_list = _flatten(comp, starting_mode=m_range[0])
            component_list += sub_list
        else:
            m_range = [m+starting_mode for m in m_range]
            component_list.append((m_range, comp))
    return component_list


def precompute_herald_pos(processor: Processor):
    component_list = _flatten(processor)
    component_list.reverse()  # iterate in reverse (right to left)
    result = {}
    for k in processor.heralds.keys():
        initial_y = k
        y = k
        # search the first component on which the herald is plugged
        for m_range, cp in component_list:
            m0 = m_range[0]
            if y not in m_range:
                continue
            if isinstance(cp, PERM):
                y = cp.perm_vector.index(y-m0) + m0
            else:
                result[initial_y] = (y-m0, cp)
                break
    return result
