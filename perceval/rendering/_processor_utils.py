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

from perceval.components import PERM, AProcessor


class ComponentHeraldInfo:
    """
    Store, for a component, indices of the heralds attached to its inputs or
    outputs
    """
    def __init__(self):
        self.input_heralds = {}
        self.output_heralds = {}

    def register_herald(self, forward_pass, mode_index, herald_mode):
        if forward_pass:
            self.input_heralds[mode_index] = herald_mode
        else:
            self.output_heralds[mode_index] = herald_mode


def collect_herald_info(processor: AProcessor, recursive: bool):
    """
    Return a dictionary mapping a component to a HeraldInfo object.

    The HeraldInfo object stores information about which inputs and outputs
    are connected to herald modes.

    For example, if d[my_component].output_heralds[0] == 5,
    then output mode 0 of the component is the herald at final mode 5.

    For correct rendering, the information should be collected at the same
    level of granularity as the drawing: for individual circuit elements when
    recursive is True, for blocks when recursive is False.
    """
    if recursive:
        component_list = processor.flatten()
    else:
        component_list = processor._components

    herald_info = {}
    for herald_mode in processor.heralds.keys():
        # Do one forward pass to identify heralds on inputs, and one
        # backward pass to identify heralds "plugged" on outputs
        for forward_pass in [True, False]:
            c_list = component_list if forward_pass else component_list[::-1]
            mode = herald_mode
            for mode_range, component in c_list:
                m0 = mode_range[0]
                if mode not in mode_range:
                    continue
                if isinstance(component, PERM):
                    # Heralds can be moved across permutations
                    if forward_pass:
                        mode = component.perm_vector[mode - m0] + m0
                    else:
                        mode = component.perm_vector.index(mode - m0) + m0
                else:
                    h_info = herald_info.setdefault(
                        component, ComponentHeraldInfo())
                    h_info.register_herald(forward_pass, mode - m0, herald_mode)
                    break
    return herald_info
