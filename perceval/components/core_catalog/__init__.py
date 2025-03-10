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

from .klm_cnot import KLMCnotItem
from .postprocessed_cnot import PostProcessedCnotItem
from .heralded_cnot import HeraldedCnotItem
from .heralded_cz import HeraldedCzItem
from .generic_2mode import Generic2ModeItem
from .mzi import MZIPhaseFirst, MZIPhaseLast, SymmetricMZI
from .postprocessed_ccz import PostProcessedCCZItem
from .postprocessed_cz import PostProcessedCzItem
from .qloq_ansatz import QLOQAnsatz
from .toffoli import ToffoliItem
from .controlled_rotation_gates import PostProcessedControlledRotationsItem
from .gates_1qubit import (PauliXItem, PauliYItem, PauliZItem, HadamardItem,
                           RxItem, RyItem, RzItem, PhaseShiftItem,
                           SGateItem, SDagGateItem, TGateItem, TDagGateItem)

catalog_items = [
    # 2 qubits gate
    KLMCnotItem, HeraldedCnotItem, PostProcessedCnotItem, HeraldedCzItem, PostProcessedCzItem,
    # MZIs
    Generic2ModeItem, MZIPhaseFirst, MZIPhaseLast, SymmetricMZI,
    # 3 qubits gate
    PostProcessedCCZItem, ToffoliItem,
    # N qubits gate
    PostProcessedControlledRotationsItem,
    # 1 qubit gate
    PauliXItem, PauliYItem, PauliZItem, HadamardItem, RxItem, RyItem, RzItem,
    PhaseShiftItem, SGateItem, SDagGateItem, TGateItem, TDagGateItem,
    # User defined gates
    QLOQAnsatz
]
