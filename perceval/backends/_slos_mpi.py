# MIT License
#
# Copyright (c) 2026 Quandela
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

import exqalibur as xq

from perceval.components import ACircuit
from perceval.utils import BSDistribution, FockState, StateVector
from perceval.utils.postselect import PostSelect

from ._abstract_backends import AStrongSimulationBackend

try:
    # Importing mpi4py initializes MPI before the native backend is constructed.
    from mpi4py import MPI
except ImportError:  # pragma: no cover - depends on the optional MPI environment
    MPI = None


class SLOSMPIBackend(AStrongSimulationBackend):
    """Rank-local wrapper around Exqalibur's distributed SLOS backend.

    Result-producing methods are collective and must be called in the same
    order on every MPI rank. Exqalibur keeps the calculation distributed, then
    gathers complete Perceval results on rank 0. Other ranks return empty
    containers after participating in the same collective calls.
    """

    def __init__(self, mask=None):
        super().__init__()
        if MPI is None:
            raise RuntimeError("SLOS_MPI requires mpi4py")
        if not hasattr(xq, "SLOS_MPI"):
            raise RuntimeError("Exqalibur was built without MPI support")
        self._slos = xq.SLOS_MPI()
        if mask:
            self.set_mask(mask)

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)
        self._slos.set_unitary(self._umat)

    def set_input_state(self, input_state: FockState):
        super().set_input_state(input_state)
        self._slos.set_input_state(input_state)

    def _init_mask(self):
        super()._init_mask()
        self._slos.set_mask(self._mask)

    def set_post_select(self, post_selection: PostSelect):
        self._slos.set_post_select(post_selection)

    def _local_amplitudes(self) -> dict[FockState, complex]:
        return dict(zip(self._slos.get_states(), self._slos.all_amplitudes()))

    def prob_amplitude(self, output_state: FockState) -> complex:
        return self._local_amplitudes().get(output_state, 0j)

    def probability(self, output_state: FockState) -> float:
        return abs(self.prob_amplitude(output_state)) ** 2

    def prob_distribution(self) -> BSDistribution:
        result = BSDistribution()
        # Exqalibur gathers contiguous C++ buffers with MPI_Gatherv.  Only
        # rank 0 materializes the complete distribution; other ranks return
        # an empty result while still participating in the collective call.
        for state, probability in self._slos.gather_distribution(root=0).items():
            result.add(state, probability)
        return result

    def all_prob_ampli(self) -> list[complex]:
        return self._slos.all_amplitudes()

    def all_prob(self, input_state: FockState = None) -> list[float]:
        self._slos.set_input_state(input_state or self._input_state)
        return self._slos.gather_probabilities(root=0)

    def evolve(self) -> StateVector:
        self._slos.set_input_state(self._input_state)
        result = StateVector()
        states = self._slos.gather_states(root=0)
        amplitudes = self._slos.gather_amplitudes(root=0)
        for output_state, amplitude in zip(states, amplitudes):
            result += output_state * amplitude
        return result

    @property
    def name(self) -> str:
        return "SLOS_MPI"

    @property
    def rank(self) -> int:
        return self._slos.get_rank()

    @property
    def process_count(self) -> int:
        return self._slos.get_process_count()
