import math
import numpy as np

from .template import Backend
import quandelibc as qc


class NaiveBackend(Backend):
    """Naive algorithm, no clever calculation path, does not cache anything,
       recompute all states on the fly"""
    name = "Naive"
    supports_symbolic = False
    supports_circuit_computing = False

    def probampli_be(self, input_state, output_state, n=None, output_idx=None):
        if input_state.n != output_state.n:
            return 0
        if n is None:
            n = input_state.n
        Ust = np.empty((n, n), dtype=complex)
        colidx = 0
        p = 1
        for ok in range(self._realm):
            p *= math.factorial(output_state[ok])
        for ik in range(self._realm):
            p *= math.factorial(input_state[ik])
            for i in range(input_state[ik]):
                rowidx = 0
                for ok in range(self._realm):
                    for j in range(output_state[ok]):
                        Ust[rowidx, colidx] = self._U[ok, ik]
                        rowidx += 1
                colidx += 1
        return qc.permanent_cx(Ust, 1)/math.sqrt(p)

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        return abs(self.probampli_be(input_state, output_state, n, output_idx))**2