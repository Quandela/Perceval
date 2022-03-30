import copy
import sys
from .template import Backend

sf = None


class SFBackend(Backend):
    """Strawberry field proxy for benchmarking
    """

    name = "SF"
    supports_symbolic = False
    supports_circuit_computing = False

    def compile(self, input_state):
        # load only on demands so that program can run with dependencies
        global sf
        if 'strawberryfields' not in sys.modules:
            self._logger.info("load strawberryfields libraries")
            import strawberryfields as sf
            self._logger.info("complete loading strawberryfields libraries")

        if self._compiled_input == input_state:
            return
        self._compiled_input = copy.copy(input_state)
        self._logger.info("compiling strawberryfields program")
        self._boson_sampling = sf.Program(self._m)
        self._eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 7})

        with self._boson_sampling.context as q:
            for idx, v in enumerate(input_state):
                if v:
                    sf.ops.Fock(1) | q[v]
                else:
                    sf.ops.Vac | q[v]

            sf.ops.Interferometer(self._U) | q

        self._logger.info("running simulator program")
        self._results = self._eng.run(self._boson_sampling)
        self._logger.info("getting results")
        self._probs = self._results.state.all_fock_probs()
        self._logger.info("complete compilation")

    def prob_be(self, input_state, output_state, n=None, output_idx=None):
        p = self._probs
        # there might a more pythonic way to unpack output state as array index, but I can't find it
        for k in output_state:
            p = p[k]
        return p
