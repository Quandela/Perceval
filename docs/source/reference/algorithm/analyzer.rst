Analyzer
^^^^^^^^

The ``Analyzer`` algorithm aims at testing a processor, computing a probability table between input states and expected
outputs, a performance score and an error rate.

For example, we call the Naive backend that we store in simulator_backend:

>>> simulator_backend = pcvl.BackendFactory().get_backend('Naive')

We can create an input state that will enter our optical scheme later on. We store it in `input_state` and use `BasicState`
from the Perceval library.

>>> input_state = pcvl.BasicState("|1,1>")

let's simulate the distribution obtained when we input two photons in a beam-splitter. We will use the Naive backend already stored in simulator_backend.

We will simulate the behavior of the circuit using the `Circuit Analyzer` which has three arguments:

- The first one is an instance of a processor containing the circuit to analyse.
- The second one is the input state (we will use `input_state`).
- The third one is the desired output states. To compute all possible output states, one just input `"*"`.

>>> p = Processor("SLOS", comp.BS()) # create a processor running on SLOS backend
>>> ca = pcvl.algorithm.Analyzer(p, [input_state], "*")

Then, we display the result of `Analyzer` via :ref:`pdisplay`.

>>> pcvl.pdisplay(ca)

.. figure:: ../../_static/img/CircuitAnalyzerHOM.png
  :align: center
  :width: 40%

.. autoclass:: perceval.algorithm.analyzer.Analyzer
   :members:
