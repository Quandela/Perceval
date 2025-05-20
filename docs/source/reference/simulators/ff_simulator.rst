FFSimulator
===========

The :code:`FFSimulator` is a simulator dedicated to simulate feed-forward experiments.

Like the :code:`Simulator`, it needs a strong simulation backend to be able to perform simulations.
However, the :code:`FFSimulator` is also able to compute circuits having :ref:`FFConfigurator` or
:ref:`FFCircuitProvider` but is unable to compute probability amplitudes.

Thus, only the :code:`probs_svd` and :code:`probs` computation methods are available.

>>> import perceval as pcvl
>>> sim = pcvl.FFSimulator(pcvl.SLOSBackend())
>>> ff_not = pcvl.FFCircuitProvider(2, 0, pcvl.Circuit(2)).add_configuration([0, 1], pcvl.PERM([1, 0]))
>>> sim.set_circuit([((0, 1), ff_not)])  # Since non-unitary components can't be added to Circuit, we directly provide a list; the number of modes is implicit
>>> sim.probs(pcvl.BasicState([0, 1, 1, 0]))
{
  |0,1,0,1>: 1.0
}

.. autoclass:: perceval.simulators.feed_forward_simulator.FFSimulator
  :members:
  :inherited-members:
