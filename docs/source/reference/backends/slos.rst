SLOSBackend
^^^^^^^^^^^

The :code:`SLOSBackend` (for Strong Linear Optical Simulation) is a strong simulation backend that is able to compute efficiently
the entire output distribution by representing in memory a calculation path in which photons are added one by one,
with the best time complexity among strong simulation backends: :math:`\mathrm{O}(nC_n^{n+m-1})`.

The major downside of this backend is the memory intensive consumption, with the same complexity of :math:`\mathrm{O}(nC_n^{n+m-1})`.
This backend is able to use masks to reduce the computation space, making it cheaper in memory and faster.

As such, this backend is well suited with a relatively small number of photons and modes (:math:`n, m < 20`) when
it is necessary to compute everything (or at least everything that befalls into a mask).
If only a few output states are needed, other backends like :ref:`NaiveBackend` are more suited.

This backend is available in :ref:`Processor` by using the name :code:`"SLOS"`.

>>> import perceval as pcvl
>>> c = pcvl.Circuit(4) // pcvl.BS() // (2, pcvl.BS())
>>> backend = pcvl.SLOSBackend()
>>> backend.set_circuit(c)
>>> backend.set_input_state(pcvl.BasicState([1, 0, 1, 0]))
>>> print(backend.prob_distribution())
{
  |1,0,1,0>: 0.2500000000000001
  |1,0,0,1>: 0.2500000000000001
  |0,1,1,0>: 0.2500000000000001
  |0,1,0,1>: 0.2500000000000001
}

.. autoclass:: perceval.backends._slos.SLOSBackend
   :members:
   :inherited-members:
