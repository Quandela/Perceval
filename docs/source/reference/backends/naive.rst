NaiveBackend
^^^^^^^^^^^^

The :code:`NaiveBackend` is a strong simulation backend that can compute a single output probability amplitude at a time,
by computing the permanent of a :math:`n \times n` matrix, with a time complexity of :math:`\mathrm{O}(n2^n)` (see
:cite:t:`ryser1963combinatorial` and :cite:t:`glynn2010permanent`).

As such, it is very efficient to compute very precise output states, but not to compute the whole distribution.

Thus, using it is not recommended in :ref:`Simulator` (except when using :meth:`probability()`) or :ref:`Processor`,
but it is well suited for applications where only a few output probabilities matter.
If the whole or most of the computational space is needed, other backends like :ref:`SLOSBackend` are more suited.

This backend is available in :ref:`Processor` by using the name :code:`"Naive"`.

>>> import perceval as pcvl
>>> c = pcvl.Circuit(4) // pcvl.BS() // (2, pcvl.BS())
>>> backend = pcvl.NaiveBackend()
>>> backend.set_circuit(c)
>>> backend.set_input_state(pcvl.BasicState([1, 0, 1, 0]))
>>> print(backend.prob_amplitude(pcvl.BasicState([1, 0, 0, 1])))
0.5j

.. autoclass:: perceval.backends._naive.NaiveBackend
   :members:
   :inherited-members:
