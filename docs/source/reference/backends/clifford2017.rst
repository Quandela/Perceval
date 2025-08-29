Clifford2017Backend
^^^^^^^^^^^^^^^^^^^

The :code:`Clifford2017Backend` is a sampling backend that is able to compute random output states according to
the exact output distribution without computing it, by computing sub-permanents of chosen matrices and computing
sampling weights from them, adding the photons one by one. The algorithm is introduced in
:cite:t:`clifford_classical_2018`.

This backend has the advantage of being able to handle more modes and photons than the strong simulation backends,
and does not need to represent the whole output space, so it is much more memory efficient, at the cost of only being
able to approximate the resulting distribution.

This backend is available in :ref:`Processor` by using the name :code:`"CliffordClifford2017"`.

>>> import perceval as pcvl
>>> c = pcvl.BS()
>>> backend = pcvl.Clifford2017Backend()
>>> backend.set_circuit(c)
>>> backend.set_input_state(pcvl.BasicState([1, 0]))
>>> print(backend.samples(10))  # Results may vary
[ |1,0>, |0,1>, |0,1>, |0,1>, |1,0>, |0,1>, |1,0>, |0,1>, |0,1>, |0,1> ]

.. autoclass:: perceval.backends._clifford2017.Clifford2017Backend
   :members:
   :inherited-members:
