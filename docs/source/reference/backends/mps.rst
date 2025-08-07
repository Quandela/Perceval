MPSBackend
^^^^^^^^^^

The :code:`MPSBackend` is a strong simulation backend that can compute the whole output distribution at once by
using higher dimensional matrices and evolving them through the components of a circuit.

It is efficient to compute the whole output distribution of a circuit.
However, this backend has several major downsides:

- It is not able to process components spanning more than two modes (except :ref:`Permutation` components).
- It needs a cutoff number that avoids growing the matrices exponentially but leads to approximate results.
- It's time complexity scales in the number of components, and not only the number of modes and photons.
- Although masks work as expected, this backend is not able to make profit of the reduced computation space.

Thus, using it is recommended only for small circuits with small components where the whole distribution is needed but
the exact value of the probabilities is not needed.
In any other case, other backends like :ref:`SLOSBackend` or :ref:`NaiveBackend` are more suited.

This backend is available in :ref:`Processor` by using the name :code:`"MPS"`.

Unlike other backends, this backend needs a cutoff number that will induce imprecision on the results.
Higher values give more accurate results at the cost of a heavier computation.
In principle, a high enough value will give exact results, but the computation will be heavier than with a :ref:`SLOSBackend` for example.

>>> import perceval as pcvl
>>> c = pcvl.Circuit(4) // pcvl.BS() // (2, pcvl.BS()) // (1, pcvl.BS())
>>> backend = pcvl.MPSBackend(cutoff=3)
>>> backend.set_circuit(c)
>>> backend.set_input_state(pcvl.BasicState([1, 0, 1, 0]))
>>> print(backend.prob_distribution())
{
  |1,1,0,0>: 0.12500000000000003
  |1,0,1,0>: 0.07775105849101832
  |1,0,0,1>: 0.29017090063073997
  |0,2,0,0>: 0.12500000000000003
  |0,1,1,0>: 0.020833333333333356
  |0,1,0,1>: 0.07775105849101838
  |0,0,2,0>: 0.12500000000000006
  |0,0,1,1>: 0.12500000000000003
}

.. autoclass:: perceval.backends._mps.MPSBackend
   :members:
   :inherited-members:
