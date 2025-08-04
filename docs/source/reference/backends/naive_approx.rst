NaiveApproxBackend
^^^^^^^^^^^^^^^^^^

Like the :ref:`NaiveBackend`, the :code:`NaiveApproxBackend` is a strong simulation backend that can compute a
single output probability amplitude at a time, by computing the permanent of a :math:`n \times n` matrix.
However, instead of computing exactly the permanent, the :code:`NaiveApproxBackend` uses the Gurvit estimate algorithm
to approximate it with a 99% confidence interval.

It is very efficient to compute very precise output states, but not to compute the whole distribution, and it can be used
with more modes and photons than the :ref:`NaiveBackend` at the cost of losing precision on the result.

Thus, using it is not recommended in :ref:`Simulator` (except when using :meth:`probability()`) or :ref:`Processor`,
but it is well suited for applications where only a few output probabilities matter with many photons.
If the whole or most of the computational space is needed, other backends like :ref:`SLOSBackend` are more suited.

This backend is available in :ref:`Processor` by using the name :code:`"NaiveApprox"`.

Unlike other backends, this backend needs a number of iterations to use to estimate the permanent.
Also, in addition to the generic backend methods, this backend offers means to get a 99% confidence interval on the probability
or a 99% sure error bound on the amplitude.

>>> import perceval as pcvl
>>> circuit_size = 60
>>> n_photons = 30
>>> backend = pcvl.NaiveApproxBackend(100_000_000)  # Number of iterations; higher values reduce the error bound
>>> backend.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(circuit_size)))
>>> input_state = pcvl.BasicState([1]*n_photons + [0]*(circuit_size-n_photons))
>>> backend.set_input_state(input_state)
>>> interval = backend.probability_confidence_interval(BasicState([1]*n_photons + [0]*(circuit_size-n_photons)))
>>> print(f"Probability in {interval}")
Probability in [6.051670221391749e-20, 1.5297683283662674e-19]

.. autoclass:: perceval.backends._naive_approx.NaiveApproxBackend
   :members:
   :inherited-members:
