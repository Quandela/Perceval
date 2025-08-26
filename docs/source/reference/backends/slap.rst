SLAPBackend
^^^^^^^^^^^

The :code:`SLAPBackend` (for Simulator of LAttice of Polynomials) is a strong simulation backend that,
like the :code:`SLOSBackend`, is able to compute efficiently all the output probabilities at once. It achieves its goal
by computing partial derivatives of a polynomial along a graph. The algorithm is introduced in :cite:t:`goubault2025`.

The main advantage compared to SLOS is that this graph is not represented in memory,
so this backend is more memory efficient, with the downside of being slower when memory is not a limitation
(for instance with relatively few photons :math:`n < 10`).
If only a few output states are needed, other backends like :ref:`NaiveBackend` are more suited.

This backend is available in :ref:`Processor` by using the name :code:`"SLAP"`.

>>> import perceval as pcvl
>>> c = pcvl.Unitary(pcvl.Matrix.random_unitary(4))
>>> backend = pcvl.SLAPBackend()
>>> backend.set_circuit(c)
>>> backend.set_input_state(pcvl.BasicState([1, 0, 1, 0]))
>>> print(backend.all_prob())  # Results are random due to random unitary
[0.22615963112684112, 0.059932460984674245, 0.11409780074515555, 0.05869159993147518, 0.06610964358209905, 0.1384083292588432, 0.1266841040718083, 0.08819140959446393, 0.05777776134512867, 0.0639472593595116]


.. autoclass:: perceval.backends._slap.SLAPBackend
   :members:
   :inherited-members:
