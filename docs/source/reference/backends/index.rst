backends
^^^^^^^^

Backends are the lowest level of computation of perceval.
They all aim at providing ways of simulating a :ref:`Circuit` or unitary for a non-annotated :ref:`BasicState`.
As such, they are only suited to non-noisy, linear, non-polarized, non-superposed input and output states and circuits
(unless the user wants to deal with such by hand).

We distinguish two kinds of backends:

- The strong simulation backends, that can compute probability amplitudes for given output states or the whole distribution.
- The sampling backends, that can provide a random output state following the output distribution without computing it.

The strong simulation backends are all capable of computing single output probability (or amplitude),
the whole distribution as a list or a :ref:`BSDistribution`, evolve a state,
and use masks to reduce the computation space to be faster.
Note that the output of the strong simulation backends is not normalized,
so the sum of the values is not always guaranteed to be 1 for approximating backends or when there is a mask.

On the other hand, the sampling backends are only able to sample randomly output states.

A comparison of the backends is available at :ref:`Computing Backends`.

.. toctree::
   :maxdepth: 2

   naive
   naive_approx
   slos
   slap
   mps
   clifford2017
