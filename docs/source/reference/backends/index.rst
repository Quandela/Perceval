backends
^^^^^^^^

Backends are the lowest level of computation of perceval.
They all aim at providing ways of simulating a :ref:`Circuit` or unitary for a non-annotated :ref:`BasicState`.
As such, they are only suited to non-noisy, linear, non-polarized, non-superposed input states and circuits
(unless the user wants to deal with such by hand).

We distinguish two kinds of backends:

- The strong simulation backends, that can compute probability amplitudes for given output states or the whole distribution.
- The sampling backends, that can provide a random output state following the output distribution without computing it.

.. toctree::
   :maxdepth: 2

   naive
