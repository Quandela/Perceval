distinct permutations
^^^^^^^^^^^^^^^^^^^^^

Perceval has a built-in method to construct efficiently the distinct permutations from an iterable.
This method is comparable to the method from the more-itertools python package.

>>> import perceval as pcvl
>>> from perceval.utils.qmath import distinct_permutations
>>> print([pcvl.BasicState(perm) for perm in distinct_permutations([1, 1, 0])])  # Generate all states with n = 2 and at most 1 photon per mode
[|0,1,1>, |1,0,1>, |1,1,0>]

.. autofunction:: perceval.utils.qmath.distinct_permutations
