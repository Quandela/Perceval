simplify
========

Circuit simplification
^^^^^^^^^^^^^^^^^^^^^^

Several strategies to simplify a circuit exist. Perceval circuit simplification takes a circuit and does the following:

* For phase shifters, add their phase if they are not parameters and combine them into a single phase shifter (work
  through permutations). If :code:`display == False`, removes them if their added phase is :math:`0` or :math:`2\pi`.
* For Permutations, if two permutations are consecutive, they are combined into a single permutation. For single
  permutations, fixed modes at the extremities are removed. If they are not just consecutive, try to compute a "better"
  permutation, then if it is better, move the components accordingly to this new permutation. Display changes how a
  permutation is evaluated.

Example:

>>> from perceval.utils.algorithms.simplification import simplify
>>> from perceval import Circuit, PERM, PS
>>> circuit = Circuit(6) // PERM([3,2,1,0]) // (1, PERM([4,1,3,2,0])) // PS(phi=0.6) // PS(phi=0.2)
>>> print(circuit.ncomponents())
4
>>> simplified_circuit = simplify(circuit, display = False)
>>> print(simplified_circuit.ncomponents())
2

simplify code reference
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: perceval.utils.algorithms.simplification.simplify
