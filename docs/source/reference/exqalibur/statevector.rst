StateVector
===========

The ``StateVector`` class is the mean to create superposed pure states in the Fock space.

State vector arithmetics
^^^^^^^^^^^^^^^^^^^^^^^^

A :code:`StateVector` can be built using arithmetic. While only applying arithmetic operations to a state vector, no
automatic normalization is called, allowing the composition of state vectors through multiple Python statements.

>>> from exqalibur import StateVector
>>> sv = StateVector("|1>") + StateVector("|2>")
>>> sv += StateVector("|3>")
>>> print(sv)  # All components of sv have the same amplitude
sqrt(3)/3*|1>+sqrt(3)/3*|2>+sqrt(3)/3*|3>

:code:`StateVector` can be built with great freedom:

>>> sv = 0.5j * BasicState([1, 1]) - math.sqrt(2) * StateVector("|2,0>") + StateVector([0, 2]) * 0.45
>>> print(sv)
0.319275*I*|1,1>-0.903047*|2,0>+0.287348*|0,2>

* **Comparison operators**

Comparing two :code:`StateVector` with operator :code:`==` or :code:`!=` normalize them then compare that each
component and each probability amplitude are exactly the same.

.. note::
  ``StateVector`` will normalize themselves only at usage, and not during state arithmetics operations.

``StateVector`` can also be multiplied through a tensor product - and exponentiation is also built-in.

>>> import perceval as pcvl

>>> sv0 = pcvl.StateVector([1,0]) + pcvl.StateVector([0,1])
>>> sv1 = ...
>>> bs = pcvl.BasicState([0])

>>> new_state = pcvl.tensorproduct([sv0, sv1, bs])
>>> # or:
>>> # new_state = sv0 * sv1 * bs

>>> new_state = sv0 ** 3 # equivalent to sv0 * sv0 * sv0

StateVector code reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: exqalibur.StateVector
   :members:

SVDistribution
==============

.. autoclass:: exqalibur.SVDistribution
   :members:
