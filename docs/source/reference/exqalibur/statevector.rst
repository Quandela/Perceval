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
0.577*|1>+0.577*|2>+0.577*|3>

:code:`StateVector` can be built with great freedom:

>>> import math
>>> from exqalibur import FockState, StateVector
>>> sv = 0.5j * FockState([1, 1]) - math.sqrt(2) * StateVector("|2,0>") + StateVector([0, 2]) * 0.45
>>> print(sv)
0.319I*|1,1>-0.903*|2,0>+0.287*|0,2>

.. warning::
  When multiplying a state by a numpy scalar (such as one returned by a numpy function), numpy takes precedence over
  the state arithmetics and tries to convert the state to a numpy array. This results in an exception with potentially
  obscure message. Two solutions exist: putting the numpy number on the right of the :code:`*` operand, or converting the numpy
  scalar to a python type using the :code:`.item()` method.

* **Comparison operators**

Comparing two :code:`StateVector` with operator :code:`==` or :code:`!=` compare normalised copies of each. probability
amplitudes are compared strictly (they have to be exactly the same to be considered equal).

.. note::
  ``StateVector`` will normalize themselves only at usage (iteration, sampling, measurement), and not during state
  arithmetics operations.

``StateVector`` can also be multiplied with a tensor product:

>>> import exqalibur as xq
>>> sv0 = xq.StateVector([1,0]) + xq.StateVector([0,1])
>>> sv1 = 1j*xq.StateVector([2]) - xq.StateVector([0])
>>> bs = xq.FockState([0])
>>> print(sv0 * sv1 * bs)
0.5I*|0,1,2,0>-0.5*|0,1,0,0>+0.5I*|1,0,2,0>-0.5*|1,0,0,0>

Exponentiation is also built-in:

>>> print(sv1 ** 3) # equivalent to sv1 * sv1 * sv1
-0.354I*|2,2,2>+0.354I*|2,0,0>+0.354*|2,2,0>+0.354*|2,0,2>+0.354I*|0,2,0>+0.354*|0,2,2>-0.354*|0,0,0>+0.354I*|0,0,2>

StateVector code reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: exqalibur.StateVector
   :members:

SVDistribution
==============

.. autoclass:: exqalibur.SVDistribution
   :members:
