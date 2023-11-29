State and StateVector
=====================

StateVector class reference
---------------------------

:code:`StateVector` is an important data structure class written in the native package :code:`exqalibur`. A StateVector is a
superposed state represented as a (complex) linear combination of :code:`BasicState` objects (its components), the
complex coefficients being probability amplitudes.

* **Constructor** :code:`__init__(bs: BasicState or List[int] or str = None)`

Initialize a StateVector from a BasicState or data to create a BasicState (list of integers, string reprsentation)

>>> empty_sv = StateVector()  # creates an empty state vector
>>> bs = BasicState("|1,0,1,0>")
>>> sv1 = StateVector(bs)  # creates a state vector containing only |1,0,1,0> with amplitude 1
>>> sv2 = StateVector([1, 0, 1, 0])  # same
>>> sv3 = StateVector("|1,0,1,0>")  # same
>>> assert sv1 == sv2 and sv1 == sv3

* **Property** :code:`n`

List the possible values of n (number of photons) in the different components of the state vector

>>> sv = StateVector("|1,0,1,0>") + StateVector("|1,1,1,0>") + StateVector("|1,1,1,1>")
>>> print(sv.n)
{2, 3, 4}

* **Property** :code:`m`

Return the mode count in the state vector

>>> sv = StateVector("|1,0>")
>>> sv.m
2

* **Method** :code:`normalize()`

Normalize the state vector: amplitudes are normalized to follow the rule
:code:`sum(abs(probability_amplitudes)**2) == 1` and components with an amplitude near 0 are erased.

* :code:`__str__(nsimplify: bool = True)`

Stringifies the :code:`StateVector`, trying to simplify numerical representations of probability amplitude when
:code:`nsimplify` is :code:`True`. The string representation is normalized but the :code:`StateVector` is left untouched.

>>> sv = StateVector("|1,0>") - StateVector("|0,1>")
>>> print(sv)  # calls __str__ with default parameters
sqrt(2)/2*|1,0>-sqrt(2)/2*|0,1>
>>> print(sv.__str__(nsimplify=False))
(0.7071067811865475+0j)*|1,0>+(-0.7071067811865475-0j)*|0,1>

* **Arithmetic operators**

A :code:`StateVector` can be built using arithmetic. While only applying arithmetic operations to a state vector, no
automatic normalization is called, allowing the composition of state vectors through multiple Python statements.

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

* **Accessors and iterators**

After building a :code:`StateVector` using arithmetic operations, there are different ways to retrieve
components and amplitudes.

>>> bs01 = BasicState("|0,1>")
>>> bs10 = BasicState("|1,0>")
>>> sv = bs10 - bs01
>>> sv.normalize()
>>> assert bs10 in sv  # Ensure sv contains bs10 as a component
>>> print(sv[bs10])  # An amplitude can be retrieved by accessing the StateVector component
(0.7071067811865475+0j)
>>> for i in range(len(sv)):  # Indexation. WARNING - the component order is not fixed (commutativity)
>>>     print(sv[i], sv[sv[i]])
|1,0> (0.7071067811865475+0j)
|0,1> (-0.7071067811865475-0j)
>>> for component, amplitude in sv:  # Iteration on the StateVector
>>>     print(component, amplitude)
|1,0> (0.7071067811865475+0j)
|0,1> (-0.7071067811865475-0j)
>>> print(sv.keys())  # Components may also be retrieved as a list
[|1,0>, |0,1>]

* **Sampling methods**

:code:`BasicState` components can be sampled from a :code:`StateVector` in regard of the probability amplitudes.

>>> sv = math.sqrt(0.75)*StateVector("|1,0>") + math.sqrt(0.25)*StateVector("|2,2>")
>>> print(sv.sample())
|1,0>
>>> print(sv.samples(10))
[|1,0>, |1,0>, |1,0>, |1,0>, |1,0>, |1,0>, |2,2>, |1,0>, |1,0>, |2,2>]

* **Method** :code:`measure(modes: List[int])`

Perform a measure on one or multiple modes and collapse the remaining :code:`StateVector`. The resulting
states are not normalised by default.

Return a Python dictionary where keys are the possible measures (as :code:`BasicState`) and values are tuples containing
(probability, :code:`StateVector`).

>>> sv = StateVector("|0,1>") + StateVector("|1,0>")
>>> print(sv.measure([0]))
{|0>: (0.5, |1>), |1>: (0.5, |0>)}

The rest of the module
----------------------

.. automodule:: perceval.utils.statevector
   :members:
