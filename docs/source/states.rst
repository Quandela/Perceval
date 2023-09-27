States
======

For detailed information, see the code reference of ``perceval.utils.statevector`` here : :ref:`State and StateVector`

Basic State
-----------

Describes the Fock state of :math:`n` photons over :math:`m` modes where photons can be annotated. If none is annotated, then all photons in the state are indistinguishable.

A Fock state, called BasicState in Perceval coding language, is represented by ``|n_1,n_2,...,n_m>`` notation where ``n_k`` is the number of photons in mode ``k``.

Simple code example with indistinguishable photons:

>>> bs = pcvl.BasicState("|0,1>")      # Creates a two-mode Fock state with 0 photons in first mode, and 1 photon in second mode.
>>> print(bs)                          # Prints out the created Fock state
|0,1>
>>> bs.n                               # Displays the number of photons of the created Fock state
1
>>> bs.m                               # Displays the number of modes of the created Fock state
2
>>> bs[0]                              # Displays the number of photons in the first mode of the created Fock state ( note that the counter of the number of modes    starts at 0 and ends at m-1 for an m-mode Fock state)
0
>>> print(pcvl.BasicState([0,1])*pcvl.BasicState([2,3]))  # Tensors the |0,1> and |2,3> Fock states, and prints out the result (the Fock state |0,1,2,3>)
|0,1,2,3>

Annotated Basic State
---------------------

A ``BasicState`` can also describe state of :math:`n` **annotated** photons over :math:`m` modes.

Annotation
----------

``Annotation`` distinguishes individual photons and is represented generically as a map of key :math:`\rightarrow` values - where key are
user-defined labels, and values can be string or complex numbers.

Special predefined annotations are:

* `P` for polarization used in *circuit with polarization operators* - see :ref:`Polarization`
* `t` used in *time circuit* with an integer value is defining the period from where the photon is generated (default ``0`` meaning that the photon is coming from current period).

Photons with annotations are represented individually using python dictionary notation:

``{key_1:value_1,...,key_k:value_k}``

``key_i`` will be referred to as an annotation key, and represents for example a degree of freedom labelled i ( time, polarization,...) that a photon can have,
``value_i`` is the value on this degree of freedom.
Note that a photon can have a set of annotation keys, representing different degrees of freedom, each with its own value.

.. NOTE::

  Two photons are indistinguishable if they share the same values on all their common annotation keys, or if they have no common annotation keys. For instance, for the following
  three photons,

  * :math:`p_1=` ``{a1:1}``
  * :math:`p_2=` ``{a2:1}``
  * :math:`p_3=` ``{a1:2,a2:2}``

  :math:`p_1` and :math:`p_3` are distinguishable because their a1 annotation keys have different values (1 for p_1 as opposed to 2 for p_3). :math:`p_2` and :math:`p_3` are also distinguishable because the values of their annotation key a2 do not agree. However, :math:`p_1` and :math:`p_2` are
  indistinguishable, because they share no common annotation keys.

Use of Annotation in BasicState
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Photon ``Annotations`` extend the ``BasicState`` notation as following:

``|AP_(1:1)...AP_(1:n_1),...,AP_(m:1)...AP_(m:n_m)>`` where ``AP_(k:i)`` is the representation of the ``i``-th photon in mode ``k``, ``n_i`` is the number of photons in mode ``i``.

To simplify the notation, for each mode, annotated photons with same annotations can be grouped and represented prefixed by
their count: e.g. ``2{...}``. Absence of photons is represented by ``0``, and non annotated photons are just represented by
their count.

For instance the following are representing different BasicStates with 2 photons having polarization annotations (here
limited to ``H``/``V``:

* ``|{P:H},{P:V}>`` corresponding to an annotated ``|1,1>`` ``BasicState``.
* ``|2{P:H},0>`` corresponding to an annotated ``|2,0>`` ``BasicState`` where the 2 photons in the first mode are annotated with the same annotation.
* ``|{P:H}{P:V},0>`` corresponding also to an annotated ``|2,0>`` ``BasicState`` where the two photons in the first mode have different annotations.
* ``|1,{t:-1}>`` corresponding to an annotated ``|1,1>`` ``BasicState``, the degree of freedom being time, with the photon in the second mode coming from previous period, and the photon in the first mode is not annotated.

Example code:

>>> print(pcvl.BasicState("|0,1>"))
|0,1>
>>> a_bs = pcvl.BasicState("|{P:H}{P:V},0>")  # Creates an annotated state |2,0>, with two photons in the first mode, one having a horizontal polarization, and the other a vertical polarization.
>>> print(a_bs)
|{P:H}{P:V},0>
>>> a_bs[0]  # prints the photon count in the first mode
2
>>> for annot in a_bs.get_mode_annotations(0):
>>>     print(annot)  # prints the annotation of each photon
P:H
P:V
>>> a_bs.clear_annotations()
>>> print(a_bs)  # prints the non-annotated state corresponding to a_bs
|2,0>

State Vector
------------

``StateVector`` is a (complex) linear combination of ``BasicState`` to represent state superposition.

See reference :class:`perceval.utils.StateVector` for detailed information.

``StateVector`` instances are constructed through addition and linear combination operations.

>>> st1 = pcvl.StateVector("|1,0>")   # write basic states or annotated basic states with the 'StateVector' command in order to enable creating a superposition using the '+' command
>>> st2 = pcvl.StateVector("|0,1>")
>>> st3 = st1 + st2
>> print(len(st3))
2
>>> print(st3)
1/sqrt(2)*|1,0>+1/sqrt(2)*|0,1>
>>> st3[0]    # outputs the first state in the superposition state st3
|1,0>
>>> st3[1]     # outputs the second state in the superposition st3
|0,1>
>>> st4 = alpha*st1 + beta*st2

.. WARNING::
  ``StateVector`` will normalize themselves at usage so normalization terms will be added to any combination.

``StateVector`` can also be multiplied through a tensor product - and exponentation is also built-in.

>>> import perceval as pcvl

>>> sv0 = pcvl.StateVector([1,0]) + pcvl.StateVector([0,1])
>>> sv1 = ...
>>> bs = pcvl.BasicState([0])

>>> new_state = pcvl.tensorproduct([sv0, sv1, bs])
>>> # or:
>>> # new_state = sv0 * sv1 * bs

>>> new_state = sv0 ** 3 # equivalent to sv0 * sv0 * sv0

Sampling
^^^^^^^^

:meth:`perceval.utils.StateVector.sample` and :meth:`perceval.utils.StateVector.samples` methods are used to generate samples from state vectors:

>>> st = pcvl.StateVector([0,1]) + pcvl.StateVector([1,0])
>>> c = Counter()
>>> for s in st.samples(10):
>>>    c[s] += 1
>>> print("\n".join(["%s: %d" % (str(k), v) for k,v in c.items()]))
|0,1>: 3
|1,0>: 7

.. note:: These methods do not modify the state vector



Measurement
^^^^^^^^^^^

:meth:`perceval.utils.StateVector.measure` is used to perform a measure on one or multiple modes. It returns for each
possible fock state value of the selected modes, its probability and the collapsed state vector on the remaining modes.

>>> sv = pcvl.StateVector("|0,1,1>")+pcvl.StateVector("|1,1,0>")
>>> map_measure_sv = sv.measure(1)
>>> for s, (p, sv) in map_measure_sv.items():
>>>    print(s, p, sv)
|1> 0.9999999999999998 sqrt(2)/2*|0,1>+sqrt(2)/2*|1,0>

Basic State Samples
-------------------

The class ``BSSamples`` is a container that collects sampled Basic States.
It is the object generated by the method ``perceval.algorithm.sampler.sample()`` when using a Processor.

Basic State Count
-----------------
The class ``BSCount`` is also a container but it only counts the Basic states without keeping in track their order.
The method ``sample_count()`` return this data type.

Basic State Distribution
------------------------

The class ``BSDistribution`` represent a probability distribution of Basic States.
It is a dictionnary were the keys are the Basic States and the values are the probability associated.
It is the type of object given by the method ``probs`` of the ``Processor`` class.

State Vector Distribution
-------------------------

``SVDistribution`` is a recipe for constructing a mixed state using ``BasicState`` and/or
``StateVector`` commands (which themselves produce pure states).

For example, The following ``SVDistribution``

+-------------------------------------+------------------+
| ``state``                           | ``probability``  |
+=====================================+==================+
| ``|0,1>``                           |     ``1/2``      |
+-------------------------------------+------------------+
| ``1/sqrt(2)*|1,0>+1/sqrt(2)*|0,1>`` |     ``1/4``      |
+-------------------------------------+------------------+
| ``|1,0>``                           |     ``1/4``      |
+-------------------------------------+------------------+

results in the mixed state ``1/2|0,1><0,1|+1/4(1/sqrt(2)*|1,0>+1/sqrt(2)*|0,1>)(1/sqrt(2)*<1,0|+1/sqrt(2)*<0,1|)+1/4|1,0><1,0|``

.. WARNING::
    The ``BSDistribution``, ``SVDistribution`` and ``BSCount`` classes inherits from dictionnaries. Thus they are not ordered data structures and must NOT be indexed with integers.
