States
======

Basic State
-----------

Describes the Fock state of  :math:`n` indistinguishable photons over  :math:`m` modes.

See reference :class:`perceval.utils.BasicState` for detailed information.

A Fock state, called BasicState in Perceval coding language,  is represented by ``|n_1,n_2,...,n_m>`` notation where ``n_k`` is the number of photons in mode ``k``.

Example code:

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

``AnnotatedBasicState`` extends ``BasicState`` and describes state of :math:`n` **annotated** photons over :math:`m` modes.

See reference :class:`perceval.utils.AnnotatedBasicState` for detailed information.

Annotation
^^^^^^^^^^

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

Use of Annotation in AnnotatedBasicState
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``AnnotatedBasicState`` notation extends the ``BasicState`` notation as following:

``|AP_(1:1)...AP_(1:n_1),...,AP_(m:1)...AP_(m:n_m)>`` where ``AP_(k:i)`` is the representation of the ``i``-th photon in mode ``k``, ``n_i`` is the number of photons in mode ``i``.

To simplify the notation, for each mode, annotated photons with same annotations can be grouped and represented prefixed by
their count: e.g. ``2{...}``. Absence of photons is represented by ``0``, and non annotated photons are just represented by
their count as for ``BasicState``.

For instance the following are representing different BasicStates with 2 photons having polarization annotations (here
limited to ``H``/``V``:

* ``|{P:H},{P:V}>`` corresponding to an annotated ``|1,1>`` ``BasicState``.
* ``|2{P:H},0>`` corresponding to an annotated ``|2,0>`` ``BasicState`` where the 2 photons in the first mode are annotated with the same annotation.
* ``|{P:H}{P:V},0>`` corresponding also to an annotated``|2,0>`` ``BasicState`` where the two photons in the first mode have different annotations.
* ``|1,{t:-1}>`` corresponding to an annotated ``|1,1>`` ``BasicState``, the degree of freedom being time,   with the photon in mode 2 coming from previous period, and the photon in mode 1 is not annotated.

Example code:

>>> print(pcvl.AnnotatedBasicState("|0,1>"))
|0,1>
>>> a_bs = pcvl.AnnotatedBasicState("|{P:H}{P:V},0>")   # Creates an annotated state |2,0> , with two photons in the first mode, one having a horizontal polarization, and the other a vertical polarization.
>>> print(a_bs)
|{P:H}{P:V},0>
>>> a_bs[0]                      # prints the photons in the first mode
({"P":"H"},{"P":"V"})
>>> print(a_bs.clear())     #prints the non-annotated Basic state corresponding to a_bs
|2,0>

State Vector
------------

``StateVector`` extends ``AnnotatedBasicState`` to represents state superpositions.

See reference :class:`perceval.utils.StateVector` for detailed information.

``StateVector`` instances are constructed through addition and linear combination operations.

>>> st1 = pcvl.StateVector("|1,0>")   # write basic states or annotated basic states with the 'StateVector' command in order to enable creating a superposition using the '+' command
>>> st2 = pcvl.StateVector("|0,1>")
>>> st3= st1 + st2
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
  ``StateVector`` will normalize themselves so normalization terms will be added to any combination.

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

.. INFO::
  These methods do not modify the state vector

Measurement
^^^^^^^^^^^

:meth:`perceval.utils.StateVector.measure` is used to perform a measure on one or multiple modes. It returns for each
possible fock state value of the selected modes, its probability and the collapsed state vector on the remaining modes.

>>> sv = pcvl.StateVector("|0,1,1>")+pcvl.StateVector("|1,1,0>")
>>> map_measure_sv = sv.measure(1)
>>> for s, (p, sv) in map_measure_sv.items():
>>>    print(s, p, sv)
|1> 0.9999999999999998 sqrt(2)/2*|0,1>+sqrt(2)/2*|1,0>

State Vector Distribution
-------------------------

``SVDistribution`` is a recipe for constructing a mixed state using BasicState and/or
``StateVector`` commands (which themselves produce pure states).

For example, The following ``SVDistribution``

+-------------------------------------+------------------+
|      ``state``                      | ``probability``  |
+=====================================+==================+
| ``|0,1>``                           |     ``1/2``      |
+-------------------------------------+------------------+
| ``1/sqrt(2)*|1,0>+1/sqrt(2)*|0,1>`` |     ``1/4``      |
+-------------------------------------+------------------+
| ``|1,0>``                           |     ``1/4``      |
+-------------------------------------+------------------+

results in the mixed state ``1/2|0,1><0,1|+1/4(1/sqrt(2)*|1,0>+1/sqrt(2)*|0,1>)(1/sqrt(2)*<1,0|+1/sqrt(2)*<0,1|)+1/4|1,0><1,0|``

TimeSVDistribution
------------------

``TimedSVDistribution`` is representing a time sequence distribution of ``StateVector``.
