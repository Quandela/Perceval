States
======

States hold the quantum data. Perceval introduces a formalism to represent all kinds of quantum states.

Basic State
-----------

Describes the Fock state of :math:`n` photons over :math:`m` modes where photons can be annotated. If none is annotated, then all photons in the state are indistinguishable.

A Fock state, called :code:`BasicState` in Perceval coding language, is represented by ``|n_1,n_2,...,n_m>`` notation where ``n_k`` is the number of photons in mode ``k``.
It is an alias of the :code:`exqalibur.FockState` class. See :ref:`FockState` class reference.

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
user-defined labels, and values are integers.

.. note::
   A special predefined annotation exists for polarization, `P`, which is used in *circuit with polarization operators*.
   See :ref:`Polarization`

Photons with annotations are represented individually using python dictionary notation:

``{key_1:value_1,...,key_k:value_k}``

``key_i`` will be referred to as an annotation key, and represents for example a degree of freedom labelled i ( time, polarization,...) that a photon can have,
``value_i`` is the value on this degree of freedom.
Note that a photon can have a set of annotation keys, representing different degrees of freedom, each with its own value.

.. note::

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
* ``|2{_:1},0>`` corresponding to an annotated ``|2,0>`` ``BasicState`` where the 2 photons in the first mode are annotated with the same annotation.
* ``|{a:0}{a:1},0>`` corresponding also to an annotated ``|2,0>`` ``BasicState`` where the two photons in the first mode have different annotations.

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

``StateVector`` represents a pure state. It is a (complex) linear combination of ``BasicState`` to represent state
superposition.

It is an alias of the :code:`exqalibur.StateVector` class. See :ref:`StateVector` class reference.

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
It is a dictionary were the keys are the Basic States and the values are the probability associated.
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
    ``BSDistribution``, ``SVDistribution`` and ``BSCount`` are NOT ordered data structures and must NOT be indexed with integers.

Density Matrices
----------------

Another way of representing a mixed state is using a density matrix. See: :ref:`DensityMatrix`.
