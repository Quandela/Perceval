StateVector
===========

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

``StateVector`` can also be multiplied through a tensor product - and exponentiation is also built-in.

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
