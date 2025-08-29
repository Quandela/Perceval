Parameter
=========

This class holds parameter values that can be used in circuits and in backends supporting symbolic computation.

>>> import perceval as pcvl
>>> p = pcvl.Parameter("phi")  # Or equivalently pcvl.P("phi")
>>> p.spv
phi
>>> p.defined
False
>>> p.set_value(3.14)
>>> float(p)
3.14
>>> p.defined
True

When defining the parameter, you can also set its numerical value, max/min boundaries and periodicity:

>>> import perceval as pcvl
>>> import math
>>> alpha = pcvl.P("phi", min_v=0, max_v=2*math.pi, periodic=True)
>>> alpha.is_periodic
True

.. autoclass:: perceval.utils.parameter.Parameter
   :members:
