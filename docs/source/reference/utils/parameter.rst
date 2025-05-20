Parameter
=========

This class holds parameter values that can be used in circuits and in backends supporting symbolic computation.

>>> import perceval as pcvl
>>> p = pcvl.Parameter("phi")  # Or equivalently pcvl.P("phi")
>>> p.spv
phi
>>> p.set_value(3.14)
>>> float(p)
3.14

.. autoclass:: perceval.utils.parameter.Parameter
   :members:
