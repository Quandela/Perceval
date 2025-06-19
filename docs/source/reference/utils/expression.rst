Expression
==========

This is a derived class from :code:`Parameter` that can hold mathematical expressions.
An :code:`Expression` is automatically generated when using operators on :code:`Parameter` or :code:`Expression`.

>>> import perceval as pcvl
>>> a, b = pcvl.Parameter("a"), pcvl.Parameter("b")
>>> e = a + b
>>> e.spv
a + b
>>> a.set_value(1)
>>> b.set_value(2)
>>> float(e)
3.0

They can also be created manually to accept more general mathematical functions (using the sympy expression parsing)

>>> phi = pcvl.Parameter("phi")
>>> e = pcvl.Expression("cos(phi)", {phi})  # Declares phi as a sub-parameter. Equivalent to pcvl.E("cos(phi)", {phi})
>>> phi.set_value(0)
>>> float(e)
1.0

.. autoclass:: perceval.utils.parameter.Expression
   :members:
