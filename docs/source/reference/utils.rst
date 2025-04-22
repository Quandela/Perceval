Matrix
======

This class is used to represent both numeric and symbolic complex matrices.
Every matrix in perceval is an instance of this class.

>>> import perceval as pcvl
>>> M = pcvl.Matrix("1 2 3\n4 5 6\n7 8 9")
>>> print(M.pdisplay())
⎡1  2  3⎤
⎢4  5  6⎥
⎣7  8  9⎦
>>> M.is_unitary()
False

It also comes with utility methods to create unitary matrices

>>> random_unitary = pcvl.Matrix.random_unitary(6)  # 6*6
>>> deterministic_unitary = pcvl.Matrix.parametrized_unitary(list(range(8)))  # 2*2 (requires 2*m**2 parameters)
>>> from_array_unitary = pcvl.Matrix.get_unitary_extension(numpy_2d_array)  # Size (row+col) * (row+col)

.. autoclass:: perceval.utils.matrix.Matrix
   :members:
   :special-members: __new__


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
