Matrix
======

This class is used to represent both numeric and symbolic complex matrices.
Every matrix in perceval is an instance of this class.

>>> import perceval as pcvl
>>> M = pcvl.Matrix("1 2 3\n4 5 6\n7 8 9")
>>> pcvl.pdisplay(M)
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
