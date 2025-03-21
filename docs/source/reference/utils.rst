Matrix
======

>>> import perceval as pcvl
>>> M = pcvl.Matrix("1 2 3\n4 5 6\n7 8 9")
>>> print(M.pdisplay())
⎡1  2  3⎤
⎢4  5  6⎥
⎣7  8  9⎦
>>> M.is_unitary()
False

.. autoclass:: perceval.utils.matrix.Matrix
   :members:
   .. automethod:: __new__


Parameter
=========

.. autoclass:: perceval.utils.parameter.Parameter
   :members:


Expression
==========

.. autoclass:: perceval.utils.parameter.Expression
   :members:
