random_seed
===========

To achieve a reproducible result, the :code:`pcvl.random_seed()` function can be used before a given computation.
This function ensures that any random numbers used in algorithms will be the same from run to run.

Example:

>>> from perceval import random_seed
>>> import random
>>>
>>> random_seed(2)  # Set the seed to 2
>>> print(random.random())
0.9478274870593494
>>> print(random.random())
0.9560342718892494
>>> random_seed(2)  # Reset the seed to 2
>>> print(random.random())
0.9478274870593494
>>> print(random.random())
0.9560342718892494

The random real numbers drawn are in the same order when the seed is fixed to the same value.

.. autofunction:: perceval.utils._random.random_seed
