State Generator
===============

This class provides a way to generate usual StateVectors with a given encoding.

>>> from perceval import StateGenerator, Encoding
>>> sg = StateGenerator(Encoding.DUAL_RAIL)
>>> sg.bell_state("phi+")
0.707*|1,0,1,0>+0.707*|0,1,0,1>
>>> sg.dicke_state(2)
0.408*|1,0,0,1,0,1,1,0>+0.408*|0,1,1,0,1,0,0,1>+0.408*|0,1,1,0,0,1,1,0>+0.408*|0,1,0,1,1,0,1,0>+0.408*|1,0,0,1,1,0,0,1>+0.408*|1,0,1,0,0,1,0,1>

.. autoclass:: perceval.utils.stategenerator.StateGenerator
   :members:
