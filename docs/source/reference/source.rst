Source
======

>>> import perceval as pcvl
>>> source = pcvl.Source(emission_probability=0.3, multiphoton_component=0.05)
>>> pcvl.pdisplay(source.probability_distribution())

+-----------------+-----------------+
| state           | probability     |
+-----------------+-----------------+
|  \|0>           |      7/10       |
+-----------------+-----------------+
|  \|{_:0}>       |    0.297716     |
+-----------------+-----------------+
|  \|{_:0}{_:2}>  |    0.002284     |
+-----------------+-----------------+

.. autoclass:: perceval.components.source.Source
   :members:
   :inherited-members:
