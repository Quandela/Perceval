Source
======

>>> import perceval as pcvl
>>> source = pcvl.Source(brightness=0.3, purity=0.95)
>>> pcvl.pdisplay(source.probability_distribution())

+--------+-------------+
| state  | probability |
+--------+-------------+
|  |0>   |    7/10     |
+--------+-------------+
|  |1>   |    0.285    |
+--------+-------------+
|  |2>   |    0.015    |
+--------+-------------+

.. autoclass:: perceval.components.source.Source
   :members:
   :inherited-members:
