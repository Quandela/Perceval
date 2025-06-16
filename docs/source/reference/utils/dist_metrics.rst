metrics
^^^^^^^

>>> import perceval as pcvl
>>> dist_a = pcvl.BSDistribution({pcvl.BasicState([1, 0]): 0.4, pcvl.BasicState([0, 1]): 0.6})
>>> dist_b = pcvl.BSDistribution({pcvl.BasicState([1, 0]): 0.3, pcvl.BasicState([0, 1]): 0.7})
>>> print(pcvl.tvd_dist(dist_a, dist_b))
0.1
>>> print(pcvl.kl_divergence(dist_a, dist_b))
0.022582421084357485

Perceval provides ways to compare :code:`BSDistribution` with mathematical metrics.

.. automodule:: perceval.utils.dist_metrics
   :members:
