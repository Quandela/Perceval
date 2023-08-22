Simulator
=========

List of simulation methods in the :code:`Simulator`

+-----------------+----------------------------------------+---------------------------+
| **method name** | **parameters**                         | **output**                |
+-----------------+----------------------------------------+---------------------------+
| prob_amplitude  | input_state: BasicState or StateVector,| prob. amplitude (complex) |
|                 | output_state: BasicState               |                           |
+-----------------+----------------------------------------+---------------------------+
| probability     | input_state: BasicState or StateVector,| probability (float [0;1]) |
|                 | output_state: BasicState               |                           |
+-----------------+----------------------------------------+---------------------------+
| probs           | input_state: BasicState or StateVector | prob. distribution (BSD)  |
+-----------------+----------------------------------------+---------------------------+
| probs_svd       | input_dist: SVDistribution             | Python dictionary         |
+-----------------+----------------------------------------+---------------------------+
| evolve          | input_state: BasicState or StateVector | evolved StateVector       |
+-----------------+----------------------------------------+---------------------------+

.. autoclass:: perceval.simulators.Simulator
   :members:
   :inherited-members:
