Simulator
=========

The :code:`Simulator` class is a mid-level class that can be used to compute
output probabilities or amplitudes for photonic experiment.

It adds logic that allows computing with objects that are more complex than non-annotated BasicStates,
contrarily to the backend it build upon to do its computations.
If possible, it will also automatically use masks on the backend to reduce the computation time and memory.

Also, contrarily to :code:`Experiment` or :code:`Processor`,
it relies on being given complete information from the start and cannot be used to do composition, remote computing...

The basic :code:`Simulator` is only able to perform computation with non-polarized unitary circuits.
For polarized circuits, see the :code:`PolarizationSimulator`.
For non-unitary circuits, see the :code:`DelaySimulator`, :code:`LossSimulator`, and :code:`FFSimulator`.
More generally, if you need another simulator than this one due to your circuit's components, see :code:`SimulatorFactory`.

Using a simulator
^^^^^^^^^^^^^^^^^^^^

.. code-block::

  import perceval as pcvl

  sim = pcvl.Simulator(pcvl.SLOSBackend())
  sim.set_circuit(pcvl.BS())  # Now, sim holds a 2 modes circuit

  # Physical and logical selection
  sim.set_selection(min_detected_photons_filter = 2)  # Other fields: heralds (accounting only the output), postselect

  sim.set_precision(1e-6)  # Relative precision; only input states having more than this times the highest input prob will be computed

  # Computation state by state
  print(sim.prob_amplitude(pcvl.BasicState("|{_:0}, {_:1}>"), pcvl.BasicState("|{_:0}, {_:1}>")))  # No selection
  # (0.5000000000000001+0j)

  # Compute everything
  print(sim.probs(pcvl.BasicState([2, 1])))  # Computes the BSD with selection; no performance and no automatic usage of masks
  # {
  #   |1,2>: 0.12500000000000003
  #   |3,0>: 0.375
  #   |2,1>: 0.12500000000000003
  #   |0,3>: 0.375
  # }

  # Compute everything from anything
  svd = pcvl.SVDistribution({pcvl.StateVector([1, 1]) + 0.5j * pcvl.StateVector([0, 2]): 0.7,
                             pcvl.StateVector([1, 0]): 0.3})
  print(sim.probs_svd(svd, [pcvl.Detector.threshold()] * 2))  # Can also simulate detectors
  # {'results': BSDistribution(<class 'float'>, {|1,1>: 1.0}), 'physical_perf': 0.06999999999999998, 'logical_perf': 1.0000000000000004}

Computation methods
^^^^^^^^^^^^^^^^^^^

A lot of computation methods exist in the :code:`Simulator` for different usages.

Here is a list of the simulation methods in the :code:`Simulator`

+----------------------+----------------------------------------------------------+---------------------------+
| **method name**      | **parameters**                                           | **output**                |
+----------------------+----------------------------------------------------------+---------------------------+
| prob_amplitude       | input_state: BasicState or StateVector,                  | prob. amplitude (complex) |
|                      | output_state: BasicState                                 |                           |
+----------------------+----------------------------------------------------------+---------------------------+
| probability          | input_state: BasicState or StateVector,                  | probability (float [0;1]) |
|                      | output_state: BasicState                                 |                           |
+----------------------+----------------------------------------------------------+---------------------------+
| probs                | input_state: BasicState or StateVector                   | prob. distribution (BSD)  |
+----------------------+----------------------------------------------------------+---------------------------+
| probs_svd            | input_dist: SVDistribution                               | Python dictionary         |
+----------------------+----------------------------------------------------------+---------------------------+
| probs_density_matrix | dm: DensityMatrix                                        | Python dictionary         |
+----------------------+----------------------------------------------------------+---------------------------+
| evolve               | input_state: BasicState or StateVector                   | evolved StateVector       |
+----------------------+----------------------------------------------------------+---------------------------+
| evolve_svd           | input_state: SVDistribution or StateVector or BasicState | Python dictionary         |
+----------------------+----------------------------------------------------------+---------------------------+
| evolve_density_matrix| dm: DensityMatrix                                        | Python dictionary         |
+----------------------+----------------------------------------------------------+---------------------------+

.. autoclass:: perceval.simulators.Simulator
   :members:
   :inherited-members:
