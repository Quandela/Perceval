NoisySamplingSimulator
======================

The :code:`NoisySamplingSimulator` is a special simulator dedicated to sample states from a noisy input state.

It is completely separated from the other simulators, and, as such, it is not available through the :code:`SimulatorFactory`.
As its name suggests, it requires a sampling able backend such as :ref:`CliffordClifford2017`.

The :code:`NoisySamplingSimulator` exposes two simulation methods:

- :code:`samples` that returns a python dictionary with a BSSamples in the "results" field
- :code:`sample_count` that returns a python dictionary with a BSCount in the "results" field

Note that these two methods require a :code:`SVDistribution` without superposed states (but can be annotated),
since we can't retrieve the probability amplitudes from the backend.

Also, this simulator can only simulate non-polarized unitary circuits.

Using a NoisySamplingSimulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

  import perceval as pcvl

  sim = pcvl.NoisySamplingSimulator(pcvl.Clifford2017Backend())
  sim.set_circuit(pcvl.BS())  # Now, sim holds a 2 modes circuit

  # Physical and logical selection
  sim.set_selection(min_detected_photons_filter = 2)  # Other fields: heralds (accounting only the output), postselect
  sim.set_detectors([pcvl.Detector.threshold(), pcvl.Detector.pnr()])

  svd = pcvl.SVDistribution({pcvl.BasicState("|{_:0}, {_:1}>"): 1})
  # Sample stream
  print(sim.samples(svd, 5)["results"])  # Random sampling; results may change at each run
  # [ |1,1>, |0,2>, |0,2>, |1,1>, |1,1> ]

  # Sample count
  print(sim.sample_count(svd, max_samples = 10, max_shots = 10)["results"])
  # {
  #   |1,1>: 7
  #   |0,2>: 2
  # }


.. autoclass:: perceval.simulators.NoisySamplingSimulator
   :members:
   :inherited-members:
