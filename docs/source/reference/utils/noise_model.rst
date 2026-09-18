Noise Model
===========

>>> import perceval as pcvl
>>> noise_model = pcvl.NoiseModel(brightness=0.3, g2=0.05)
>>> computer = pcvl.SimulatedComputer("SLOS")
>>> computer.noise = noise_model

.. autoclass:: perceval.utils.noise_model.NoiseModel
   :members:
   :inherited-members:
