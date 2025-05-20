Detector
========

Detectors are components that aim at simulating real hardware detectors, or perfect detectors.

They all share a common interface for your own usage.

>>> import perceval as pcvl
>>> d = pcvl.Detector.threshold()
>>> d.detect(3)  # Common method for all kind of detectors
|1>

Detectors can also be added to :ref:`Processor` or :ref:`Experiment` mode by mode for an automatic usage.
Once a detector is added to a :code:`Processor` or an :code:`Experiment` mode, this mode is considered to be a classical mode,
so optical components can no longer be added to it, but classical components can now be added to it.

>>> import perceval as pcvl
>>> e = pcvl.Experiment(2)
>>> e.add(0, pcvl.BS())  # Can add optical component
>>> e.add(0, pcvl.Detector.ppnr(12))
>>> # e.add(0, pcvl.PS(1))  # Can no longer add optical component
>>> ff_config = pcvl.FFCircuitProvider(1, 0, pcvl.PS(3.14)).add_configuration((1,), pcvl.PS(1.57))
>>> e.add(0, ff_config)  # Can add classical component


.. autoclass:: perceval.components.detector.Detector
   :members:
   :inherited-members:
   :exclude-members: is_composite

.. autoclass:: perceval.components.detector.BSLayeredPPNR
   :members:
   :inherited-members:
   :exclude-members: is_composite

.. autoenum:: perceval.components.detector.DetectionType

.. autofunction:: perceval.components.detector.get_detection_type
