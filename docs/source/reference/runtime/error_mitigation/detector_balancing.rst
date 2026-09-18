DetectorBalancing
=================

Detector balancing compensates for photon loss and unequal efficiency across output detectors. It uses the detector
models known by the computer to estimate how likely each output state was to be observed, then adjusts the output
probabilities accordingly.

The mitigation has no parameters:

>>> import perceval as pcvl
>>>
>>> computer.mitigations = [pcvl.DetectorBalancing()]

Practical considerations
------------------------

* The correction is only as accurate as the detector models supplied by the simulator or remote platform.
* The result should include every relevant state, including bunched states, for the most reliable correction.
  Using this with low number of samples may introduce more problems than not using this.

.. autoclass:: perceval.runtime.error_mitigation.detector_balancing.DetectorBalancing
   :members:
