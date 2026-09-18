PhotonRecycling
===============

Photon recycling uses the measured events in which one or two photons were lost to estimate the ideal-photon-number
output distribution. The method was introduced in :cite:t:`mills2024`.

Automatic use with a Computer
-----------------------------

Add :code:`PhotonRecycling` to the computer's mitigation list:

>>> import perceval as pcvl
>>>
>>> computer.mitigations = [pcvl.PhotonRecycling()]

The mitigation automatically requests the lower-photon events required by the algorithm and returns the corrected
distribution through the normal execution result.

Photon recycling is not part of any :ref:`MitigationFactory` preset. It can be enabled explicitly:

>>> factory = pcvl.MitigationFactory(pcvl.MitigationLevel.medium)
>>> factory.set_photon_recycling()
>>> computer.mitigations = factory.build()

Practical considerations
------------------------

* The experiment must be unitary and have a fixed input photon count of at least three.
* The technique estimates a probability distribution, and does not need prior characterization of the computer.

.. autoclass:: perceval.runtime.error_mitigation.photon_recycling.PhotonRecycling
   :members:

Applying photon recycling directly
----------------------------------

The standalone :func:`photon_recycling()` function can correct an existing :code:`BSCount` or
:code:`BSDistribution` when its ideal photon count is known:

>>> mitigated_distribution = pcvl.photon_recycling(
...     noisy_distribution,
...     ideal_photon_count=4,
... )

The input must contain both three-photon and two-photon events for an ideal count of four. Passing a lossless
distribution, an incompatible result type, or insufficient loss statistics raises an error.

.. autofunction:: perceval.runtime.error_mitigation.photon_recycling.photon_recycling
