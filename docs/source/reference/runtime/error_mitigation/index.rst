Error mitigation
================

Error mitigation uses knowledge about a computer's imperfections to reduce their impact on the returned results. It
does not make the underlying hardware error-free: a mitigation may prepare additional sub-computations, change how the
measurement budget is distributed, and post-process the collected data.

Perceval provides four mitigation techniques:

* :ref:`CompilationAveraging` reduces dependence on one physical compilation of an experiment.
* :ref:`DistinguishablePhotonMitigation` corrects part of the error caused by imperfect photon indistinguishability
  and multi-photon emission.
* :ref:`DetectorBalancing` compensates for detector losses and unequal detector efficiencies.
* :ref:`PhotonRecycling` recovers information from events in which one or two photons were lost.

The techniques address different imperfections and can be combined. A higher correction strength is not always a
better choice: it can add compilation or execution overhead, may require more samples and shots to work properly,
and every technique has conditions under which it is useful.

Using mitigations with a Computer
---------------------------------

Assign an ordered list of mitigation objects to a :ref:`Computer` before creating or running an execution:

>>> import perceval as pcvl
>>>
>>> computer = pcvl.SimulatedComputer("SLOS")
>>> computer.mitigations = [
...     pcvl.DistinguishablePhotonMitigation(order=1),
...     pcvl.DetectorBalancing(),
... ]

The computer applies the mitigations automatically to compatible computations. The caller still uses the normal
:ref:`ExecutionFactory` and :ref:`Execution` interfaces, and receives the usual result dictionary:

>>> experiment = pcvl.Experiment(pcvl.BS())
>>> experiment.with_input(pcvl.BasicState("|1,1>"))
>>> experiment.min_detected_photons_filter(1)
>>> factory = pcvl.ExecutionFactory(computer, experiment)
>>> with computer.acquire():
...     results = factory.sample_count(max_samples=10_000)

:ref:`Computations <Computation>` are decomposed in list order, and post-processing happens in reverse order.
The :ref:`MitigationFactory` provides a standard, recommended order for the mitigations.

.. note::
   Error mitigation is applied only to :ref:`Commands <Command>` that declare themselves compatible with it.
   Standard probability and sampling commands are compatible.

.. note::
   The pre and post processing always apply synchronously, no matter what was asked at the :ref:`Execution` level.

Choosing a preset
-----------------

The :ref:`MitigationFactory` provides convenient presets:

>>> mitigation_factory = pcvl.MitigationFactory(pcvl.MitigationLevel.medium)
>>> computer.mitigations = mitigation_factory.build()

The presets are:

- :code:`MitigationLevel.none`
- :code:`MitigationLevel.low`
- :code:`MitigationLevel.medium`
- :code:`MitigationLevel.high`

Photon recycling is never enabled by a preset because its experiment and input requirements are more restrictive. It
can be enabled explicitly on the factory when appropriate. See :ref:`MitigationFactory` for customization examples.

Disabling and temporarily changing mitigations
----------------------------------------------

Assign an empty list or :code:`None` to disable all mitigations explicitly:

>>> computer.mitigations = []

Use :meth:`Computer.apply_configuration
<perceval.runtime.abstract_computer.AComputer.apply_configuration>` to change mitigations temporarily:

>>> temporary_mitigations = [pcvl.DetectorBalancing()]
>>> with computer.apply_configuration(mitigations=temporary_mitigations):
...     results = factory.probs()

The previous configuration is restored when the context exits.

.. warning::
   Temporarily changing a computer configuration is generally unsafe when using asynchronous work.

Remote computers
----------------

For a :ref:`RemoteComputer`, :attr:`mitigations` set to :code:`None` means that the remote provider may apply its
default mitigations. An empty list explicitly disables them, while a non-empty list requests the selected techniques:

>>> remote_computer.mitigations = None  # Use the remote platform defaults
>>> remote_computer.mitigations = []    # Disable remote mitigations
>>> remote_computer.mitigations = [pcvl.DetectorBalancing()]

By default, the requested mitigations are sent to the remote platform. Set :attr:`use_mitigations_remotely` to
:code:`False` to execute the required sub-computations remotely but combine and correct their results on the local
machine:

>>> remote_computer.use_mitigations_remotely = False

.. warning::
   With local mitigation of remote results, Perceval uses the imperfections known when the execution is launched. They
   may differ from the platform conditions when the remote job actually runs.

Class reference
---------------

.. toctree::

   mitigation_factory
   compilation_averaging
   distinguishable_photon_mitigation
   detector_balancing
   photon_recycling
