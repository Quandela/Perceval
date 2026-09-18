MitigationFactory
=================

:code:`MitigationFactory` creates an ordered list of standard mitigations suitable for assigning to a
:ref:`Computer`. Start from a preset level, optionally customize individual techniques, then call :meth:`build()`:

>>> import perceval as pcvl
>>>
>>> factory = pcvl.MitigationFactory(pcvl.MitigationLevel.medium)
>>> factory.set_compilation_averaging(repetitions=4)
>>> factory.set_detector_balancing(False)  # Deactivate detector balancing
>>> mitigations = factory.build()
>>> computer.mitigations = mitigations

Preset levels
-------------

The :code:`MitigationLevel` values provide progressively stronger starting configurations:

* :code:`MitigationLevel.none` produces an empty list.
* :code:`MitigationLevel.low` enables detector balancing.
* :code:`MitigationLevel.medium` adds compilation averaging with three repetitions and distinguishable-photon
  mitigation at order one.
* :code:`MitigationLevel.high` uses five compilation repetitions and distinguishable-photon mitigation at order three,
  together with detector balancing.

Higher levels can require more sub-computations and compilation work. Choose a level based on the known imperfections,
the available execution budget, and the mitigation requirements—not only on the level name.

Customizing a factory
---------------------

Every built-in technique has a setter. Passing :code:`None` disables compilation averaging or
distinguishable-photon mitigation; passing :code:`False` disables photon recycling or detector balancing:

>>> factory = pcvl.MitigationFactory(pcvl.MitigationLevel.none)
>>> factory.set_compilation_averaging(3)
>>> factory.set_distinguishable_photon_mitigation(1)
>>> factory.set_photon_recycling(True)
>>> factory.set_detector_balancing(True)

Use :meth:`set_custom_mitigation()` to supply an already configured instance of one of the built-in mitigation types:

>>> factory.set_custom_mitigation(pcvl.CompilationAveraging(4, starting_seed=10))

This method replaces the instance for that built-in type. It does not register arbitrary new mitigation classes.

Calling :meth:`reset_to_level()` discards the custom settings and restores a preset:

>>> factory.reset_to_level(pcvl.MitigationLevel.low)

.. autoclass:: perceval.runtime.error_mitigation.mitigation_factory.MitigationLevel
   :members:

.. autoclass:: perceval.runtime.error_mitigation.mitigation_factory.MitigationFactory
   :members:
