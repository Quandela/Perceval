CompilationAveraging
====================

Compilation averaging reduces the effect of choosing one particular physical compilation of an experiment. Perceval
divides the requested measurement budget over several compilation seeds, runs those sub-computations, and combines
their sample counts into one result.

It can be used when the target platform supports a :code:`compilation_seed` parameter and different valid compilations
may have different physical performance:

>>> import perceval as pcvl
>>>
>>> computer.mitigations = [pcvl.CompilationAveraging(repetitions=3)]

The optional :code:`starting_seed` makes the sequence of compilation seeds reproducible:

>>> mitigation = pcvl.CompilationAveraging(repetitions=3, starting_seed=100)

Practical considerations
------------------------

* The command must accept :code:`compilation_seed`. Otherwise, this mitigation leaves the computation unchanged.
* :code:`repetitions` must be a positive integer.
* When :code:`max_samples` or :code:`max_shots` is set, its value must be at least the number of repetitions. The total
  requested budget is divided between the repetitions rather than repeated in full for each one.

.. note::
   Avoid setting a too high value for the number of repetitions, as compilation can take some time.

.. autoclass:: perceval.runtime.error_mitigation.compilation_averaging.CompilationAveraging
   :members:
