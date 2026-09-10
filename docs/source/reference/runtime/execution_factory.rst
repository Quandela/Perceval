.. _ExecutionFactory:

ExecutionFactory
^^^^^^^^^^^^^^^^

An :code:`ExecutionFactory` creates :ref:`Computations <Computation>`, :ref:`ComputationIterators
<ComputationIterator>`, and :ref:`Executions <Execution>` for a fixed :ref:`Computer` and :ref:`Experiment`. It is the
usual entry point for requesting probabilities or samples without constructing these runtime objects manually.

Creating a factory
==================

Pass the computer that will perform the work and the experiment to compute:

>>> import perceval as pcvl
>>>
>>> experiment = pcvl.Experiment(pcvl.BS())
>>> experiment.with_input(pcvl.BasicState("|1,1>"))
>>> computer = pcvl.SimulatedComputer("SLOS")
>>> factory = pcvl.ExecutionFactory(computer, experiment)

For a :ref:`RemoteComputer`, :code:`max_shots_per_call` is mandatory.
It provides a positive default upper bound on the shots used by every computation built by the factory:

>>> communication_layer = pcvl.QuandelaCommunicationLayer("sim:belenos")
>>> remote_computer = pcvl.RemoteComputer(communication_layer)
>>> remote_factory = pcvl.ExecutionFactory(
...     remote_computer,
...     experiment,
...     max_shots_per_call=100_000,
... )

Standard commands
=================

The factory exposes the three standard :ref:`Commands <Command>` as properties:

* :attr:`probs` creates an execution that computes output probabilities.
* :attr:`samples` creates an execution that returns individual output samples.
* :attr:`sample_count` creates an execution that groups samples by state and occurrence count.

Accessing a property creates a fresh, unsent execution. Execution parameters can then be supplied when it is run:

>>> sample_count = factory.sample_count
>>> with computer.acquire():
...     results = sample_count(max_samples=10_000)

Because every property access produces a new object, executions with different parameters can be prepared
independently:

>>> short_run = factory.sample_count
>>> long_run = factory.sample_count
>>> short_run is long_run
False

Custom commands
===============

Commands registered by the computer are exposed dynamically using their command name. For example, if a provider's
computer advertises a :code:`custom_command`, it can be accessed as follows:

>>> custom_execution = factory.custom_command

Use :attr:`Computer.available_commands <perceval.runtime.abstract_computer.AComputer.available_commands>`
to discover the names supported by a computer.

Building computations and executions
=====================================

:meth:`build_computation()` creates a :ref:`Computation` for a named command without wrapping it in an execution.
:meth:`build_execution()` then associates a computation with the factory's computer:

>>> computation = factory.build_computation("sample_count")
>>> computation.add_params(max_samples=10_000)
>>> execution = factory.build_execution(computation)

The factory's :attr:`default_job_name` is copied to every execution created afterward when it is not :code:`None`:

>>> factory.default_job_name = "my sampling run"
>>> execution = factory.sample_count
>>> execution.name
'my sampling run'

Iterations
==========

Iterations describe several variants of the base computation. Add them individually with :meth:`add_iteration()` or
in a list with :meth:`add_iteration_list()`:

>>> factory.add_iteration(
...     input_state=pcvl.BasicState("|1,1>"),
...     min_detected_photons=1,
... )
>>> factory.add_iteration_list([
...     {
...         "input_state": pcvl.BasicState("|2,0>"),
...         "min_detected_photons": 1,
...         "max_samples": 2_000,
...     },
... ])

When at least one iteration is stored, :meth:`build_computation()` and the command properties build a
:ref:`ComputationIterator` instead of a single computation:

>>> execution = factory.sample_count
>>> isinstance(execution.computation, pcvl.ComputationIterator)
True
>>> factory.n_iterations
2

The supported iteration keys are :code:`circuit_params`, :code:`input_state`,
:code:`min_detected_photons`, :code:`max_samples`, :code:`max_shots`, and :code:`postselect`.
They are validated when a computation is built.

Iterations remain in the factory and affect every later computation until :meth:`clear_iterations()` is called:

>>> factory.clear_iterations()
>>> factory.n_iterations
0
>>> isinstance(factory.build_computation("sample_count"), pcvl.Computation)
True

An execution backed by a computation iterator returns its individual results in the :code:`"results_list"` field, in
the same order as the iterations.

Class reference
===============

.. autoclass:: perceval.runtime.execution_factory.ExecutionFactory
   :members:
