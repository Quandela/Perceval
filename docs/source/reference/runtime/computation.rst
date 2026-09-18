Computation
^^^^^^^^^^^

A :code:`Computation` describes what should be computed, independently of the :ref:`Computer` that will perform the
work. It combines a :ref:`Command`, which defines the requested result and its accepted parameters, with the
:ref:`Experiment` with which that command will run.

Computations are normally created by an :code:`ExecutionFactory`:

>>> import perceval as pcvl
>>>
>>> experiment = pcvl.Experiment(pcvl.BS())
>>> experiment.with_input(pcvl.BasicState("|1,1>"))
>>> computer = pcvl.SimulatedComputer("SLOS")
>>> factory = pcvl.ExecutionFactory(computer, experiment)
>>> computation = factory.build_computation("sample_count")

They can also be created directly from a command and an experiment:

>>> computation = pcvl.Computation(computer.get_command("sample_count"), experiment)

Parameters
==========

The parameters accepted by a computation are defined by its command. :meth:`add_params()` accepts positional or
keyword arguments, validates their names and types against the command signature, and adds them to the
:attr:`parameters` dictionary:

>>> computation.add_params(max_samples=10_000, max_shots=100_000)
>>> computation.parameters
{'max_samples': 10000, 'max_shots': 100000}

Calling :meth:`validate()` checks that every mandatory command parameter has been provided. It is also called
automatically before a computer executes the computation.

Standard Commands
=================

The :code:`"probs"`, :code:`"sample_count"` and :code:`"samples"` :ref:`Commands <Command>` are standardized
for all :ref:`Computers <Computer>`.
Most of their parameters are shared, so it should be possible to use the same :code:`Computation`
on any :ref:`Computer` for these commands.

Class reference
===============

.. autoclass:: perceval.runtime.computation.Computation
   :members:

ComputationIterator
^^^^^^^^^^^^^^^^^^^

A :code:`ComputationIterator` describes several independent variants of one base :code:`Computation`. Each iteration
can change a supported subset of the experiment or computation parameters. This is useful for parameter sweeps because
the variants keep the same :ref:`Execution`.

An iterator can be created using the factory
(in which case creating an :ref:`Execution` directly creates it with a :code:`ComputationIterator`):

>>> factory = pcvl.ExecutionFactory(computer, experiment)
>>> factory.add_iteration(input_state=pcvl.BasicState("|1,1>"))
>>> factory.add_iteration(input_state=pcvl.BasicState("|2,0>"), max_samples=2_000)
>>> computation = factory.build_computation("sample_count")
>>> computation.add_params(max_samples = 1_000)

An iterator can also be created directly:

>>> base_computation = pcvl.Computation(computer.get_command("sample_count"), experiment)
>>> base_computation.add_params(max_samples=1_000)
>>> computation_iterator = pcvl.ComputationIterator(base_computation)
>>> computation_iterator.add_iteration(input_state=pcvl.BasicState("|1,1>"))
>>> computation_iterator.add_iteration(input_state=pcvl.BasicState("|2,0>"), max_samples=2_000)

The supported iteration parameters are:

* :code:`circuit_params`: numerical values for named circuit parameters
* :code:`input_state`: the input :class:`~perceval.utils.statevector.BasicState`
* :code:`min_detected_photons`: minimum accepted photon count
* :code:`max_samples`: maximum number of samples to collect
* :code:`max_shots`: maximum number of shots to perform
* :code:`postselect`: a :class:`~perceval.utils.postselect.PostSelect` condition

Iteration parameters are checked when :meth:`add_iteration()` is called. Iterating over the object yields a new,
independent :code:`Computation` for each set of parameters, leaving the base computation unchanged:

>>> for computation in computation_iterator:
...     print(computation)

When an iterator is executed, its output dictionary contains a :code:`"results_list"` entry. Results appear in
iteration order, and each result includes the iteration parameters that produced it.

The recommended way to prepare an iterator is through :class:`ExecutionFactory
<perceval.runtime.execution_factory.ExecutionFactory>`, which builds the iterator automatically when iterations have
been added:

>>> factory.add_iteration(input_state=pcvl.BasicState("|1,1>"))
>>> factory.add_iteration(input_state=pcvl.BasicState("|2,0>"))
>>> execution = factory.sample_count
>>> isinstance(execution.computation, pcvl.ComputationIterator)
True

Class reference
===============

.. autoclass:: perceval.runtime.computation_iterator.ComputationIterator
   :members:
