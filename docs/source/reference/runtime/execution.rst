Execution
^^^^^^^^^

An :code:`Execution` is responsible for running a :code:`Computation` on a :ref:`Computer` and retrieving its results.
Internally, it hides the complexity of the execution, stores intermediate variables, and offers monitoring convenience.
It provides the same interface for local and remote computers, and supports both synchronous and asynchronous runs.
The computer within the execution determines where and how the computation is performed.

Executions are normally created by an :code:`ExecutionFactory`:

>>> import perceval as pcvl
>>>
>>> experiment = pcvl.Experiment(pcvl.BS())
>>> experiment.with_input(pcvl.BasicState("|1,1>"))
>>> computer = pcvl.SimulatedComputer("SLOS")
>>> factory = pcvl.ExecutionFactory(computer, experiment)
>>> execution = factory.sample_count

They can also be created directly using a :ref:`Computation` or a :ref:`ComputationIterator`:

>>> computation = factory.build_computation("sample_count")
>>> computation.add_params(max_samples = 10000)
>>> execution = pcvl.Execution(computation, computer)

Synchronous execution
=====================

Call :meth:`execute_sync()` to run an execution synchronously. The call blocks until the results are ready. Calling
the execution directly is equivalent:

>>> with computer.acquire():
...     results = execution.execute_sync(max_samples=10_000)  # Giving parameters here override the ones given in the Computation
...     # Equivalently: results = execution(max_samples=10_000)

The returned dictionary contains a :code:`"results"` if it has a :ref:`Computation`,
or :code:`"results_list"` entry if it has a :ref:`ComputationIterator`,
and may contain additional data such as performance values.

A progress callback can be installed before a synchronous run. Returning :code:`True` from the callback requests
cancellation:

>>> def progress_callback(progress: float, message: str):
...     print(f"{progress:.0%}: {message}")
...     return False
>>>
>>> execution = factory.sample_count
>>> execution.set_progress_callback(progress_callback)

Asynchronous execution
======================

Call :meth:`execute_async()` to launch an execution without waiting for completion. The method returns the same
execution object, whose status can then be monitored:

>>> import time
>>>
>>> execution = factory.sample_count
>>> with execution.computer.acquire():
...     execution.execute_async(max_samples=10_000)
...     while not execution.is_complete:
...         print(execution.status.progress)
...         time.sleep(1)
...     if execution.is_failed:
...         print(execution.status.stop_message)
...     else:
...         results = execution.get_results()

.. warning::
   A computer must remain started until all asynchronous executions associated with it have finished. Use
   :meth:`Computer.start <perceval.runtime.abstract_computer.AComputer.start>` and
   :meth:`Computer.stop <perceval.runtime.abstract_computer.AComputer.stop>` when launching an execution and
   waiting for its completion in different scopes.

Cancellation and rerunning
==========================

Cancellation can be requested for an execution that has already been launched and has not completed:

>>> execution.cancel()

Cancellation may take some time. Partial results, when available, can be retrieved with:

>>> partial_results = execution.get_results(allow_partial_results=True)

An execution cannot be launched more than once. Use :meth:`clone()` to create an equivalent unsent execution, or
:meth:`rerun()` to create and asynchronously launch a replacement for a failed execution:

>>> another_execution = execution.clone()

Storage
=======

Launched remote :code:`Executions` can't be directly created using their cloud id, as they could be linked to
several cloud jobs due to :ref:`Error Mitigation`.
To be able to access the results of an :code:`Execution` in another script,
it must be serialized after launch and stored somewhere, then deserialized in the other script.

Use an :ref:`ExecutionGroup` to do that automatically for you, or use the :ref:`serialization` system.

Execution status
================

The :attr:`status` property returns an :ref:`ExecutionStatus`. Convenience properties are available for common state
checks:

>>> status = execution.status
>>> execution.was_sent
>>> execution.is_waiting
>>> execution.is_running
>>> execution.is_complete
>>> execution.is_success
>>> execution.is_failed

Class reference
===============

.. autoclass:: perceval.runtime.execution.Execution
   :members:
