ExecutionGroup
==============

The :code:`ExecutionGroup` class manages a named collection of :ref:`Execution` objects. Groups are stored locally,
so large experiments can be split into several executions, launched over multiple Python sessions, and retrieved from
a single place. Creating an :code:`ExecutionGroup` with an existing name loads the previously stored group.

An execution group can contain both local and remote executions. It can launch unsent executions sequentially or in
parallel, rerun unsuccessful executions, monitor their progress, cancel active executions, and retrieve all available
results at once.

.. warning::
   Execution groups store their data in a "execution_groups" directory under the current directory by default.
   These files can grow quite large and are not removed automatically.
   They can be removed by hand if they are no longer needed.

.. note::
   The storage directory can be changed at instantiation of an :code:`ExecutionGroup`.
   An existing group can be retrieved if and only if both the directory and the name match the original values.

Usage example
-------------

The following example prepares a group containing two executions of the same acquisition, one using a post-processed
CNOT gate and the other a heralded CNOT gate:

>>> import perceval as pcvl
>>>
>>> computer = pcvl.RemoteComputer(pcvl.QuandelaCommunicationLayer("sim:belenos"))
>>>
>>> ralph_experiment = pcvl.catalog["postprocessed cnot"].build_experiment()
>>> ralph_experiment.min_detected_photons_filter(2)
>>> ralph_experiment.with_input(pcvl.BasicState([0, 1, 0, 1]))
>>> ralph_factory = pcvl.ExecutionFactory(computer, ralph_experiment, max_shots_per_call=1_000_000)
>>>
>>> knill_experiment = pcvl.catalog["heralded cnot"].build_experiment()
>>> knill_experiment.min_detected_photons_filter(2)
>>> knill_experiment.with_input(pcvl.BasicState([0, 1, 0, 1]))
>>> knill_factory = pcvl.ExecutionFactory(computer, knill_experiment, max_shots_per_call=1_000_000)
>>>
>>> group = pcvl.ExecutionGroup("compare_knill_and_ralph_cnot")
>>> group.add(ralph_factory.sample_count, max_samples=10_000)
>>> group.add(knill_factory.sample_count, max_samples=10_000)

At this point the executions have only been prepared and saved locally. No computation has been launched. A later
script can load the group by name and run its unsent executions sequentially:

>>> import perceval as pcvl
>>>
>>> group = pcvl.ExecutionGroup("compare_knill_and_ralph_cnot")
>>> # Starts the computer of the first execution - Assume the same computer is used for all executions
>>> with pcvl.acquire(*group.list_unsent_computers()):
...     group.run_sequential(0)  # Launch the second execution after the first one finishes

Use :code:`group.run_parallel()` to run as many executions concurrently as their computers allow. The corresponding
:code:`group.rerun_failed_sequential(delay)` and :code:`group.rerun_failed_parallel()` methods rerun unsuccessful
executions.

.. note::
   The computers are not started and stopped automatically within the run.
   They should be started and stopped outside the run or rerun methods.

   The :meth:`encapsulate_manager_list()` utility function and the computer listing methods
   (:meth:`list_unsent_computers()`, :meth:`list_active_computers()` and :meth:`list_unsuccessful_computers()`)
   can be used for this purpose.

.. note::
   Like :ref:`RemoteComputer`, an :code:`ExecutionGroup` is not stored with your credentials.
   For automatic retrieval of an existing group, they must be inserted back using the correct :ref:`RemoteConfig`.

Executions can also be launched without waiting for them to finish:

>>> group.launch_async_executions(concurrent_execution_count=2)
>>> group.track_progress()  # Block and display progress until no execution remains active

.. note::
   :code:`launch_async_executions()` launches as many executions as the computers' available capacity permits, up to
   :code:`concurrent_execution_count` when it is provided. Local asynchronous executions that are still running when
   the Python process exits must be started again.

.. warning::
   A computer should not be stopped until all async executions associated to it have finished.
   Splitting the execution launch and the result retrieval into two scripts requires calling :meth:`start()`
   and :meth:`stop()` manually.

Finally, another script can load the group and retrieve its results in insertion order. An entry is :code:`None` when
the corresponding execution has not completed or its result is unavailable.

>>> import perceval as pcvl
>>>
>>> group = pcvl.ExecutionGroup("compare_knill_and_ralph_cnot")
>>> results = group.get_results()
>>> ralph_result = results[0]
>>> knill_result = results[1]

Class reference
---------------

.. autoclass:: perceval.runtime.execution_group.ExecutionGroup
   :members:
