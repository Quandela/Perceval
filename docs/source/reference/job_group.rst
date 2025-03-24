JobGroup
========

The :code:`JobGroup` class is designed to help manage jobs client-side by storing them in named groups.
Large experiments can be easily cut in chunks and even ran during multiple days, over multiple Python sessions, the job
group will make sure all data can be retrieved from a single location.

.. warning::
   JobGroups store their job data in the persistent data directory.

   As these files can grow quite large, you will have to explicitely erase the ones you don't want to keep.

   JobGroup provides the following commands:

   * :code:`JobGroup.delete_job_group(name)`
   * :code:`JobGroup.delete_job_groups_date(del_before_date: datetime)`
   * :code:`JobGroup.delete_all_job_groups()`

Usage example
-------------

Here's an example creating a job group with two jobs, running the same acquisition on a post-processed vs an heralded
CNOT gate:

>>> import perceval as pcvl
>>> from perceval.algorithm import Sampler
>>>
>>> p_ralph = pcvl.RemoteProcessor("sim:altair")
>>> p_ralph.add(0, pcvl.catalog["postprocessed cnot"].build_processor())
>>> p_ralph.min_detected_photons_filter(2)
>>> p_ralph.with_input(pcvl.BasicState([0, 1, 0, 1]))
>>> sampler_ralph = Sampler(p_ralph, max_shots_per_call=1_000_000)
>>>
>>> p_knill = pcvl.RemoteProcessor("sim:altair")
>>> p_knill.add(0, pcvl.catalog["heralded cnot"].build_processor())
>>> p_knill.min_detected_photons_filter(2)
>>> p_knill.with_input(pcvl.BasicState([0, 1, 0, 1]))
>>> sampler_knill = Sampler(p_knill, max_shots_per_call=1_000_000)
>>>
>>> jg = pcvl.JobGroup("compare_knill_and_ralph_cnot")
>>> jg.add(sampler_ralph.sample_count, max_samples=10_000)
>>> jg.add(sampler_knill.sample_count, max_samples=10_000)

This first script only prepared the experiment, nothing was executed remotely. Before going on, it's important for a
user to know the details of their plan on the Cloud, for this will establish the number of job they can run
concurrently.

The job group supports executing jobs sequentially or in parallel and includes the ability to rerun
failed jobs, if needed.

The second script may be used exclusively to run jobs. It includes a built-in `tqdm` progress bar to
provide real-time updates on job execution. To run jobs sequentially with a given delay:

>>> import perceval as pcvl
>>>
>>> jg = pcvl.JobGroup("compare_knill_and_ralph_cnot")  # Loads prepared experiment data
>>> jg.run_sequential(0)  # Will send the 2nd job to the Cloud as soon as the first one is complete

Other methods - :code:`jg.run_parallel()`, :code:`jg.rerun_failed_parallel()`, and :code:`jg.rerun_failed_sequential(delay)`.

.. note:: The :code:`jg.run_parallel()` method tries to start all jobs in the group on Cloud. An error will occur if it exceeds the limitations defined by the pricing plan (see `Quandela Cloud <https://cloud.quandela.com/pricing>`_).

A third script can then prepared to analyze results:

>>> import perceval as pcvl
>>>
>>> jg = pcvl.JobGroup("compare_knill_and_ralph_cnot")
>>> results = jg.get_results()
>>> ralph_res = results[0]
>>> knill_res = results[1]
>>> perf_ratio = (ralph_res['physical_perf'] * ralph_res['logical_perf']) / (knill_res['physical_perf'] * knill_res['logical_perf'])
>>> print(f"Ralph CNOT is {perf_ratio} times better than Knill CNOT, but needs a measurement to work")
Ralph CNOT is 490.01059 times better than Knill CNOT, but needs a measurement to work

.. note:: If the connection token you use in a :code:`JobGroup` expires or gets revoked, said :code:`JobGroup` will not
          be usable anymore. Stay tuned for further improvements on this feature, fixing this issue.

Class reference
---------------

.. autoclass:: perceval.runtime.job_group.JobGroup
   :members:
