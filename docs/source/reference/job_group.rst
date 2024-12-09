JobGroup
========

The :code:`JobGroup` class is designed to help manage jobs client-side by storing them in named groups.
Large experiments can be easily cut in chunks and even ran during multiple days, over multiple Python sessions, the job
group will make sure all data can be retrieved from a single location.

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
>>> p_knill.min_detected_photons_filter(4)
>>> p_knill.with_input(pcvl.BasicState([0, 1, 0, 1]))
>>> sampler_knill = Sampler(p_knill, max_shots_per_call=1_000_000)
>>>
>>> jg = pcvl.JobGroup("compare_knill_and_ralph_cnot")
>>> jg.add(sampler_ralph.sample_count, max_samples=10_000)
>>> jg.add(sampler_knill.sample_count, max_samples=10_000)

This first script only prepared the experiment, nothing was executed remotely. Before going on, it's important for a
user to know the details of their plan on the Cloud, for this will establish the number of job they can run
concurrently.

The second script may only be used run the jobs:

>>> import perceval as pcvl
>>>
>>> jg = JobGroup("compare_knill_and_ralph_cnot")  # Loads prepared experiment data
>>> jg.run_sequential(0)  # Will send the 2nd job to the Cloud as soons as the first one is complete

A third script can then prepared to analyze results:

>>> import perceval as pcvl
>>>
>>> jg = JobGroup("compare_knill_and_ralph_cnot")
>>> results = jg.retrieve_results()
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
