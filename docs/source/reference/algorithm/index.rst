quantum algorithms
^^^^^^^^^^^^^^^^^^

The :code:`perceval.algorithm` package contains the code for several simple and generic quantum algorithms.
It provides a :ref:`Processor`-centric syntax to run an algorithm locally or remotely, on a simulator or an actual QPU.

All algorithms take either a local or a remote processor as parameter, in order to perform a task. :ref:`Processor`
runs simulations on a local computer while a :ref:`RemoteProcessor` turns Perceval into the client of a remote service
such as `Quandela Cloud <https://cloud.quandela.com>`_, and the computation is performed remotely, on the selected
platform.

However, for user experience, an algorithm has the same behavior be it run locally or remotely: every call to an
algorithm command returns a :ref:`Job` object, hiding this complexity.

>>> local_p = pcvl.Processor("CliffordClifford2017", pcvl.BS())
>>> local_p.with_input(pcvl.BasicState('|1,1>'))
>>> sampler = pcvl.algorithm.Sampler(local_p)
>>> local_sample_job = sampler.sample_count

Here, the computation has not started yet, but it's been prepared in :code:`local_sample_job` to run locally.

.. toctree::
   sampler
   analyzer
   tomography

Samples of interest vs Shots
============================

On a QPU, the acquisition is measured in **shots**. A shot is any coincidence with at least 1 detected photon.
Shots act as credits on the Cloud services. Users have to set a maximum shots value they are willing to use for any
given task.

>>> remote_p = pcvl.RemoteProcessor("sim:sampling")
>>> remote_p.set_circuit(pcvl.BS())
>>> remote_p.with_input(pcvl.BasicState('|1,1>'))
>>> sampler = pcvl.algorithm.Sampler(remote_p, max_shots_per_call=500_000)
>>> remote_sample_job = sampler.sample_count

Here, the computation was set-up to run on `sim:sampling` platform when :code:`remote_sample_job` is executed.

For more information about the shots and shots/samples ratio estimate, please read
:ref:`Remote computing on Quandela Cloud`.
