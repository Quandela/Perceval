Run quantum algorithms
======================

Perceval provides a processor-centric syntax to run an algorithm locally or remotely, on a simulator or an actual QPU.

Build a processor
-----------------

A :ref:`Processor` is a composite element aiming at simulating an actual QPU, in real world conditions.

* It holds a single photon :ref:`Source`
* It is a composition of unitary circuits and non-unitary components
* Input and output ports can be defined, with encoding semantics
* Logical post-processing can be set-up through heralded modes (ancillas) and a final post-selection function
* It contains the means of simulating the setup it describes with one of the provided :ref:`Computing Backends`

Create a local processor
^^^^^^^^^^^^^^^^^^^^^^^^

To create a local processor, you just need to start with a :ref:`Circuit`, provide a backend name, and either an
optional :ref:`Source` or a :code:`NoiseModel` with the following syntax:

>>> processor = pcvl.Processor("MY_BACKEND", my_circuit, source=my_source)
>>> processor = pcvl.Processor("MY_BACKEND", my_circuit, noise=my_noise_model)

If omitted, the source is a perfect single photon source.

The processor definition is then fine-tuned with ports (:code:`add_port`) and herald (:code:`add_herald`) definition as
in the following post-processed CNOT gate example:

>>> import perceval as pcvl
>>> from perceval.components import BS, PERM, Port
>>> from perceval.utils import Encoding
>>> theta_13 = BS.r_to_theta(1 / 3)
>>> c_cnot = (Circuit(6, name="PostProcessed CNOT")
...          .add(0, PERM([0, 2, 3, 4, 1]))
...          .add((0, 1), BS.H(theta_13))
...          .add((3, 4), BS.H())
...          .add((2, 3), PERM([1, 0]))
...          .add((2, 3), BS.H(theta_13))
...          .add((2, 3), PERM([1, 0]))
...          .add((4, 5), BS.H(theta_13))
...          .add((3, 4), BS.H())
...          .add(1, PERM([3, 0, 1, 2])))
>>> processor_cnot = pcvl.Processor("SLOS", c_cnot)
>>> processor_cnot.add_port(0, Port(Encoding.DUAL_RAIL, 'ctrl')) \
...               .add_port(2, Port(Encoding.DUAL_RAIL, 'data')) \
...               .add_herald(4, 0) \
...               .add_herald(5, 0)
>>> p.set_postselection(PostSelect("[0,1]==1 & [2,3]==1"))  # Add a post-selection checking for a logical output state
>>> pcvl.pdisplay(processor_cnot, recursive=True)

.. figure:: _static/img/postprocessed-cnot-processor.png
    :align: center

    Post-processed CNOT gate rendering - ancillary modes are not shown for readability

Processor composition
^^^^^^^^^^^^^^^^^^^^^

Processors can be composed with another processor. That is, for example, how the :ref:`Qiskit converter` outputs a
complex preconfigured processor from a gate-based circuit.

.. figure:: _static/img/complex-processor.png
    :align: center

    A processor composed of a Hadamard gate and two heralded CNOT gates.

Remote processors
^^^^^^^^^^^^^^^^^

:code:`RemoteProcessor` class is the entry point for sending a computation on a remote platform (a simulator or a QPU).
`Quandela Cloud <https://cloud.quandela.com>`_ is a public cloud service with available QPUs and simulators.
An access token on the selected service is required to connect to a remote platform (e.g. an access token to Quandela
Cloud with rights is required to follow this tutorial: :ref:`Remote computing on Quandela Cloud`).

Once you have created a token suiting your needs (it needs to be given the rights to run jobs on target platforms), you
may save it once and for all on your computer by running:

>>> pcvl.save_token('YOUR_TOKEN')

.. note:: We recommend you save your token only on a personal computer, not on shared/public ones.

A token value can also be set to every :code:`RemoteProcessor` object

>>> remote_simulator = RemoteProcessor("platform:name", "YOUR_TOKEN")

For the rest of this page, let's assume a token is saved in your environment.

A given remote platform is only able to support a specific set of commands.
For instance, a real QPU is natively able to sample output detections, but not to compute probabilities of output states
versus an input state. Hardware constraints might also enforce the coincidence counting type, or the type of detection
(threshold detection or photon number resolving).

When creating a :code:`RemoteProcessor`, you can query its capabilities

>>> remote_simulator = RemoteProcessor("qpu:ascella")
>>> print(remote_simulator.available_commands)
['sample_count', 'samples']

This means `sim:ascella` is only able to natively answer to `sample_count` and `samples` commands (i.e. return the
results of a sample acquisition task).

Work with algorithms
--------------------

All algorithms take either a local or a remote processor as parameter, in order to perform a task. A :code:`Processor`
runs simulations on a local computer while a :code:`RemoteProcessor` turns Perceval into the client of a remote service
such as `Quandela Cloud <https://cloud.quandela.com>`_, and the computation is performed remotely, on the selected platform.

However, for user experience, an algorithm has the same behavior be it run locally or remotely: every call to an
algorithm command returns a :code:`Job` object, hiding this complexity.

>>> local_p = pcvl.Processor("CliffordClifford2017", pcvl.BS())
>>> local_p.with_input(pcvl.BasicState('|1,1>'))
>>> sampler = pcvl.algorithm.Sampler(local_p)
>>> local_sample_job = sampler.sample_count

Here, the computation has not started yet, but it's been prepared in :code:`local_sample_job` to run locally.

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

Handle a Job object
^^^^^^^^^^^^^^^^^^^

Both :code:`LocalJob` and ``RemoteJob`` share the same interface.

* Execute a job synchronously

>>> args = [10_000]  # for instance, the expected sample count
>>> results = job.execute_sync(*args)  # Executes the job synchronously (blocks the execution until results are ready)
>>> results = job(*args)  # Same as above

* Execute a job asynchronously

>>> job.execute_async(*args)

This call is non-blocking, however results are not available when this line has finished executing. The job object
provides information on the progress.

>>> while not job.is_complete:  # Check if the job has finished running
...     print(job.status.progress)  # Progress is a float value between 0. and 1. representing a progress from 0 to 100%
...     time.sleep(1)
>>> if job.is_failed:  # Check if the job has failed
...     print(job.status.stop_message)  # If so, print the reason
>>> results = job.get_results()  # Retrieve the results if any

Typically, the results returned by an algorithm is a Python dictionary containing a ``'results'`` key, plus additional
data (performance scores, etc.).

* A job cancellation can be requested programmatically by the user

>>> job.cancel()  # Ask for job cancelation. The actual end of the execution may take some time

When a job is canceled, it may contain partial results. To retrieve them, call :meth:`get_results()`.

* A remote job can be resumed as following:

>>> remote_processor = pcvl.RemoteProcessor("any:platform")
>>> job = remote_processor.resume_job("job_id")  # You can find job IDs on Quandela Cloud's website
>>> print(job.id)  # The ID field is also available in every remote job object

Provided algorithms
-------------------

Algorithms provided with Perceval are available in the Python package ``perceval.algorithm``. They can perform as simple
tasks as the :code`Sampler`, or more complex computations. They're all meant to be generic and versatile.

Sampler
^^^^^^^

The :code:`Sampler` is the simplest algorithm provided, yet an important gateway to using processors.

All processors do not share the same capabilities. For instance, a QPU is able to sample, but not to sample output
probabilities given an input. The :code:`Sampler` allows users to call any of the three main `primitives` on any
processor:

>>> sampler = pcvl.algorithm.Sampler(processor)
>>> samples = sampler.samples(10000)  # Sampler exposes 'samples' primitive returning a list of ordered samples
>>> print(samples['results'])
[|0,1,0,1,0,0>, |0,1,0,0,1,0>, |0,2,0,0,0,0>, |0,0,0,1,0,0>, |0,1,0,1,0,0>, |0,1,0,1,0,0>, |0,1,1,0,0,0>, |0,1,0,1,0,0>, |0,1,1,0,0,0>, |0,1,0,1,0,0>, ... (size=10000)]
>>> sample_count = sampler.sample_count(10000)  # Sampler exposes 'sample_count' returning a dictionary {state: count}
>>> prob_dist = sampler.probs()  # Sampler exposes 'probs' returning a probability distribution of all possible output states

When a `primitive` is not available on a processor, a conversion occurs automatically after the computation is complete.

Batch jobs
++++++++++

The :code:`Sampler` can setup a batch of different sampling tasks within a single job. Such a job enables you to gain
some time (overhead of job management) as well as tidy up your job list, especially when running on the Quandela Cloud
(but it can still be used in a local simulation context).

The system relies on defining a circuit containing variable parameters, then with each iteration of the batch job,
you can set values for:

* The circuit `variable parameters` - each iteration must define a value for all variable parameters so that the circuit
  is fully defined,
* the `input state`,
* the `detected photons filter`.

>>> c = BS() // PS(phi=pcvl.P("my_phase")) // BS()  # Define a circuit containing "my_phase" variable
>>> processor = pcvl.RemoteProcessor("qpu:ascella", token_qcloud)
>>> processor.set_circuit(c)
>>> sampler = Sampler(processor)
>>> sampler.add_iteration(circuit_params={"my_phase": 0.1},
>>>                       input_state=BasicState([1, 1]),
>>>                       min_detected_photons=1)  # You can add a single iteration
>>> sampler.add_iteration_list([
>>>     {"circuit_params": {"my_phase": i/2},
>>>      "input_state": BasicState([1, 1]),
>>>      "min_detected_photons": 1
>>>     } for i in range(1, 6)
>>> ])  # Or you can add multiple iterations at once
>>> sample_count = sampler.sample_count(10000)

.. note:: As the same input state is used for all iterations, it could have been set once with
   :code:`processor.with_input` method and :code:`input_state` removed from every iteration definition.

This job will iterate over all the sampling parameters in a batch and return all the results at once.

>>> results_list = sample_count["results_list"]  # Note that all the results are stored in the "results_list" field
>>> for r in results_list:
>>>     print(r["iteration"]['circuit_params'])  # Iteration params are available along with the other result fields
>>>     print(r["results"])
{'my_phase': 0.1}
{
  |1,0>: 3735
  |0,1>: 3828
  |1,1>: 2437
}
{'my_phase': 0.5}
{
  |1,0>: 4103
  |0,1>: 3972
  |1,1>: 1925
}
{'my_phase': 1.0}
{
  |1,0>: 4650
  |0,1>: 4607
  |1,1>: 743
}
{'my_phase': 1.5}
{
  |1,0>: 5028
  |0,1>: 4959
  |1,1>: 13
}
{'my_phase': 2.0}
{
  |1,0>: 4760
  |0,1>: 4788
  |1,1>: 452
}
{'my_phase': 2.5}
{
  |1,0>: 4155
  |0,1>: 4252
  |1,1>: 1593
}

Analyzer
^^^^^^^^

The ``Analyzer`` algorithm aims at testing a processor, computing a probability table between input states and expected
outputs, a performance score and an error rate.

See usage in :ref:`Ralph CNOT Gate`
