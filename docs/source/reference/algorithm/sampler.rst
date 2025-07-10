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

When a `primitive` is not available on a processor, a :ref:`conversion` occurs automatically after the computation is complete.

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
>>> processor = pcvl.RemoteProcessor("qpu:altair", token_qcloud)
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

Sampler code reference
++++++++++++++++++++++

.. autoclass:: perceval.algorithm.sampler.Sampler
   :members:
