RemoteProcessor
^^^^^^^^^^^^^^^

:code:`RemoteProcessor` class is the entry point for sending a computation on a remote platform (a simulator or a QPU).
`Quandela Cloud <https://cloud.quandela.com>`_ is a public cloud service with available QPUs and simulators.
An access token on the selected service is required to connect to a remote platform (e.g. an access token to Quandela
Cloud with rights is required to follow this tutorial: :ref:`Remote computing<Remote Computing>`).

Once you have created a token suiting your needs (it needs to be given the rights to run jobs on target platforms), you
may save it once and for all on your computer by using the :ref:`RemoteConfig`.

A token value can also be set to every :code:`RemoteProcessor` object

>>> remote_simulator = RemoteProcessor("platform:name", "YOUR_TOKEN")

For the rest of this page, let's assume a token is saved in your environment.

A given remote platform is only able to support a specific set of commands.
For instance, a real QPU is natively able to sample output detections, but not to compute probabilities of output states
versus an input state. Hardware constraints might also enforce the coincidence counting type, or the type of detection
(threshold detection or photon number resolving).

When creating a :code:`RemoteProcessor`, you can query its capabilities

>>> remote_simulator = RemoteProcessor("qpu:belenos")
>>> print(remote_simulator.available_commands)
['sample_count', 'samples']

This means `qpu:belenos` is only able to natively answer to `sample_count` and `samples` commands (i.e. return the
results of a sampling task).

Creation
--------

:code:`RemoteProcessor` has the same API and fills the same role as a local :ref:`Processor`
but are executed remotely by a Cloud platform (a real QPU or an online simulator).

RemoteProcessors are created slightly differently than normal Processors.

First, they require connexion information to a given Cloud provider:
  * A token (or API key) being the credentials to authenticate the user remotely.
  * A URL to the Cloud API
  * Optionally, a proxy configuration

All these information are used to create a :ref:`RPCHandler` which could be passed instead.
Also, these info can be saved in your local computer persistent :ref:`RemoteConfig`.

In terms of circuit initialisation, here are the specifics:

>>> rp = pcvl.RemoteProcessor("sim:slos", token=..., m=3, noise=pcvl.NoiseModel(0.9))  # m is an optional kwarg here

If :code:`m` is not specified, it is inferred from the first added component.
They can also be created by converting a local Processor, keeping all defined objects (input state, filter, ports...).

>>> rp = pcvl.RemoteProcessor.from_local_processor(p, "sim:slos", token=...)

From there, all composition rules are the same, and local processors can be added to remote processors.

Input state
-----------

Only non-polarized BasicState and LogicalState input are accepted for RemoteProcessors.

Computation
-----------

The only way to compute with a RemoteProcessor is to use it in a Quantum Algorithm.

Misceallenous
-------------

Some platforms expose specs that must be fulfilled in order for a Job to be able to be completed.
These include (but are not limited to) the number of photons, the number of modes, the number of photons per mode...
They can be retrieved using the property :code:`rp.specs` or :code:`rp.constraints`

The performances of the source can also be retrieved using the property :code:`rp.performance`.

The needed resources in terms of samples or shots can be estimated by a RemoteProcessor

>>> rp.estimate_required_shots(nb_samples=10000)

Note that this uses a partially noisy local simulation, so it can be expensive to compute.

.. autoclass:: perceval.runtime.remote_processor.RemoteProcessor
   :members:
