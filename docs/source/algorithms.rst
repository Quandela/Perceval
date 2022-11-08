Run quantum algorithms
======================

Perceval provides a processor-centric syntax to run an algorithm locally or remotely, on a simulator or an actual QPU.

Build a processor
------------------

A :ref:`Processor` is a composite element aiming at simulating an actual QPU, in real world conditions.

* It holds a single photon :ref:`Source`
* It is a composition of unitary circuits and non-unitary components
* Input and output ports can be defined, with encoding semantics
* Logical post-processing can be set-up through heralded modes (ancillas) and a final post-selection function
* It contains the means of simulating the setup it describes with one of :ref:`The Backends` which are provided

Create a processor from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, let's create locally a heralded CNOT gate from its circuit:

>>> import perceval as pcvl
>>> from perceval.components import BS, PERM, Port, Encoding
>>> c_hcnot = (pcvl.Circuit(8, name="Heralded CNOT")
...            .add((0, 1, 2), PERM([1, 2, 0]))
...            .add((4, 5), BS.H())
...            .add((5, 6, 7), PERM([1, 2, 0]))
...            .add((3, 4), BS.H())
...            .add((2, 3), BS.H(theta=self.theta1, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
...            .add((4, 5), BS.H(theta=self.theta1))
...            .add((3, 4), BS.H())
...            .add((5, 6, 7), PERM([2, 1, 0]))
...            .add((1, 2), PERM([1, 0]))
...            .add((2, 3), BS.H(theta=self.theta2))
...            .add((4, 5), BS.H(theta=self.theta2, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
...            .add((5, 6), PERM([1, 0]))
...            .add((4, 5), BS.H())
...            .add((4, 5), PERM([1, 0]))
...            .add((0, 1, 2), PERM([2, 1, 0])))
>>> processor_hcnot = pcvl.Processor("SLOS", c_hcnot)
>>> processor_hcnot.add_herald(0, 0)
...                .add_herald(1, 1)
...                .add_port(2, Port(Encoding.DUAL_RAIL, 'data'))
...                .add_port(4, Port(Encoding.DUAL_RAIL, 'ctrl'))
...                .add_herald(6, 0)
...                .add_herald(7, 1)
>>> pcvl.pdisplay(processor_hcnot, recursive=False)

.. figure:: _static/img/heralded-cnot-processor.png
    :align: center

    Heralded CNOT gate rendering - heralded modes are not shown for readability

Processor composition
^^^^^^^^^^^^^^^^^^^^^

Processors can also be composed with one another. That's for example, how :ref:`Qiskit converter` outputs a complex
preconfigured processor from a gate-based circuit.

.. figure:: _static/img/complex-processor.png
    :align: center

    A processor composed of a Hadamard gate and two heralded CNOT gates.

Remote processors
^^^^^^^^^^^^^^^^^

RemoteProcessor class is the entry point for sending a computation on a remote platform (that can be a simulator or a
QPU). An access token on Quandela Cloud with rights to run on any existing platform is required to follow this tutorial:
:ref:`Remote computing with Perceval`

No processor is able to execute all types of command. For instance, a real QPU is natively able to sample output
detections, but not to compute probabilities of output states versus an input state.

When creating a RemoteProcessor, you can query its capabilities

>>> token_qcloud = 'YOUR_API_KEY'
>>> remote_simulator = RemoteProcessor("sim:ascella", token_qcloud)
>>> print(remote_simulator.available_commands)
['probs']

This means, `sim:ascella` is only able to answer to `probs` commands (i.e. compute the probability of all output states
given an input state).

Provided algorithms
-------------------

Algorithms provided with Perceval are available in the Python package `perceval.algorithm`. They can be as simple as
a `sampler` algorithm, as specific as `QRNG` (certified random number generator), which would work only on some
certified QPUs.

Algorithm interface
^^^^^^^^^^^^^^^^^^^

All algorithms take either a local or a remote processor as parameter, in order to work on it. A `Processor` runs
simulations on the local computer while a `RemoteProcessor` turns Perceval into a client of the Quandela Cloud server,
and the computation is performed on the selected platform.

However, for user experience, an algorithm has the same behavior be it run locally or remotely: every call to an
algorithm command returns a `Job` hiding this complexity.

>>> local_p = pcvl.Processor("CliffordClifford2017", pcvl.BS())
>>> local_p.with_input(pcvl.BasicState('|1,1>'))
>>> sampler = pcvl.algorithm.Sampler(local_p)
>>> local_job = sampler.sample_count(10000)

Here, the computation has not started yet, but it's been prepared in `local_job` to run locally.

>>> token_qcloud = 'YOUR_API_KEY'
>>> remote_p = pcvl.RemoteProcessor("sim:clifford", token_qcloud)
>>> remote_p.set_circuit(pcvl.BS())
>>> remote_p.with_input(pcvl.BasicState('|1,1>'))
>>> sampler = pcvl.algorithm.Sampler(remote_p)
>>> remote_job = sampler.sample_count(10000)

Here, the computation was set-up to run on `sim:clifford` platform when `remote_job` is executed.
