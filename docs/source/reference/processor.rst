Processor and RemoteProcessor
=============================

Processors and RemoteProcessors expose the same behaviour in many ways, and most of the time,
when a Processor is needed, it can be replaced with a RemoteProcessor

Creating Processors
^^^^^^^^^^^^^^^^^^^

A processor describes an optical experiment with a computation method (backend).

>>> import perceval as pcvl
>>> p = pcvl.Processor("SLOS", 4, name="my proc")  # Creates a 4-modes Processor named "my proc" that will be simulated using SLOS
>>> p.m
4

A processor can be created empty with a given number of modes, or using a circuit.

>>> p = pcvl.Processor("SLOS", pcvl.BS())  # Creates a 2-modes Processor with a single beam splitter as component

Method :code:`add`
------------------

Just like circuits, components, circuits and processors can be added to processors using the :code:`add` method
(note however that :code:`//` doesn't work for processors).
When adding a Processor, only the backend from the left processor is kept.

>>> p.add(0, pcvl.PS(3.14))  # Add a phase shifter on mode 0

However, unlike circuits, non-linear components can also be added to Processors

>>> p.add(1, pcvl.TD(1))  # Adds a time-delay on mode 1

Secondly, the mode on which a component is added has a few more options than just an integer.
One can use a list or a dict of integers to map the output of the current processor to the input of the added component.
If the left processor has output ports and the right processor has input ports, it can also be a dict describing the port names.

This adds up a permutation before inserting the new component, and its inverse at the end (so modes don't move when doing this).
Note however that when adding a processor with asymmetrical heralds (see below),
the inverse permutation is not added since it doesn't exist, so modes might move (check with a :code:`pdisplay`).

>>> p.add([1, 0], pcvl.BS(theta=0.7))  # Left mode 1 will connect to right mode 0, and left mode 0 will connect to right mode 1
>>> p.add({1: 0, 0: 1}, pcvl.BS(theta=0.7))  # Same as above

Detectors can also be added to a Processor using the same syntax

>>> p.add(0, pcvl.Detector.threshold())

Once a :code:`Detector` has been added, no optical component can be added anymore on this mode.

Setting an input state
----------------------

Before a Processor can be simulated, an input state must be provided.

>>> p.with_input(pcvl.BasicState([1, 0]))

The input state can be:

- A :code:`BasicState`, in which case the noise from the noise model is computed.
- A :code:`LogicalState` if ports have been defined, in which case the noise is computed.
- A :code:`StateVector`
- A :code:`SVDistribution`

If a :code:`BasicState` has polarization, the method to use is :code:`p.with_polarized_input`,
and no noise from the source will be applied.

Noise model
-----------

Processors can be given noise model to apply noise both on the source and on the components

This noise model can be given at instantiation

>>> p = pcvl.Processor("SLOS", 4, pcvl.NoiseModel(brightness=0.9, phase_error=0.01))

or changed later during the life of a processor

>>> p.noise = pcvl.NoiseModel(brightness=0.8, g2=0.03)

Min photons filter
------------------

A threshold on the number of detected photons can be set so outputs having less than this number of photons are filtered out.
This has an impact on the perfs of the Processor.

>>> p.min_detected_photons_filter(3)  # Outputs will all have at least 3 photons

Ports
-----

Once a Processor has been defined in terms of components, one can add ports and heralds to it.
If a port spans over several modes, the specified mode is considered to be the upper one.

>>> p.add_port(0, pcvl.Port(pcvl.Encoding.DUAL_RAIL, "qubit0"))  # Adds a dual rail port on modes 0 and 1 on both sides
>>> p.remove_port(0)
>>> p.add_port(0, pcvl.Port(pcvl.Encoding.DUAL_RAIL, "qubit0"), location=pcvl.PortLocation.INPUT)  # Add the port on the left of the processor

Ports have three main purposes:

- Showing the circuit's logic in display
- Composing processors using ports
- Setting an input state

>>> p.with_input(pcvl.LogicalState([0]))  # Equivalent to BasicState([1, 0]) for a dual rail. Adapts automatically to the ports

Heralds
-------

Heralds are a special kind of ports that act as modes that the user "doesn't want to see".
Note that ports and heralds are mutually exclusive mode-wise.
At the input, they declare a number of photon in a mode that the user won't have to specify when using :code:`with_input`.

>>> p = pcvl.Processor("SLOS", pcvl.BS())
>>> p.add_herald(0, 1, location=pcvl.PortLocation.INPUT)  # Add an herald of value 1 on input mode 0
>>> p.with_input(pcvl.BasicState([1]))  # Only one mode
>>> p.m_in
1
>>> p.heralds_in
{0: 1}

At the output, they will automatically filter states so only states matching the given number of photons will be selected.
They also remove these modes from the resulting BasicStates.
This filtering has an impact on the perf of the processor.

>>> p = pcvl.Processor("SLOS", pcvl.BS())
>>> p.add_herald(0, 1, location=pcvl.PortLocation.OUTPUT)  # Output will have only one mode
>>> p.m
1
>>> p.circuit_size  # Real size of the circuit
2
>>> p.heralds
{0: 1}

Heralded output modes can still be seen using :code:`p.keep_heralds(True)`.
In this case, heralded modes can still be removed afterward using :code:`state = p.remove_heralded_modes(state)`
Heralds at output are independent from the min detected photons filter, as the filter looks only at non-heralded modes.

>>> p.min_detected_photons_filter(2)
>>> p.add_herald(0, 1)  # There will actually be at least 3 photons

A :code:`Processor` that has at least one mode that defines an herald only at input or output is considered asymmetrical.
By default, heralds are added on both sides, so Processors are kept symmetrical.

When composing processors, the processors are considered to have :code:`m` output modes and :code:`m_in` input modes.
Heralds are considered to be outside the processors. Thus, they can be moved to new modes to keep a good structure.
Most 2-qubit gates from the catalog are symmetrical processors that use heralds.

When composing with a symmetrical processor, the inverse permutation is added at the right to keep the order of the modes.
This is not the case when composing with an asymmetric processor.

>>> from perceval import catalog
>>> p = pcvl.Processor("SLOS", 4)
>>> cnot = catalog["postprocessed cnot"].build_processor()
>>> cnot.m
4
>>> cnot.circuit_size
6
>>> p.add(0, cnot)  # Works despite the cnot having 6 modes
>>> p.circuit_size  # p is now bigger due to the added heralds from cnot
6
>>> p.heralds
{4: 0, 5: 0}

PostSelect
----------

A post-selection method can be added to a Processor to filter only states matching it.

>>> p.set_postselection(pcvl.PostSelect("[0, 1] == 2"))
>>> p.post_select_fn
[0, 1] == 2

When composing, the modes are swapped to match the new modes of the composition.
Also, it is not allowed to add something to an experiment that has a post-selection
if the modes overlap one of the nodes of the post-selection (they should be entirely included or disjoint)

If the user knows what they are doing,
they can remove the post-selection using :code:`p.clear_postselection()` then apply it again.

Computation
^^^^^^^^^^^

Depending on the backend that was specified at the beginning, a Processor can perform probability computation or sampling.

>>> p = pcvl.Processor("SLOS", 4)
>>> p.available_commands
["probs"]
>>> p = pcvl.Processor("Clifford", 3)
>>> p.available_commands
["samples"]

Any of the available methods can be used to compute the results for this processor, taking into account the components,
the input state, the noise, the heralds, the post-selection...

In any case, the results is a dict containing the results in the field "results"
and some performance score corresponding to the probability of getting a selected state.

>>> p = pcvl.Processor("SLOS", pcvl.BS())
>>> p.with_input(pcvl.BasicState([1, 0]))
>>> p.probs()["results"]
BSDistribution(float, {|1,0>: 0.5, |0,1>: 0.5})

.. autoclass:: perceval.components.processor.Processor
   :members:
   :inherited-members:

RemoteProcessor specificities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creation
--------

RemoteProcessors describe the same kind of experiments than regular Processors,
but are executed remotely by a cloud platform (possibly by a real QPU).

RemoteProcessors are created slightly differently than normal Processors.

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

Primitives to obtain results (probs, samples) can't be used with RemoteProcessor.
Instead, the user must use the Sampler algorithm to get what they want.

Misc
----

Some platforms expose specs that must be fulfilled in order for a Job to be able to be completed.
These include (but are not limited to) the number of photons, the number of modes, the number of photons per mode...
They can be retrieved using the property :code:`rp.specs` or :code:`rp.constraints`

The performances of the source can also be retrieved using the property :code:`rp.performance`.

The needed resources in terms of samples or shots can be estimated by a RemoteProcessor

>>> rp.estimate_required_shots(nb_samples = 10000)

Note that this uses a partially noisy local simulation, so it can be expensive to compute.

.. autoclass:: perceval.runtime.remote_processor.RemoteProcessor
   :members:
