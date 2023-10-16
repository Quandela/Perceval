Legacy
======

Perceval has evolved quickly from the initial release, some evolution are introducing breaking changes for existing code.
While we are trying hard to avoid unnecessary API changes, some of necessary to bring new features and keep a consistent
code base.

This section lists the major breaking changes introduced.

Breaking changes in Perceval 0.10
---------------------------------
The main changes between versions 0.9 and 0.10 comes from the migration of the :code:`StateVector` code into our C++ library, Exqalibur.

StateVector
^^^^^^^^^^^

Iterate through a State Vector
++++++++++++++++++++++++++++++
State Vector is still a hash map (state, amplitude) but works a bit differently than a python dictionary.

State Vector keys, :code:`states`, are obtained with method :code:`keys`:

From version 0.9

>>> for state in state_vector:
>>>   assert state in state_vector

To version 0.10

>>> for state in state_vector.keys():
>>>   assert state in state_vector

State Vector items, :code:`(states, amplitude)`, are obtained by iterate directly through the state vector object:

From version 0.9

>>> for state, amplitude in state_vector.items():
>>>   assert state_vector[state] == amplitude

To version 0.10

>>> for state, amplitude in state_vector:
>>>   assert state_vector[state] == amplitude

Using :code:`numpy` scalars in StateVector arithmetic
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Exqalibur C++ package may interact badly with :code:`numpy` types depending on the operand order in some arithmetic operations.
Multiplying a :code:`numpy` scalar (left operand) with a StateVector (right operand) fails as :code:`numpy` has the priority on an operation it's unable to perform correctly.

From version 0.9

>>> import numpy
>>> sv1 = numpy.int16(4) * state_vector
>>> sv2 = state_vector * numpy.int16(4)
>>> assert sv1 == sv2


To version 0.10

>>> import numpy
>>> # sv1 = numpy.int16(4) * state_vector # will raise a ValueError
>>> sv2 = state_vector * numpy.int16(4)

.. note:: StateVector will interact badly with any :code:`numpy` scalar type

AnnotatedBasicState
^^^^^^^^^^^^^^^^^^^
:code:`AnnotatedBasicState` has been deprecated since Perceval 0.7.0, it's time to say goodbye.

See :ref:`AnnotatedBasicState was deprecated`

Breaking changes in Perceval 0.9
--------------------------------

The main changes between versions 0.8 and 0.9 come from the simulation rework. The simulation code was split in three
different layers: backends, simulators, processor. Some syntax was changed and your code might be broken. Note that if
you were using the :code:`Processor` layer to compute your simulations, the 0.8 syntax is still working with only two
deprecated methods (see :ref:`Simulation rework: processor`).

Simulation rework: backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `backend` classes were reworked in order to let them do what they do best: perform a perfect simulation with a pure
input fock state. The rest of the features (e.g. simulating a :code:`StateVector` input, with distinguishable photons,
etc.) were moved to a new class: the :ref:`Simulator`. Thus, former backend users should now preferably use the
:code:`Simulator`.

Backend syntax changes
++++++++++++++++++++++

If you still need to use the backend level, here are the following changes from version 0.8 to version 0.9:

From version 0.8

>>> backend_name = "SLOS"
>>> backend_type = pcvl.BackendFactory.get_backend(backend_name) # In 0.8, the BackendFactory would only be a mapping between a name and a type
>>> backend_obj = backend_type(circuit) # You'd have to instantiate the backend on the next line using the type
>>> pa = backend_obj.probampli(input_state, output_state) # You can then start simulating

To version 0.9

>>> backend_name = "SLOS"
>>> backend_obj = pcvl.BackendFactory.get_backend(backend_name) # In 0.9, the BackendFactory returns an empty backend instance
>>>
>>> from perceval.backends import SLOSBackend
>>> slos = SLOSBackend() # This is equivalent to using the BackendFactory
>>> slos_with_mask = SLOSBackend(mask=["0    0"], n=2) # You can also use the specifics of each backend when creating one
>>>
>>> slos.set_circuit(circuit) # Set a circuit first
>>> slos.set_input_state(input_state) # Input state has to be a Fock state (all indistinguishable photons)
>>> pa = slos.prob_amplitude(output_state) # Then you can start simulating

.. note:: As all simulation methods signature changed slightly, their name was changed too (e.g. :code:`probampli` to
   :code:`prob_amplitude`) in order to get an error message as soon as possible in your script. In API-break cases, it's
   better to get an error than a seemingly working code with an unexpected behavior!

.. note:: Backends are more specialized than before. For instance, :code:`sample()` cannot be called on `SLOS` and `Naive`
   anymore because they are natively probability amplitude computing backend. They however offer a way to compute the
   whole output probability distribution (:code:`prob_distribution()` method) from which it is possible to sample. On a
   similar note, `Clifford & Clifford` backend is only capable of sampling (its native simulation method).

How to use the simulator layer
++++++++++++++++++++++++++++++

The :code:`Simulator` is a versatile class which can simulate state evolution and sampling, using any of the probability
amplitude capable backend for its computations.

>>> from perceval.simulators import Simulator
>>> from perceval.backends import SLOSBackend
>>>
>>> simulator = Simulator(SLOSBackend()) # Initialize a simulator instance with a backend object
>>> simulator.set_circuit(circuit)
>>> # Here input state can be a BasicState or a StateVector, with or without photon annotations
>>> pa = simulator.prob_amplitude(input_state, output_state)

The :code:`Simulator` is also optimized to simulate a whole input distribution in one pass

>>> from perceval.components import Source
>>> from perceval.utils import BasicState
>>>
>>> # A simple example with a source-generated input distribution
>>> source = Source(losses=0.85, indistinguishability=0.9)
>>> input_distribution = source.generate_distribution(expected_input=BasicState([1, 0, 1, 0]))
{
  |0,0,0,0>: 0.7224999999999999
  |0,0,{_:0},0>: 0.1275
  |{_:0},0,0,0>: 0.1275
  |{_:0},0,{_:0},0>: 0.020250000000000004
  |{_:0},0,{_:1},0>: 0.002250000000000002
}
>>> simulator.set_min_detected_photon_filter(1)
>>> probs = simulator.probs_svd(input_distribution)
>>> print("physical performance:", probs["physical_perf"])
>>> print("output distribution:", probs["results"])
physical performance: 0.2775000000000001
output distribution: {
  |0,1,0,0>: 0.1456843866834125
  |0,0,1,0>: 0.1456843866834125
  |0,0,0,1>: 0.22972972972972971
  |1,0,0,0>: 0.39782041582236416
  |1,1,0,0>: 0.017550900698045487
  |1,0,1,0>: 0.017550900698045487
  |1,0,0,1>: 0.03510180139609097
  |0,2,0,0>: 0.00258340109361355
  |0,1,0,1>: 0.0027193695722247894
  |0,0,2,0>: 0.00258340109361355
  |0,0,1,1>: 0.0027193695722247894
  |0,1,1,0>: 0.00027193695722247914
}

See :ref:`Simulator` for the list of available simulation methods.

Simulation rework: processor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`Processor` can be used exactly as in version 0.8. However, please note that :code:`set_postprocess` and
:code:`clear_postprocess` methods have been deprecated in favor of :code:`set_postselection` and
:code:`clear_postselection`.

:code:`set_postselection` is more restrictive as it only allows :ref:`PostSelect` objects allowing Perceval to get rid
of Python free functions / lambdas.
We suggest you update your existing code base which is using :code:`set_postprocess` with Python functions as it will be
removed in an upcoming release without further notice.

See also: :ref:`PostSelect` code reference


Breaking changes in Perceval 0.8
--------------------------------

:code:`Processors.mode_post_selection` changes to :code:`min_detected_photons_filter`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Perceval 0.7, you could filter results by setting a minimum number of threshold detector "clicks" (which was
translated, in simulators, to the number of modes with at least one photon)

>>> import perceval as pcvl
>>> p = pcvl.Processor("SLOS", 8, pcvl.Source(emission_probability=.8))
>>> p.with_input(pcvl.BasicState([1, 0, 1, 0, 0, 0, 0, 0]))
>>> p.mode_post_selection(2)  # In Perceval 0.7, Processor p would reject results with less than 2 modes with detections

Even though this filtering works well with QPU simulators and actual QPU acquisitions, it implied that more theoretical
simulations was impacted by a threshold detection rule when they use perfect detectors. In this case, you could retrieve
unexpected results.

Perceval introduces :code:`min_detected_photons_filter` to improve its behavior. Updating to Perceval 0.8 and using
:code:`min_detected_photons_filter` as you would have used :code:`mode_post_selection`, will not change results
for threshold detections, and will improve them for perfect simulations (less states will be rejected, improving
*physical performance*).

>>> p.min_detected_photons_filter(2)  # In Perceval 0.8, the new filter rejects states based on photon count


Breaking changes in Perceval 0.7
--------------------------------

:code:`lib.phys` and :code:`lib.symb` have been removed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base components, originally duplicated in the two libraries were merged in two modules :code:`perceval.components.unitary_components` and :code:`perceval.components.non_unitary_components`.
One direct benefit of this change is that the beam splitter definition is now the same (see :ref:`BS conventions`), and does not depend on how it renders (see :ref:`Display components`).

>>> import perceval as pcvl
>>> from perceval.components.unitary_components import PS, BS, PERM
>>> import numpy as np
>>>
>>> c = pcvl.Circuit(2) // PS(np.pi) // BS() // PERM([1, 0]) // (1, PS(np.pi))

Display components
^^^^^^^^^^^^^^^^^^

Initially, use of `lib.symb` or `lib.phys` was deciding how the circuit was displayed.
Now, a skin system is available to use whichever representation you want.

>>> import perceval as pcvl
>>> from perceval.rendering import SymbSkin
>>>
>>> pcvl.pdisplay(c)  # defaults to PhysSkin, similar to lib.phys
>>> pcvl.pdisplay(c, skin=SymbSkin())  # Renders using SymbSkin, similar to lib.symb

see :ref:`Circuit Rendering` for more details.

BS conventions
^^^^^^^^^^^^^^

`lib.phys.BS` used a different convention from `lib.symb.BS`. After merging both libs, only one BS class remains,
handling 3 different conventions suited to any need. See :ref:`Beam splitter` for details.

>>> from perceval.components.base_components import BS, BSConvention
>>>
>>> bs = BS()  # Defaults to Rx convention. Ideally, in an upcoming Perceval release, the default could be changed in a persistent user config.
>>> BS.H() == BS(convention=BSConvention.H)  # Both syntaxes give the same result.
>>> BS.Ry() == BS(convention=BSConvention.Ry)  # Same

This new BS class handles only `theta` (instead of a mutually exclusive `theta` or `R`) which is used differently from before:
Half of theta is used when computing the unitary matrix (i.e. `cos(theta/2)` now, `cos(theta)` before).

Also, the new BS can be configured with 4 phases, one on each mode (`phi_tl`, `phi_tr`, `phi_bl`, `phi_br`) corresponding respectively to top left, top right, bottom left and bottom right arms of the beam splitter.

There is no direct conversion from former symb.BS or phys.BS.

* BS conventions - existing code:

In all the existing code base, :code:`phys.BS` were replaced by :code:`BS.H` and :code:`symb.BS` by :code:`BS.Rx` which have the same unitary matrices when no phase are applied to them.

Create a backend instance
^^^^^^^^^^^^^^^^^^^^^^^^^

Originally, you would call

>>> backend_type = BackendFactory().get_backend(backend_name)  # For instance backend_name = "SLOS"
>>> simu_backend = backend_type(circuit)

While this is still functional, this can also be misleading. Indeed, simulation backends can provide features that you
cannot measure with actual QPU - typically the probability amplitude. This is good for developing theoretical algorithms
but using these will not port to actual QPUs. We recommend using the class :class:`Processor` by default.

AnnotatedBasicState was deprecated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please use BasicState instead which holds every feature previously held by AnnotatedBasicState

Processor definition and composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perceval is getting more and more Processor-centric as we implement more features. The Processor class has got some
serious refactoring.
You may find examples of Processor created from scratch in perceval.components.core_catalog content.
You may use several processors / circuits and compose them : a good example is the QiskitConvert convert method
implementation.

Access to circuit parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It was possible to access a named parameters on a circuit using :code:`[]` notation:

>>> c['phi']

This has been replaced by explicit use of `params` accessor:

>>> c.param('phi')

The `__getitem__` notation is now used to access components in a circuit (see :ref:`Accessing components in a circuit`).

New Source in Perceval 0.7.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new source model has been introduced in Perceval 0.7.3. The `Source` class initialization parameters have changed
and imperfect simulated sources will return results closer to the actual photonic sources which are used in the QPUs.
Backward compatibility with pre-0.7.3 sources is broken.

* :code:`brightness` was replaced by :code:`emission_probability`. Balanced losses from the source output to the circuit
  output can be modelled with :code:`losses` parameter.

* :code:`purity` and :code:`purity_model` were respectively replaced by :code:`multiphoton_component` and
  :code:`multiphoton_model`.
  :code:`purity` represented the ratio of time when photon is emitted alone whereas :code:`multiphoton_component` is
  the :math:`g^{(2)}`. There is no direct conversion from the former purity to :math:`g^{(2)}`, note however that the
  greater the purity, the lower the :math:`g^{(2)}`.

* The default distinguishability of multiple emitted photons changed from `indistinguishable` to `distinguishable`.

>>> source = pcvl.Source(brightness=0.3, purity=0.95, purity_model="distinguishable")

can be changed to (without returning the same results):

>>> source = pcvl.Source(emission_probability=0.3, multiphoton_component=0.05)

See :ref:`Source` class reference for more information.
