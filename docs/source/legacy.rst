Legacy
======

Perceval has evolved quickly from the initial release, some evolution are introducing breaking changes for existing code.
While we are trying hard to avoid unncessary API changes, some of necessary to bring new features and keep a consistent
code base.

This section lists the major breaking changes introduced.

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

Perceval is getting more and more Processor-centric as we implement more features. The Processor class has got some serious refactoring.
You may find examples of Processor created from scratch in perceval.components.core_catalog content
You may use several processors / circuits and compose them : a good example is the QiskitConvert convert method implementation

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
  output can be modelled with :code:`losses` paramater.

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
