Legacy
======

While, with its latest versions, Perceval tends to stabilise its public API, some changes may break existing user code.

This section lists the major breaking changes.

Breaking changes in Perceval 1.0
--------------------------------

FockState was split in three different classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To achieve better optimisation in noisy simulation and to clarify the intent of different states usage, it has been
decided to get rid of the former generic :code:`FockState` that could hold richly annotated photons as well as just a
plain perfect state.

Definition of the new classes
.............................

* :code:`FockState`: A light-weight object only containing photon positions in mode (e.g. :code:`|1,0,1>`). Can be used to
  represent detections.
* :code:`NoisyFockState`: A collection of indistinguishable photon groups, that are totally distinguishable. The
  distinguishability index is an integer and is referred to as the `noise tag` (e.g. :code:`|{0},{1},{0}{2}>` contains
  three groups of indistinguishable photons tagged 0, 1 and 2).
* :code:`AnnotatedBasicState`: Replace the previous :code:`FockState` by allowing rich annotations, having one or more
  string types, each having a complex number for value. This enables to accurately encode physical parameters and
  play with partial distinguishability (e.g. :code:`|{P:H,lambda:0.625},{P:V,lambda:0.618}>`). Please note that apart
  from polarisation, `Perceval` does not provide a generic algorithm to separate rich annotated states, and the user
  would have to write one.

Some calls will use or return only the type that makes sense (e.g. :code:`AnnotatedFockState::threshold_detection()`
always returns a :code:`FockState` as a detected state naturally loses all kinds of photon annotation.

.. note:: Note that arithmetic still works between states of different types. The result is the most complex type of
          both operands (e.g. :code:`NoisyFockState` ⊗ :code:`FockState` gives a :code:`NoisyFockState`).

Usage in Perceval
.................

The :code:`BasicState` class still exists and has the same responsibility as before: representing any non superposed
pure state. It can construct any of the forementioned state type from a string representation, of vectors of position,
and optionally noise tags or annotations.

Even though, `Perceval` code makes it so :code:`isinstance(any_fockstate, BasicState)` returns :code:`True`, the type
hinting of user code in an IDE could alert that the types do not match after the update.

.. note:: :code:`StateVector` (and therefore :code:`SVDistribution`) accept any of the three Fock state types as
  components.

Processor add with Component or Circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When adding a Circuit or a Component to a Processor on non-consecutive modes, a permutation was added so that we could
add the component to the Processor. The inverse permutation is now also added after the component so that the in-between
modes are not impacted by the addition, similarly to what was already done when adding a Processor to a Processor.

BSDistribution and SVDistribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These classes have been moved to Exqalibur with a C++ implementation.
As such, they are no longer Python dictionaries and may not support some advanced dict features.
This has several consequences:

- You can no longer instantiate :code:`BSDistribution` or :code:`SVDistribution` using a dictionary with mixed type keys,
  nor with non-BasicState or non-StateVector keys.
- :code:`BSDistribution` and :code:`SVDistribution` can no longer be compared to a regular :code:`dict` (for example by using :code:`==`).
- The order of insertion is no longer preserved.
- :code:`keys()` and :code:`values()` methods now return an iterator, so methods like :code:`len` no longer work on
  their result.

Also, note that:

- Inserting a :code:`StateVector` in :code:`SVDistribution` no longer normalises it.
- Using the tensor product with an empty distribution now always returns an empty distribution.
  To keep the same behaviour as before (the result was the non-empty distribution), one would have to
  replace the empty distribution by a distribution containing a void state (:code:`BSDistribution(BasicState())`) for
  tensor product or a 0-photon state (:code:`BSDistribution(BasicState(m))`) for a merge.

StateVector
^^^^^^^^^^^

The method :code:`StateVector.keys()` now returns an iterator on the keys instead of a BSSamples.
This avoids doing unnecessary copy.

Please note that due to this change:

- Keys must now be copied before being modified when iterating on :code:`StateVector.keys()`.
- :code:`StateVector.keys()` no longer has list methods such as :code:`len`, :code:`__getitem__`...

Removal of deprecated methods and classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following methods and classes have been removed or definitely modified as they were deprecated:

- :code:`TokenProvider` (deprecated since 0.13, replaced by :code:`RemoteConfig`)
- :code:`AProbAmpliBackend` (deprecated since 0.12, replaced by :code:`AStrongSimulationBackend`)
- :code:`postselect_independent` (deprecated since 0.12, replaced by :code:`PostSelect` method :code:`is_independent_with`)
- The :code:`n` parameter of SLOS backend (deprecated since 0.12, now automatically chosen when using :code:`set_input_state`)
- :code:`thresholded_output` method of :code:`Processor` and :code:`RemoteProcessor`
  (deprecated since 0.12, replaced by adding several :code:`Detector.threshold()`)
- :code:`with_polarized_input` method of :code:`Processor` (because :code:`Processor.with_input` is now able to handle
  a polarized :code:`AnnotatedFockState` transparently)
- :code:`tensorproduct(states: list)` from :code:`perceval.utils` (due to tensor products being handled well by
  multiplication operators and specific methods - see :code:`BSDistribution.list_tensor_product`, for instance)
- :code:`JobGroup.list_existing()` has been renamed into :code:`JobGroup.list_locally_saved()`


Older changes
-------------

The documentation to update from an older legacy version to a more recent one can still be found
`here <https://perceval.quandela.net/docs/v0.13/legacy.html>`_.
