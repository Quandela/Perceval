Legacy
======

While, with its latest versions, Perceval tends to stabilise its public API, some changes may break existing user code.

This section lists the major breaking changes.

Breaking changes in Perceval 1.3
--------------------------------

Global workflow change
^^^^^^^^^^^^^^^^^^^^^^

The top layers of perceval (starting from :code:`Processor`) have been completely re-written to support new features,
and make the whole process more versatile and easy to write.

The former runtime workflow is still available from :code:`perceval.runtime.legacy`, and should continue to work,
but will not offer all the possibilities that the new workflow offers.
It is strongly recommended to swap to the new workflow as soon as possible.

In the new workflow, an :code:`Experiment` describes what is run, a computer describes where it is run, and an
:code:`Execution` controls the run. The following table shows the equivalent code for every class exported by the
legacy runtime package. Most of the new classes have a one-to-one relationship with the legacy workflow,
and most of their methods share the same name and signature to help the transition, but there are still a few exceptions.
Also, the two workflows can't be mixed together.

The snippets use the following imports (and placeholder values :code:`TOKEN` and :code:`PROJECT_ID` for remote calls):

.. code-block:: python

   import perceval as pcvl
   from perceval.algorithm import Sampler, Analyzer
   from perceval.runtime import legacy

.. list-table:: Legacy runtime migration examples
   :header-rows: 1

   * - Old class
     - Old workflow
     - New workflow
   * - :code:`Processor`
     - .. code-block:: python

          processor = legacy.Processor("SLOS", pcvl.BS(), noise)
          processor.with_input(pcvl.FockState([1, 1]))
     - .. code-block:: python

          experiment = pcvl.Experiment(pcvl.BS())
          experiment.with_input(pcvl.FockState([1, 1]))
          computer = pcvl.SimulatedComputer("SLOS")
          computer.noise = noise
   * - :code:`RemoteProcessor`
     - .. code-block:: python

          processor = legacy.RemoteProcessor("sim:slos", TOKEN)
     - .. code-block:: python

          communication_layer = pcvl.QuandelaCommunicationLayer("sim:slos", TOKEN)
          computer = pcvl.RemoteComputer(communication_layer)
   * - :code:`Sampler`
     - .. code-block:: python

          sampler = Sampler(processor)
     - .. code-block:: python

          factory = pcvl.ExecutionFactory(computer, experiment)
   * - :code:`ScalewaySession`

       :code:`KipuSession`

       :code:`QuandelaSession`
     - .. code-block:: python

          session = pcvl.ScalewaySession(
            "EMU-SAMPLING-L4", PROJECT_ID, TOKEN)
          with session:
              processor = session.build_remote_processor()
              # Run jobs
     - .. code-block:: python

          communication_layer = pcvl.ScalewayCommunicationLayer(
            "EMU-SAMPLING-L4", PROJECT_ID, TOKEN)
          computer = pcvl.RemoteComputer(communication_layer)
          with computer.acquire():
              # Run executions
   * - :code:`Job`

       :code:`LocalJob`

       :code:`RemoteJob`
     - .. code-block:: python

          sampler = Sampler(processor)
          job: legacy.Job = sampler.probs
          results = job(n_samples=1000)
          job = sampler.probs.execute_async(n_samples = 5000)
     - .. code-block:: python

          factory = pcvl.ExecutionFactory(computer, experiment)
          execution: pcvl.Execution = factory.probs
          with computer.acquire():
              results = execution(n_samples=1000)
              execution = factory.probs.execute_async(n_samples = 5000)
   * - :code:`JobGroup`
     - .. code-block:: python

          group = legacy.JobGroup("my group")
          group.add(Sampler(processor).samples,
                    max_samples=1_000)
          group.launch_async_jobs()
     - .. code-block:: python

          group = pcvl.ExecutionGroup("my group")
          factory = pcvl.ExecutionFactory(computer, experiment)
          group.add(factory.samples, max_samples=1_000)
          with computer.acquire():
             group.launch_async_executions()
             ...
   * - :code:`Analyzer`

       :code:`StateTomography`
     - .. code-block:: python

          analyzer = Analyzer(processor, [pcvl.FockState([1, 1])], "*")
          analyzer.compute()
     - .. code-block:: python

          with computer.acquire()
              # Same class - only the arguments change
              analyzer = Analyzer(
                experiment, computer, [pcvl.FockState([1, 1])], "*")
              analyzer.compute()

.. note::
   :code:`LocalJob` and :code:`RemoteJob` both become :ref:`Execution`; whether the execution is local or remote is
   determined by its computer, allowing both remote and local :ref:`Execution` to be used in :ref:`ExecutionGroup`.

.. warning::
   Already stored :code:`JobGroup` won't be loaded automatically when using the same name in an :ref:`ExecutionGroup`
   since the inner representation and storage location have changed.

Note that the :code:`with computer.acquire()` context is not mandatory,
but will handle automatically the computer's lifetime if one exists (start/stop) and make your code more versatile.
As such, it is advised to use it only once and place it at the highest level possible.

Noise and Experiment
^^^^^^^^^^^^^^^^^^^^

The :ref:`NoiseModel` used to be an attribute of the :ref:`Experiment` class.
While this is still possible, and should still work,
it is strongly advised to set this as a member of the :ref:`Computer` that will execute the computation.

RemoteProcessor vs RemoteComputer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few things have been removed between the :code:`RemoteProcessor` and the :ref:`RemoteComputer`. Here is a list of changes:

- The :code:`name` attribute doesn't exist anymore, so platforms can no longer be changed on-the-fly
- The :meth:`get_rpc_handler()` method no longer exists as :ref:`RemoteComputer` do not have to rely on it.
  Using a :code:`RPCHandler` manually should not be done anymore.
- The :meth:`resume_job()` method no longer exists.
  To retrieve an :ref:`Execution` from the cloud, it must have been serialized before (manually, or automatically by using an :ref:`ExecutionGroup`)

Algorithms
^^^^^^^^^^

The algorithm classes (:ref:`Analyzer`, :ref:`Tomography`) now store a copy of the given :ref:`Experiment`
(including if it's given as a :code:`Processor`), so any change to it after instantiating the algorithm won't affect
the results of the algorithm anymore
(including setting values to the original parameters - that must be done before instantiating the algorithm).


Breaking changes in Perceval 1.2.3
----------------------------------

BSDistribution must be imported from Perceval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BSDistribution class now relies on 2 utility classes and cannot be imported from exqalibur anymore.

Breaking changes in Perceval 1.2
--------------------------------

Processor place in the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`AProcessor` and :code:`Processor` classes have been moved from :code:`perceval.components` to :code:`perceval.runtime`,
so they are at the same place as the :code:`RemoteProcessor`.
While importing from :code:`perceval.components` should still work, it is expected to be completely removed in a few versions, and now produces a warning.
Any code importing these classes directly from the root of perceval should continue to work fine.

Also, the :code:`build_processor()` method from the catalog items is now deprecated.
The method :code:`build_experiment()` should now be used instead.

Although some classes were ported to Exqalibur, their python versions can still be accessed if needed.
No class name has been changed, so :code:`Simulator` or :code:`SLOSBackend` still points to the python version.
The python SLOS is still available in Processors under the name :code:`"SLOS_LEGACY"`

Tokens that were saved before perceval 0.13 will no longer be loaded by perceval due to previous changes to their storage.
Loading them and saving them in a perceval 1.1 should be enough to do the transition.

The new :code:`PlatformSpecs` object that is now returned by :code:`RemoteProcessor.specs` should be accessed through its attributes and no longer as a dictionary. While this remains possible, it is now deprecated.

Breaking changes in Perceval 1.1
--------------------------------

JobGroup number of parallel launch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The number of jobs that a user can run is now directly retrieved from the cloud.
AS such, the `set_cloud_maximal_job_count` and `get_cloud_maximal_job_count` from `RemoteConfig` are now deprecated and no longer work.

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
* :code:`AnnotatedFockState`: Replace the previous :code:`FockState` by allowing rich annotations, having one or more
  string types, each having a complex number for value. This enables to accurately encode physical parameters and
  play with partial distinguishability (e.g. :code:`|{P:H,lambda:0.625},{P:V,lambda:0.618}>`). Please note that apart
  from polarisation, `Perceval` does not provide a generic algorithm to separate rich annotated states, and the user
  would have to write one.

The most breaking change here is that perceval is not able to simulate :code:`AnnotatedFockState`, apart from polarized ones.
Any code manually using annotations to generate distinguishability must be changed to use the new :code:`NoisyFockState` class.
For instance, a :code:`BasicState("|{_:0}, {_:1}>")` from perceval 0.x must be changed to :code:`BasicState("|{0}, {1}>")`
to be able to be simulated.

For more advanced usage of :code:`AnnotatedFockState` and :code:`NoisyFockState`, see the new :ref:`Quantum States` notebook.

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

.. note:: :code:`StateVector` (and therefore :code:`SVDistribution`) accepts any of the three Fock state types as
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

NoiseModel
^^^^^^^^^^

The way of :code:`NoiseModel` to handle its attributes has changed to be more pythonic.
Now, your IDE should be able to tell that the attributes exist in the class,
and the attributes can be changed using a syntax like :code:`noise_model.g2 = 0.1`.

This change is accompanied by the removal of some methods:

- The :code:`__getitem__` has been removed since it was giving a class that is not accessible anymore
- The :code:`set_value` method has been removed, and can be replaced either by spelling directly the attribute (:code:`noise_model.g2 = 0.1`)
  or by using the python method :code:`setattr(noise_model, "g2", 0.1)`.


Older changes
-------------

The documentation to update from an older legacy version to a more recent one can still be found
`here <https://perceval.quandela.net/docs/v0.13/legacy.html>`_.
