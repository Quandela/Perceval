Getting started
===============

What is Perceval?
^^^^^^^^^^^^^^^^^

Perceval is a toolbox containing generic functions and classes, built around an optimised native core (see
:ref:`exqalibur` code reference).

It offers tools to:

* Manipulate quantum states in the Fock space

  * Pure states (:ref:`FockState`, :ref:`StateVector`)
  * Mixed states (:ref:`SVDistribution`, :ref:`DensityMatrix`)

* Build a linear optics :ref:`Experiment` containing

  * A unitary :ref:`Circuit` composed of :ref:`Unitary Components`
  * Some :ref:`Non-unitary Components`
  * Feed-forward through :ref:`Feed-forward Configurators`
  * Variable :ref:`parameters<Parameter>` and :ref:`expressions<Expression>` to parametrise components

* Display circuits and data (:ref:`pdisplay`), serialise them (:ref:`serialization`)
* Define real-world noise parameters applied in the input, the linear-optics circuit and the photon detectors (:ref:`Noise Model`)
* Simulate these experiments through :ref:`different layers of simulations<I. Different layers of simulations>`

  * Perfect simulations with :ref:`Simulation Back-ends`
  * Noisy and non-unitary simulations with the :ref:`Simulator` layer

* Control the flow of quantum computations and choose where they are run:

  * Locally with the :ref:`Processor`, remotely with the :ref:`RemoteProcessor`
  * Manage your :ref:`jobs<Job>` with the :ref:`JobGroup`

Installing Perceval
^^^^^^^^^^^^^^^^^^^

*Perceval* supports several *Python* versions (typically, `those that are not in "end-of-life" <https://devguide.python.org/versions/>`_).
In a virtual environment of any *Python* supported version, a single :code:`pip` command installs Perceval and all of
its dependencies.

.. code-block:: bash

   $ pip install perceval-quandela

.. warning::
   Pay attention that the *Python* package name is "perceval-quandela" and not "perceval"

Once the above command succeeds, you can start typing code in your favorite IDE!

Hello world
^^^^^^^^^^^

The following example is a minimal code to simulate the `Hong–Ou–Mandel effect <https://en.wikipedia.org/wiki/Hong%E2%80%93Ou%E2%80%93Mandel_effect>`_
on the user's computer in a noisy situation, and retrieve both a sample count and exact probabilities computed by a
strong simulation back-end.

>>> import perceval as pcvl
>>> from perceval.algorithm import Sampler
>>>
>>> input_state = pcvl.BasicState("|1,1>")  # Inject one photon on each input mode...
>>> circuit = pcvl.BS()                     # ... of a perfect beam splitter
>>> noise_model = pcvl.NoiseModel(transmittance=0.2, indistinguishability=0.96, g2=0.03)  # Define some noise level
>>>
>>> processor = pcvl.Processor("SLOS", circuit, noise=noise_model)  # Use SLOS, a strong simulation back-end
>>> processor.min_detected_photons_filter(1)  # Accept all output states containing at least 1 photon
>>> processor.with_input(input_state)
>>>
>>> sampler = Sampler(processor)
>>> samples = sampler.sample_count(10_000)['results']  # Ask to generate 10k samples, and get back only the raw results
>>> probs = sampler.probs()['results']  # Ask for the exact probabilities
>>> print(f"Samples: {samples}")
>>> print(f"Probabilities: {probs}")
Samples: {
  |2,0>: 117
  |0,2>: 147
  |1,0>: 4822
  |1,1>: 22
  |0,1>: 4892
}
Probabilities: {
  |2,0>: 0.011858974358974369
  |0,2>: 0.011858974358974369
  |1,1>: 0.0019230769230769245
  |1,0>: 0.48717948717948717
  |0,1>: 0.48717948717948717
}

Now that you can run some code, let's continue with a tutorial to learn Perceval syntax.
