SimulatorFactory
================

This class exposes a single :code:`build` static method that can be used to create a simulator suited to your circuit.
In particular, it allows you to simulate circuits with non-unitary components or polarized components.

>>> import perceval as pcvl
>>> p = pcvl.Processor("SLOS", 2)
>>> p.add(0, pcvl.BS())
>>> p.add(0, pcvl.TD(1))  # Add a non-unitary component
>>> p.add(0, pcvl.BS())
>>> sim = pcvl.SimulatorFactory.build(p)  # SLOS is transmitted from p to sim
>>> sim.probs(pcvl.BasicState([1, 1]))
{
  |1,0>: 0.25
  |0,0>: 0.24999999999999994
  |0,1>: 0.25
  |0,2>: 0.12500000000000006
  |2,0>: 0.12500000000000006
}

The type of the simulator will depend on your components.
However, it is guaranteed that the resulting simulator will have at least these methods:

- :code:`set_selection` for heralds, postselect, and min_detected_photons_filter (already done if the circuit is a :code:`Processor`).
- :code:`probs` to compute the output probabilities for one input state.
- :code:`probs_svd` to compute the output probabilities for a :code:`SVDistribution`, with possible support for detectors

The simulator factory aims at creating simulators for strong simulation, so it requires the backend to be capable of computing
probability amplitudes.

.. autoclass:: perceval.simulators.simulator_factory.SimulatorFactory
  :members:

Simulator specificities
^^^^^^^^^^^^^^^^^^^^^^^

Depending on the components you have, there may be some specificities to the resulting simulator:

Polarization
------------

If you have polarization components, then you will still be able to use :code:`evolve`.
The input state will also be expected to be a polarized :code:`BasicState`,
or, in the case of :code:`probs_svd`, a :code:`SVDistribution` with a single polarized non-superposed :code:`StateVector`.
If the input is not polarized, the simulator will assume all photons have horizontal polarization.

However, in any method, detectors won't be simulated if given.

Time Delay and Loss
-------------------

If you have :code:`TD` components or :code:`LC` components, the number of photons will not be conserved.
You can still use :code:`evolve`, but you could expect strange results (do it at your own risks).

Feed-forward
------------

If you have :code:`FFConfigurator` or :code:`FFCircuitProvider` in your circuit, you won't be able to call :code:`evolve`.
Also, the dictionary returned by the call to :code:`probs_svd` will only have one performance field, named :code:`global_perf`.
