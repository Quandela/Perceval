Stepper Simulator
=================

The :code:`Stepper` is a special simulator that simulates the evolution of a :code:`StateVector` component by component.
It's main purpose is to be able to see what the circuit does step by step.
The internal behaviour of the :code:`Stepper` can be seen by using the debugger and going into the code,
or by using the :code:`apply` method manually.

.. warning::
   The Stepper is much more slower than the usual :code:`Simulator`,
   as well as being less flexible,
   so it shouldn't be used for anything other than visualization.

>>> import perceval as pcvl
>>> c = pcvl.BS() // pcvl.PS(1.) // pcvl.BS()
>>> sim = pcvl.Stepper(pcvl.SLOSBackend())
>>> sim.set_circuit(c)
>>> state = pcvl.StateVector([1, 1])
>>> for r, component in c:
>>>     state = sim.apply(state, r, component)
>>>     print(component.describe(), state)
BS.Rx() 0.707I*|2,0>+0.707I*|0,2>
PS(phi=1) (-0.643-0.294I)*|2,0>+0.707I*|0,2>
BS.Rx() (-0.321-0.501I)*|2,0>+(-0.292-0.455I)*|1,1>+(0.321+0.501I)*|0,2>

The :code:`Stepper` can also be used to simulate :code:`LC` components,
but the :code:`apply` method can't be used in that case (see the code of the :code:`compile` method to see how to do it manually)

Note however that the :code:`Stepper` doesn't support annotated or polarized :code:`BasicState`,
nor does it apply logical selection like heralding and post-selection on its own.
It can still perform physical selection, including after simulating detectors.

Also, when calling :code:`probs_svd` on the :code:`Stepper`, the returned dictionary has only the 'results' field.

.. autoclass:: perceval.simulators.stepper.Stepper
  :members:
  :inherited-members:
