DistinguishablePhotonMitigation
===============================

:code:`DistinguishablePhotonMitigation` reduces errors associated with partial photon distinguishability and unwanted
multi-photon emission. It runs related experiments with fewer input photons and combines their results using the noise
characterization of the computer.

The :code:`order` controls how far the correction is expanded:

>>> import perceval as pcvl
>>>
>>> computer.mitigations = [pcvl.DistinguishablePhotonMitigation(order=1)]

A larger order can correct higher-order contributions but requires more sub-computations
(scaling as :math:`\sum_{k=0}^{order} C_k^n`). It should be increased only
when the execution budget and the input photon count justify the extra work.

For workflows containing several input photon counts, pass a dictionary to select a different order for each count:

>>> mitigation = pcvl.DistinguishablePhotonMitigation({2: 1, 4: 2, 6: 3})

Practical considerations
------------------------

* The experiment input must be a :class:`FockState <perceval.utils.statevector.FockState>`.
* The technique uses the computer's noise model, especially :attr:`indistinguishability` and :attr:`g2`.
* It has no effect when the photons are perfectly indistinguishable and :attr:`g2` is zero.
* Requested samples and shots are divided among the generated sub-computations. Very small budgets may therefore be
  incompatible with a high order.
* Output states containing more photons than the original input are removed as part of the correction,
  meaning all :attr:`g2` related output states are lost in the process.

The :meth:`overhead()` method reports how many sub-computations a given input state would require and can help choose an
appropriate order before launching an execution:

>>> mitigation = pcvl.DistinguishablePhotonMitigation(order=2)
>>> mitigation.overhead(pcvl.FockState("|1,1,1>"))
7

.. autoclass:: perceval.runtime.error_mitigation.distinguishable_photon_mitigation.DistinguishablePhotonMitigation
   :members:
