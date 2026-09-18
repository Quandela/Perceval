algorithm
^^^^^^^^^

The :code:`perceval.algorithm` package contains the code of several simple and generic **quantum algorithms**.
It provides a :ref:`Computer` and :ref:`Experiment`-centric syntax to run an algorithm locally or remotely,
on a simulator or an actual QPU.

All algorithms take either a local or a remote computer as parameter, in order to perform a task based on a given :ref:`Experiment`.

>>> from perceval import SimulatedComputer, BasicState, Experiment
>>> from perceval.algorithm import Analyzer
>>> experiment = pcvl.Experiment(pcvl.BS())  # No need to declare the input state here
>>> computer = pcvl.SimulatedComputer("CliffordClifford2017")
>>> with computer.acquire():
...     ca = pcvl.algorithm.Analyzer(computer, experiment, [BasicState([1, 1])], "*")
...     ca.compute()

.. note::
   Algorithms may create one or several :ref:`Execution` internally, that are run sequentially.

**Samples of interest vs Shots**

On a QPU, the acquisition is measured in **shots**. A shot is any coincidence with at least 1 detected photon.
Shots act as credits on the Cloud services. Users have to set a maximum shots value they are willing to use for any
algorithm. Note that this is a per-:ref:`Execution` limit (so globally, an algorithm may use more than this number of shots).

>>> import perceval as pcvl
>>> remote_c = pcvl.RemoteComputer(pcvl.QuandelaCommunicationLayer("sim:sampling"))
>>> experiment = Experiment(pcvl.BS())
>>> with remote_c.acquire():
...     analyzer = pcvl.algorithm.Analyzer(remote_c, experiment, [BasicState([1, 1])], "*", max_shots_per_call=500_000)
...     analyzer.compute()

Here, the computation runs on the `sim:sampling` platform.

For more information about the shots and shots/samples ratio estimate, please read
:ref:`Remote computing tutorial<Remote Computing>`.

.. toctree::
   analyzer
   tomography
   sampler
