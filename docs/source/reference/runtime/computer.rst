Computer
^^^^^^^^

A :code:`Computer` is a class dedicated at executing :ref:`Computation` or :ref:`ComputationIterator`.
It represents a physical or simulated way of performing some data acquisition.

:code:`Computers` can be local, in which case they do everything on the user machine, or remote, in which case they
send HTTP requests to some distant platform that will perform the acquisition.
A :code:`Computer` can be a simulator or real QPU. In any case, its intent is to represent a QPU.

As such, it has some noise, or can be given some if it's a simulator.

>>> remote_computer = RemoteComputer(QuandelaCommunicationLayer("qpu:belenos"))
>>> print(computer.noise)  # Will print the current qpu noise
>>> local_computer = SimulatedComputer("SLOS")
>>> local_computer.noise = NoiseModel(brightness = 0.8)  # Set the noise for this simulation computer


Any :code:`Computer` can apply some :ref:`Error Mitigation` techniques automatically when given a :ref:`Computation`.

>>> local_computer.mitigations = MitigationFactory(MitigationLevel.medium).build()


Also, some :code:`Computers` may have particular possible parameters that will act on the computation process.

>>> print(remote_computer.available_parameters)
>>> remote_computer.reset_parameters()  # Reset to default parameters
>>> remote_computer.parameters |= {"compute_physical_logical_perf": True}


:code:`Computers` are made to be interchangeable as much as possible. As such, their public interface is essentially the same.
In particular, they can all do computations synchronously and asynchronously.
The details on how they do it depends on their particular implementation.

Since they share the same interface, it is a good idea to always use the common public methods.
For instance, you should always start and stop them before and after doing all your computations, even for computers where this does nothing.
This automatic start and stop process can be done using the :meth:`acquire()` context manager.

>>> with acquire(remote_computer):
...     # All computations here

.. note::
   A computer should not be stopped until all computations are executed. Hence, the acquisition should be done at the
   topmost level. For instance, methods that take a computer as argument should not acquire it.

.. automethod:: perceval.runtime.abstract_computer.acquire


SimulatedComputer
^^^^^^^^^^^^^^^^^

Unless you have a real QPU on your machine, the only local computer you're going to use is the :code:`SimulatedComputer`.
This computer can be used to simulate experiments, by giving it a backend or a backend name.
It will automatically choose which method to use to do the simulation based on the backend and the request (sampling, feed-forward...).

Its main way to compute is synchronous, but it can create threads to allow for asynchronous calls.

.. autoclass:: perceval.runtime.simulated_computer.SimulatedComputer
   :members:
   :inherited-members:


RemoteComputer
^^^^^^^^^^^^^^

The :code:`RemoteComputer` lets you have access to a distant QPU or simulator.
It requires a :code:`CommunicationLayer`, that can be one given by any of the :ref:`providers` (Quandela, Scaleway, Kipu...).

Its main way of compute is asynchronous, but it can wait internally for job completion to allow for synchronous calls.

A :code:`RemoteComputer` can have default mitigations if its :code:`mitigation` member is :code:`None`.
The mitigations can be completely disabled by explicitly setting them to an empty list.

>>> remote_computer.mitigations = []  # Disable default mitigations

Also, the mitigations are usually sent through the :code:`CommunicationLayer` and applied directly on the distant platform.
This can be changed so that the mitigations happen on your machine.
Note however that default mitigations are still applied remotely if the :code:`mitigations` member is None.

>>> remote_computer.use_mitigations_remotely = False

.. warning::
   When applying the mitigations locally, the mitigated imperfections are taken from when the execution was sent,
   not when it was executed on the platform, so this parameter should not be modified unless necessary.

.. autoclass:: perceval.runtime.remote_computer.RemoteComputer
   :members:
   :inherited-members:
