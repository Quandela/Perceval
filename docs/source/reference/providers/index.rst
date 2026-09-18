providers
^^^^^^^^^

Cloud providers are ways to send :ref:`Computations <Computation>` to be executed remotely by a distant computer.

They are all based on common principles: they provide a :ref:`CommunicationLayer`
that can be inserted into a :ref:`RemoteComputer`.

>>> communication_layer = ProviderCommunicationLayer(...)  # Provider-specific arguments
>>> computer = RemoteComputer(communication_layer)

Note that the :ref:`CommunicationLayer` is not intended to be used directly,
but only accessed through the :ref:`RemoteComputer` interface.

Providers can also provide custom :ref:`RemoteConfig` to be able to store their specific user credentials.

.. toctree::

   quandela
   scaleway
   kipu
