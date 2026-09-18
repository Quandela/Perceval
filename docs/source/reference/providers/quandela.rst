Quandela
^^^^^^^^

`Quandela Cloud <https://cloud.quandela.com/>`_ provides access to physical and emulated quantum processing units.
It is Perceval's historical provider and offers all the capabilities that Perceval has.

QuandelaCommunicationLayer
==========================

.. autoclass:: perceval.providers.quandela.quandela_communication_layer.QuandelaCommunicationLayer

RemoteConfig
============

.. note::
   :ref:`Execution` serialization does not store credentials.
   To be able to deserialize an execution made through a Quandela computer,
   the :class:`RemoteConfig` needs to be configured.

>>> config = pcvl.RemoteConfig()
>>> config.set_token("your-personal-access-token")
>>> config.set_proxies({"https": "your-proxy"})  # Shared among all provider configs
>>> config.save()

After configuration, the token and provider name may be omitted from the communication layer:

>>> communication_layer = pcvl.QuandelaCommunicationLayer(
...     name=PLATFORM_NAME,
... )

.. note::
   Do not persist authentication tokens on shared or public computers. Using an environment variable or the in-memory
   configuration cache avoids writing the token to disk.

.. autoclass:: perceval.providers.quandela.remote_config.RemoteConfig
   :members:
   :inherited-members:

Legacy Session
==============

.. warning::
   This class is about the legacy workflow for providers.
   It should not be used, and is kept here only for backward compatibility

.. autoclass:: perceval.providers.quandela.quandela_session.Session
   :members:
