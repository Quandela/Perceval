Scaleway
^^^^^^^^

`Scaleway Quantum as a Service <https://www.scaleway.com/en/quantum-as-a-service/>`_ provides access to physical and
emulated quantum processing units. Perceval connects to a Scaleway platform through a
:class:`ScalewayCommunicationLayer
<perceval.providers.scaleway.scaleway_communication_layer.ScalewayCommunicationLayer>` inserted into a
:ref:`RemoteComputer`.

Authentication
==============

Using Scaleway QaaS requires a Scaleway account, project ID, and API secret key:

1. `Create a Scaleway account <https://www.scaleway.com/en/docs/console/account/how-to/create-an-account/>`_.
2. `Create a Scaleway project <https://www.scaleway.com/en/docs/console/project/how-to/create-a-project/>`_.
3. `Create a Scaleway API key
   <https://www.scaleway.com/en/docs/identity-and-access-management/iam/how-to/create-api-keys/>`_.

The secret key can be passed directly to the communication layer, stored in the :code:`SCALEWAY_CLOUD_TOKEN`
environment variable, or managed with :code:`ScalewayConfig`. The project ID is always passed explicitly.

ScalewayCommunicationLayer
==========================

Choose a platform listed by Scaleway, then create the communication layer and remote computer:

>>> import perceval as pcvl
>>> import perceval.providers.scaleway as scw
>>>
>>> PROJECT_ID = "your-scaleway-project-id"
>>> TOKEN = "your-scaleway-api-secret-key"
>>> PLATFORM_NAME = "EMU-SAMPLING-L4" # For emulated QPU
>>> # PLATFORM_NAME = "QPU-BELENOS-12PQ" # For real QPU
>>>
>>> communication_layer = scw.ScalewayCommunicationLayer(
...     platform_name=PLATFORM_NAME,
...     project_id=PROJECT_ID,
...     token=TOKEN,
... )
>>> computer = pcvl.RemoteComputer(communication_layer)

Scaleway requires a QaaS session. The computer lifecycle creates and terminates it, so use
:meth:`Computer.acquire <perceval.runtime.abstract_computer.AComputer.acquire>` around all executions sharing
the session. Try to start and stop the session only once to avoid overhead:

>>> experiment = pcvl.Experiment(pcvl.BS())
>>> experiment.with_input(pcvl.BasicState("|0,1>"))
>>> experiment.min_detected_photons_filter(1)
>>> factory = pcvl.ExecutionFactory(computer, experiment, max_shots_per_call=10_000)
>>>
>>> with computer.acquire():
...     results = factory.samples(max_samples=100)
...     # All computation goes here

The session can instead be managed explicitly with :meth:`computer.start() <perceval.runtime.remote_computer.RemoteComputer.start>`
and :meth:`computer.stop() <perceval.runtime.remote_computer.RemoteComputer.stop>`.
Calling :meth:`computer.delete() <perceval.runtime.remote_computer.RemoteComputer.delete>` deletes the attached session
and its jobs.

Using an existing Scaleway QPU session
======================================

If you created your session from the `Scaleway console <https://console.scaleway.com/qaas>`_, you can retrieve it from Perceval.

For this, you only have to go to your session's settings on the console, copy the deduplication identifier and put it to the session creation on your Perceval code.

>>> DEDUPLICATION_ID = "my-quantum-workshop-identifier"
>>> communicationLayer = scw.ScalewayCommunicationLayer(
...     platform=PLATFORM_NAME,
...     project_id=PROJECT_ID,
...     token=TOKEN,
...     deduplication_id=DEDUPLICATION_ID)

A session can be fetched until termination or timeout. If there is no alive session matching the deduplication_id, a new one will be created and returned.
It is highly convenient if you wish to keep a specific amount of session alive at a time.

.. autoclass:: perceval.providers.scaleway.scaleway_communication_layer.ScalewayCommunicationLayer
   :members:

ScalewayConfig
==============

.. note::
   :ref:`Execution` serialization does not store credentials.
   To be able to deserialize an execution made through a Scaleway computer,
   the :class:`ScalewayConfig` needs to be configured.

:code:`ScalewayConfig` manages the API secret key, API URL, proxies, and default platform provider. Values set on the
class are cached for the current Python process. Call :meth:`save()` only on a personal machine when they should also
be written to Perceval's persistent configuration:

.. note::
   :ref:`Execution` serialization does not store credentials.
   To be able to deserialize an execution made through a Quandela computer,
   the :class:`RemoteConfig` needs to be configured.

>>> config = scw.ScalewayConfig()
>>> config.set_token(TOKEN)
>>> config.set_provider("quandela")
>>> config.save()

After configuration, the token and provider name may be omitted from the communication layer:

>>> communication_layer = scw.ScalewayCommunicationLayer(
...     platform_name=PLATFORM_NAME,
...     project_id=PROJECT_ID,
... )

.. note::
   Do not persist authentication tokens on shared or public computers. Using an environment variable or the in-memory
   configuration cache avoids writing the token to disk.

.. autoclass:: perceval.providers.scaleway.scaleway_config.ScalewayConfig
   :members:
   :inherited-members:

Legacy Session
==============

.. warning::
   :code:`Session` belongs to the legacy processor workflow. New code should use
   :code:`ScalewayCommunicationLayer` with :code:`RemoteComputer`.

.. autoclass:: perceval.providers.scaleway.scaleway_session.Session
   :members:
