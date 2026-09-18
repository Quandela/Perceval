Kipu Quantum Hub
^^^^^^^^^^^^^^^^

The `Kipu Quantum Hub <https://dashboard.hub.kipu-quantum.com/>`_ brokers quantum jobs to multiple providers.
Perceval can use it to run Quandela photonic backends through a :class:`KipuCommunicationLayer
<perceval.providers.kipu.kipu_communication_layer.KipuCommunicationLayer>` and a :ref:`RemoteComputer`.

Installation
============

The Kipu integration relies on the optional :code:`qhub-api` dependency. Install Perceval with the :code:`kipu` extra:

.. code-block:: bash

   pip install perceval-quandela[kipu]

If the dependency is missing, creating a Kipu communication layer raises an :class:`ImportError` containing the
installation instruction.

Authentication
==============

Create a Kipu Quantum Hub account and copy a Personal Access Token from the
`Hub dashboard <https://dashboard.hub.kipu-quantum.com/>`_. A token can be passed directly, stored in the
:code:`KIPU_CLOUD_TOKEN` environment variable, or managed with :code:`KipuConfig`.

Alternatively, authenticate with the `qhubctl CLI <https://docs.hub.kipu-quantum.com/quickstart>`_:

.. code-block:: bash

   qhubctl login

When using :code:`qhubctl` credentials, omit the token and let :code:`qhub-api` resolve it. An optional
:code:`organization_id` runs jobs in an organization context; omitting it uses the personal account associated with
the credentials.

KipuCommunicationLayer
======================

The supported backend IDs and aliases are:

==============================  =================
Backend ID                      Alias
==============================  =================
:code:`quandela.sim.belenos`     :code:`sim:belenos`
:code:`quandela.qpu.belenos`     :code:`qpu:belenos`
==============================  =================

Create a communication layer and pass it to :code:`RemoteComputer`:

>>> import perceval as pcvl
>>>
>>> communication_layer = pcvl.KipuCommunicationLayer(
...     platform_name="quandela.sim.belenos",
...     token="your-personal-access-token",
... )
>>> computer = pcvl.RemoteComputer(communication_layer)

.. autoclass:: perceval.providers.kipu.kipu_communication_layer.KipuCommunicationLayer
   :members:

KipuConfig
==========

.. note::
   :ref:`Execution` serialization does not store credentials.
   To be able to deserialize an execution made through a Kipu computer,
   the :class:`KipuConfig` needs to be configured.

:code:`KipuConfig` manages the Personal Access Token, optional Hub URL, proxies, and organization ID. Values set on the
class are cached for the current Python process. Call :meth:`save()` only on a personal machine when they should also
be written to Perceval's persistent configuration:

>>> config = pcvl.KipuConfig()
>>> config.set_token("your-personal-access-token")
>>> config.set_organization_id("your-organization-id")
>>> config.save()

After configuration, the token and organization ID may be omitted:

>>> communication_layer = pcvl.KipuCommunicationLayer("sim:belenos")

To return to the personal account context, set the organization ID to :code:`None` before creating the communication
layer:

>>> config.set_organization_id(None)

.. note::
   Do not persist authentication tokens on shared or public computers. Using :code:`qhubctl`, an environment variable,
   or the in-memory configuration cache avoids writing the token through Perceval.

.. autoclass:: perceval.providers.kipu.kipu_config.KipuConfig
   :members:
   :inherited-members:

Legacy Session
==============

.. warning::
   :code:`Session` belongs to the legacy processor workflow. New code should use :code:`KipuCommunicationLayer` with
   :code:`RemoteComputer`.

.. autoclass:: perceval.providers.kipu.kipu_session.Session
   :members:
