providers
=========

Quandela
--------

Quandela is Perceval default Cloud provider. If no session is created, Quandela Cloud endpoints will be used.

When using Quandela Cloud, you have the same capabilities with and without using a session. It's a matter of code style.

.. autoclass:: perceval.providers.quandela.quandela_session.Session
   :members:

Scaleway
--------

`Scaleway Quantum-as-a-Service <https://www.scaleway.com/en/quantum-as-a-service/>`_ provides access to allocate and program Quantum Processing Units (QPUs), physical or emulated.

Scaleway authentication
^^^^^^^^^^^^^^^^^^^^^^^

To use Scaleway QaaS as a provider you need a Scaleway account, a Scaleway Project ID and an API key.

1. `Create a Scaleway account <https://www.scaleway.com/en/docs/console/account/how-to/create-an-account/>`_
2. `Create a Scaleway Project <https://www.scaleway.com/en/docs/console/project/how-to/create-a-project/>`_
3. `Create a Scaleway API key <https://www.scaleway.com/en/docs/identity-and-access-management/iam/how-to/create-api-keys/>`_

Allocate a QPU session
^^^^^^^^^^^^^^^^^^^^^^^^

Let's see step by step how to instantiate and use a `Scaleway` session.

Import the library and Scaleway from the providers library:

>>> import perceval as pcvl
>>> import perceval.providers.scaleway as scw

Provide your Scaleway Project ID and API key:

>>> PROJECT_ID = "your-scaleway-project-id"
>>> TOKEN = "your-scaleway-api-key"

Choose one of the Perceval compatible platforms `provided by Scaleway <https://www.scaleway.com/en/quantum-as-a-service/>`_:

>>> PLATFORM_NAME = "EMU-SAMPLING-L4" # For emulated QPU
>>> # PLATFORM_NAME = "QPU-BELENOS-12PQ" # For real QPU

You can now create a Scaleway session:

>>> session = scw.Session(platform_name=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN)
>>> session.start()
>>> /*
...  * Session scope
...  */
>>> session.stop()

You can also create a Scaleway session using a ``with`` block:

>>> with scw.Session(platform_name=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN) as session:
...     #
...     # Session scope
...     #

Note: using a ``with`` block you do not need to start and stop your session: it starts automatically at the beginning of the block and stops automatically at its end.

Note: while using a Jupyter Notebook for convenience python objects are kept alive and we recommend using directly ``start`` and ``stop`` methods.

Using an existing Scaleway QPU session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you created your session from the `Scaleway console <https://console.scaleway.com/qaas>`_, you can retrieve it from Perceval.

For this, you only have to go to your session's settings on the console, copy the deduplication identifier and put it to the session creation on your Perceval code.

>>> DEDUPLICATION_ID = "my-quantum-workshop-identifier"
>>> session = scw.Session(platform=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN, deduplication_id=DEDUPLICATION_ID)

A session can be fetched until termination or timeout. If there is no alive session matching the deduplication_id, a new one will be created and returned.
It is highly convenient if you wish to keep a specific amount of session alive at a time.

Send a circuit to a Scaleway QPU session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you are handling a session, you can instantiate a :code:`RemoteProcessor` linked to the session:

>>> processor = session.build_remote_processor()

Then, we can attach a toy circuit and send it on our session

>>> processor.set_circuit(pcvl.Circuit(m=2, name="a-toy-circuit") // pcvl.BS.H())
>>> processor.with_input(pcvl.BasicState("|0,1>"))
>>> sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=10_000)
>>> job = sampler.samples(100)
>>> print(job)

Congratulation you can now design and send jobs to Scaleway QaaS through your processor. You can continue with the documentation of :ref:`algorithm`.
