Providers
=========

Scaleway
--------

`Scaleway <https://www.scaleway.com/>`_ is a French cloud provider that provides, among a range of offers, binary power to emulate quantum computing compatible with Perceval.

This Scaleway Quantum as a Service (QaaS) leverages from GPUs like Nvidia P100 and H100 to increase mode limit and accelerate simulations.

You can find prices and additional information on the `Scaleway Labs QaaS page <https://labs.scaleway.com/en/qaas/>`_.

Scaleway authentication
^^^^^^^^^^^^^^^^^^^^^^^

To use Scaleway QaaS as a provider you need a Scaleway account, a Scaleway Project ID and an API key.

1. `Create a Scaleway account <https://www.scaleway.com/en/docs/console/account/how-to/create-an-account/>`_
2. `Create a Scaleway Project <https://www.scaleway.com/en/docs/console/project/how-to/create-a-project/>`_
3. `Create a Scaleway API key <https://www.scaleway.com/en/docs/identity-and-access-management/iam/how-to/create-api-keys/>`_

Using a Scaleway session
^^^^^^^^^^^^^^^^^^^^^^^^

Let's see step by step how to instantiate and use a :ref:`Scaleway Session`.

Import the library and Scaleway from the providers library:

>>> import perceval as pcvl
>>> import perceval.providers.scaleway as scw

Provide your Scaleway Project ID and API key:

>>> PROJECT_ID = "your-scaleway-project-id"
>>> TOKEN = "your-scaleway-api-key"

Choose one of the Perceval compatible platforms `provided by Scaleway <https://labs.scaleway.com/en/qaas/#pricing>`_:

>>> PLATFORM_NAME = "sim:sampling:h100"

You can now create a Scaleway session:

>>> session = scw.Session(platform=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN)
>>> session.start()
>>> /*
...  * Session scope
...  */
>>> session.stop()

You can also create a Scaleway session using a ``with`` block:

>>> with scw.Session(platform=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN) as session:
...     /*
...      * Session scope
...      */

Note: using a ``with`` block you do not need to start and stop your session: it starts automatically at the beginning of the block and stops automatically at its end.

Note: while using a Jupyter Notebook for convenience python objects are kept alive and we recommand using directly ``start`` and ``stop`` methods.

Using an existing Scaleway session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you created your session from the `Scaleway console <https://console.scaleway.com/qaas>`_, you can retrieve it from Perceval.

For this, you only have to go to your session's settings on the console, copy the deduplication identifier and put it to the session creation on your Perceval code.

>>> DEDUPLICATION_ID = "my-quantum-workshop-identifier"
>>> session = scw.Session(platform=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN, deduplication_id=DEDUPLICATION_ID)

A session can be fetched until termination or timeout. If there is no alive session matching the deduplication_id, a new one will be created and returned. 
It is highly convenient if you wish to keep a specific amount of session alive at a time.

Send a circuit to a Scaleway session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you are handling a session, you can instantiate a ``RemoteProcessor`` linked to the session:

>>> processor = session.build_remote_processor()

Then, we can attached a toy circuit to send on our session

>>> processor.set_circuit(pcvl.Circuit(m=2, name="a-toy-circuit") // pcvl.BS.H())
>>> processor.with_input(pcvl.BasicState("|0,1>"))
>>> sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=10_000)
>>> job = sampler.samples(100)
>>> print(job)

Congratulation you can now design and send jobs to Scaleway QaaS through your processor. You can continue with the documentation through :ref:`Work with algorithms`.
