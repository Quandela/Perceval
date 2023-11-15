Providers
=========

Scaleway
--------

`Scaleway <https://www.scaleway.com/>`_ is a French cloud provider that provides, among a range of offers, binary power to emulate quantum computing compatible with Perceval.

This Scaleway Quantum as a Service (QaaS) leverages from GPUs like Nvidia P100 and H100 to increase mode limit and accelerate simulations.

You can find prices and additional information on the `Scaleway QaaS Labs page <https://labs.scaleway.com/en/qaas/>`_.

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

Inside a session scope you can instantiate a ``Remote Processor`` linked to the session:

>>> processor = session.build_remote_processor()

Congratulation you can now design and send jobs to Scaleway QaaS through your processor. You now can continue with the documentation through :ref:`Work with algorithms`.
