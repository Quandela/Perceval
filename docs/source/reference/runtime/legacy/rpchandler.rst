RPCHandler
^^^^^^^^^^

.. note::
   This class has been moved to :code:`perceval.providers.quandela.rpc_handler`.
   It should not be used on its own anymore, but is only used internally in the :ref:`QuandelaCommunicationLayer`.

A :code:`RPCHandler` (RPC stands for `Remote Procedure Call`) is responsible for all the requests to a Cloud that
Perceval supports. It sends the authentication info along with the request data, and reacts to the HTTP errors which
might occur.

.. autoclass:: perceval.providers.quandela.rpc_handler.RPCHandler
