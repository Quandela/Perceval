RPCHandler
^^^^^^^^^^

A :code:`RPCHanlder` (RPC stands for `Remote Procedure Call`) is responsible for all the requests to a Cloud that
Perceval supports. It sends the authentication info along with the request data, and reacts to the HTTP errors which
might occur.

.. autoclass:: perceval.runtime.rpc_handler.RPCHandler
