ExecutionStatus
^^^^^^^^^^^^^^^

.. note::
   This class used to be called :code:`JobStatus`. However, it was renamed for consistency reasons.
   While the old name will still work, it should not be used.

An :ref:`Execution` object contains a lot of metadata on top of the computation results a user wants to get. These can be
retrieved from the :code:`ExecutionStatus` object every execution contains.

>>> s = my_execution.status  # s is an ExecutionStatus instance
>>> if s.completed:
...    print(f"My job lasted {s.duration} seconds.")
My job lasted 37 seconds.

.. autoclass:: perceval.runtime.execution_status.ExecutionStatus
   :members:

.. autoenum:: perceval.runtime.execution_status.RunningStatus
   :members:
