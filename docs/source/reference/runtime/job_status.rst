JobStatus
^^^^^^^^^

A :ref:`Job` object contains a lot of metadata on top of the computation results a user wants to get. These can be
retrieved from the :code:`JobStatus` object every job contains.

>>> s = my_job.status  # s is a JobStatus instance
>>> if s.completed:
...    print(f"My job lasted {s.duration} seconds.")
My job lasted 37 seconds.

.. autoclass:: perceval.runtime.job_status.JobStatus
   :members:

.. autoenum:: perceval.runtime.job_status.RunningStatus
   :members:
