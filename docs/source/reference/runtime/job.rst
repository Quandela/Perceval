Job
^^^

Job is class responsible for the computation and retrieval of a task's results. It hides the `local` vs `remote`, and
`synchronous` vs `asynchronous` executions, which are orthogonal concepts, even if it's more natural for a local job
to be run synchronously, and a remote job asynchronously.

The local vs remote question is handled by two different classes, :code:`LocalJob` and :code:`RemoteJob`, sharing the
same interface.

* Execute a job synchronously

>>> args = [10_000]  # for instance, the expected sample count
>>> results = job.execute_sync(*args)  # Executes the job synchronously (blocks the execution until results are ready)
>>> results = job(*args)  # Same as above

* Execute a job asynchronously

>>> job.execute_async(*args)

This call is non-blocking, however results are not available when this line has finished executing. The job object
provides information on the progress.

>>> while not job.is_complete:  # Check if the job has finished running
...     print(job.status.progress)  # Progress is a float value between 0. and 1. representing a progress from 0 to 100%
...     time.sleep(1)
>>> if job.is_failed:  # Check if the job has failed
...     print(job.status.stop_message)  # If so, print the reason
>>> results = job.get_results()  # Retrieve the results if any

Typically, the results returned by an algorithm is a Python dictionary containing a ``'results'`` key, plus additional
data (performance scores, etc.).

* A job cancellation can be requested programmatically by the user

>>> job.cancel()  # Ask for job cancelation. The actual end of the execution may take some time

When a job is canceled, it may contain partial results. To retrieve them, call :meth:`get_results()`.

* A remote job can be resumed as following:

>>> remote_processor = pcvl.RemoteProcessor("any:platform")
>>> job = remote_processor.resume_job("job_id")  # You can find job IDs on Quandela Cloud's website
>>> print(job.id)  # The ID field is also available in every remote job object


LocalJob
========

.. autoclass:: perceval.runtime.local_job.LocalJob

RemoteJob
=========

.. autoclass:: perceval.runtime.remote_job.RemoteJob
