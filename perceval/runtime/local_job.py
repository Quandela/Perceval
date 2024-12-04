# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import threading

from perceval.utils.logging import get_logger, channel
from .job import Job
from .job_status import JobStatus, RunningStatus


class LocalJob(Job):
    def __init__(self,
                 fn: callable,
                 result_mapping_function: callable = None,
                 delta_parameters: dict = None,
                 command_param_names: list = None):
        """
        Runs a task locally (i.e. on the computer running Perceval)

        :param fn: Function to be called by the job
        :param result_mapping_function: Optional results post-processing function (e.g. can be used to convert results)
        :param delta_parameters: mapping of {'param_name': param_value}
        :param command_param_names: names of `fn` parameters which can be mapped from the *args of __call__
        """
        super().__init__(result_mapping_function=result_mapping_function,
                         delta_parameters=delta_parameters,
                         command_param_names=command_param_names)
        self._fn = fn
        self._status = JobStatus()
        self._worker = None
        self._user_cb = None
        self._cancel_requested = False

    def set_progress_callback(self, callback: callable):  # Signature must be (float, str | None)
        self._user_cb = callback

    @property
    def status(self) -> JobStatus:
        # for local job
        if self._status.running and not self._worker.is_alive():
            self._status.stop_run()
        return self._status

    def _progress_cb(self, progress: float, phase: str | None = None):
        self._status.update_progress(progress, phase)
        if self._cancel_requested:
            return {'cancel_requested': True}
        if self._user_cb is not None:
            return self._user_cb(progress, phase)
        return None

    def execute_sync(self, *args, **kwargs):
        assert self._status.waiting, "job has already been executed"

        if 'progress_callback' in kwargs:
            self._user_cb = kwargs['progress_callback']
        self._delta_parameters['command']['progress_callback'] = self._progress_cb
        self._handle_params(args, kwargs)
        self._call_fn_safe(**self._delta_parameters['command'])
        return self.get_results()

    def _call_fn_safe(self, *args, **kwargs):
        r"""
        Wrapping function called in the job thread in charge of catching exception and correctly setting status
        """
        try:
            # it has already been called, calling it again to get more precise running time
            self._status.start_run()
            self._results = self._fn(*args, **kwargs)
            if self._cancel_requested:
                self._status.stop_run(RunningStatus.CANCELED, "User has canceled the job")
            else:
                self._status.stop_run()
        except Exception as e:
            get_logger().warn(f"An exception was raised during job execution.\n{type(e).__name__}: {e}", channel.user)
            self._status.stop_run(RunningStatus.ERROR, type(e).__name__+": "+str(e))

    def execute_async(self, *args, **kwargs) -> Job:
        assert self._status.waiting, "job has already been executed"
        if 'progress_callback' in kwargs:
            self._user_cb = kwargs['progress_callback']
        self._delta_parameters['command']['progress_callback'] = self._progress_cb
        self._handle_params(args, kwargs)
        self._status.start_run()
        # we are launching the function in a separate thread
        self._worker = threading.Thread(target=self._call_fn_safe, kwargs=self._delta_parameters['command'])
        self._worker.start()
        return self

    def cancel(self):
        self._cancel_requested = True

    def _get_results(self):
        if self._result_mapping_function:
            get_logger().info(
                f"Converting local job results with {self._result_mapping_function.__name__}", channel.general)
            if 'results' in self._results:
                self._results['results'] = self._result_mapping_function(self._results['results'],
                                                                         **self._delta_parameters['mapping'])
            elif 'results_list' in self._results:
                for res in self._results["results_list"]:
                    mapping_args = {key: res["iteration"].get(key, val)
                                    for key, val in self._delta_parameters['mapping'].items()}

                    res["results"] = self._result_mapping_function(res['results'], **mapping_args)
            else:
                raise KeyError("Cannot find either 'result' or 'results_list' in self._results")
            self._result_mapping_function = None
        return self._results
