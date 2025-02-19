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

from perceval.runtime import Job, RemoteJob, RunningStatus
from perceval.runtime.rpc_handler import RPCHandler
from perceval.utils import PersistentData, FileFormat
from perceval.utils.logging import get_logger, channel
import os
import json
from datetime import datetime
import time
from tqdm import tqdm


FILE_EXT_JGRP = 'jgrp'
JGRP_DIR_NAME = "job_group"
DATE_TIME_FORMAT = "%Y%m%d_%H%M%S"


class JobGroup:
    """
    JobGroup handles a collection of remote jobs.
    A job group is named and persistent (job metadata will be written on disk).
    Job results will never be stored and will be retrieved every time from the Cloud.

    The JobGroup class can perform various tasks such as:
        - Saving information for a collection of jobs, whether they have been sent to the cloud or not.
        - Running jobs within the group either in parallel or sequentially.
        - Rerunning failed jobs within the group.
        - Retrieving all results at once.

    :param name: A name uniquely identifying the group (also, the filename used to save data on disk).
                 If the same name is used more than once, jobs can be appended to the same group.
    """
    _PERSISTENT_DATA = PersistentData()  # Persistent data object for the job group class
    _DIR_PATH = os.path.join(_PERSISTENT_DATA.directory, JGRP_DIR_NAME)

    def __init__(self, name: str):
        self._name = name
        now: datetime = datetime.now()
        self.created_date = now
        self.modified_date = now
        self._jobs: list[RemoteJob] = []
        self._file_path = os.path.join(JobGroup._DIR_PATH, f"{self._name}.{FILE_EXT_JGRP}")

        if self._exists_on_disk(name):
            get_logger().info(f'Job Group with name {name} exists; subsequent jobs will be appended to it',
                              channel.user)
            self._load_job_group()
        else:
            self._write_to_file()

    @property
    def name(self) -> str:
        """
        Name of the job group
        """
        return self._name

    @property
    def list_remote_jobs(self) -> list[RemoteJob]:
        """
        Returns a chronologically ordered list of RemoteJobs in the group.
        Jobs never sent to the cloud will be represented by None.
        """
        return [job if job.has_been_send else None for job in self._jobs]

    def to_dict(self) -> dict:
        group_data = {}
        group_data['created_date'] = self.created_date.strftime(DATE_TIME_FORMAT)
        group_data['modified_date'] = self.modified_date.strftime(DATE_TIME_FORMAT)
        group_data['job_group_data'] = []
        for job in self._jobs:
            group_data['job_group_data'].append(job.to_dict())
        return group_data

    def from_dict(self, group_data):
        self.created_date = datetime.strptime(group_data['created_date'], DATE_TIME_FORMAT)
        self.modified_date = datetime.strptime(group_data['modified_date'], DATE_TIME_FORMAT)
        for job_entry in group_data['job_group_data']:
            self._jobs.append(self._build_remote_job(job_entry))

    def _write_to_file(self, modified: datetime = None):
        """
        Writes job group data to disk
        """
        if modified:
            self.modified_date = modified
        JobGroup._PERSISTENT_DATA.write_file(self._file_path, json.dumps(self.to_dict()), FileFormat.TEXT)

    def _load_job_group(self):
        """
        Creates a Job Group by loading an existing one from file
        """
        group_data = json.loads(JobGroup._PERSISTENT_DATA.read_file(self._file_path, FileFormat.TEXT))

        self.from_dict(group_data)

    @staticmethod
    def _build_remote_job(job_entry: dict) -> RemoteJob:
        """
        Returns a RemoteJob object recreated using its id and platform metadata
        """
        platform_metadata = job_entry['metadata']
        user_token = platform_metadata['headers']['Authorization'].split(' ')[1]

        rpc_handler = RPCHandler(platform_metadata['platform'],
                                 platform_metadata['url'], user_token)

        return RemoteJob.from_dict(job_entry, rpc_handler)

    @staticmethod
    def list_existing() -> list[str]:
        """
        Returns a list of filenames of all JobGroups saved to disk
        """
        jgrp_path = JobGroup._DIR_PATH
        files = [os.path.splitext(f)[0] for f in os.listdir(jgrp_path) if f.endswith(FILE_EXT_JGRP)]
        return files

    @staticmethod
    def _exists_on_disk(name: str) -> bool:
        """
        Returns True if a JobGroup with an identical name is already saved on disk
        """
        return JobGroup._PERSISTENT_DATA.has_file(os.path.join(JobGroup._DIR_PATH, name + '.' + FILE_EXT_JGRP))

    def add(self, job_to_add: Job, **kwargs):
        """
        Adds information of the new RemoteJob to an existing Group.
        Saves the data in a chronological order in the group (each entry is
        a dictionary of necessary information - status, id, body, metadata)

        :param job_to_add: a remote job to add to the list of existing job group
        """
        if not isinstance(job_to_add, RemoteJob):
            raise TypeError(f'Only a RemoteJob can be added to a JobGroup (got {type(job_to_add)})')

        # Reject adding a duplicate RemoteJob
        if job_to_add.id and job_to_add.id in [job.id for job in self._jobs]:
            raise ValueError(f"Duplicate job detected : job id {job_to_add.id} exists in the group.")
        if kwargs:
            job_to_add.set_args(kwargs)
        self._jobs.append(job_to_add)
        self._write_to_file()

    def _update_job_statuses(self) -> list:
        """
        Lists through all jobs in the group and returns a list of status
        If a status is changed from existing in file, that entry is
        updated with new information
        """
        for job in self._jobs:
            if job.has_been_send:
                old_status = job._job_status
                status = job.status()
                if old_status != status:
                    self._write_to_file()

    def progress(self) -> dict:
        """
        Iterates over all jobs in the group to create a dictionary of the current status of jobs.
        Jobs in the group are categorized as follows (depending on their RunningStatus on the Cloud)

        - Finished
            - successful {'SUCCESS'}
            - unsuccessful {'CANCELED', 'ERROR', 'UNKNOWN', 'SUSPENDED'}
        - Unfinished
            - sent {'WAITING', 'RUNNING', 'CANCEL_REQUESTED'}
            - not sent {None}

        :return: dictionary of the current status of jobs
        """
        self._update_job_statuses()

        unsent_job_cnt = 0
        success_job_cnt = 0
        other_job_cnt = 0
        sent_job_cnt = 0

        for job in self._jobs:
            if not job.has_been_send:
                unsent_job_cnt += 1
                continue
            status = job._job_status
            if status.success:
                success_job_cnt += 1
                continue
            if status.waiting or status.running:
                sent_job_cnt += 1
                continue
            other_job_cnt += 1

        fin_job_prog = {'successful': success_job_cnt, 'unsuccessful': other_job_cnt}
        unfin_job_prog = {'sent': sent_job_cnt, 'not sent': unsent_job_cnt}

        progress = dict()
        progress['Total'] = len(self._jobs)
        progress['Finished'] = [other_job_cnt + success_job_cnt, fin_job_prog]
        progress['Unfinished'] = [sent_job_cnt + unsent_job_cnt, unfin_job_prog]

        return progress

    def track_progress(self):
        """
        Displays the status and progress of each job in the group using `tqdm` progress bars.
        Jobs are categorized into "Successful," "Running/Active on Cloud," and
        "Inactive/Unsuccessful." The method iterates over the list of jobs, continuously
        refreshing their statuses and updating the progress bars to provide real-time feedback
        until no "Running/Waiting" jobs remain on the Cloud.
        """
        tot_jobs = len(self._jobs)

        # define tqdm bars
        bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
        success_bar = tqdm(total=tot_jobs, bar_format=bar_format, desc="Successful Jobs", position=0, leave=True)
        active_bar = tqdm(total=len(self.list_active_jobs()), bar_format=bar_format, desc="Running/Waiting Jobs",
                          position=1, leave=True)  # non-active job can't become active, as long as this function is blocking
        inactive_bar = tqdm(total=tot_jobs, bar_format=bar_format, desc="Inactive/Unsuccessful Jobs", position=2,
                            leave=True)

        while True:
            self._update_job_statuses()

            count_success = 0
            count_running = 0
            count_inactive = 0

            for job in self._jobs:
                status = job._job_status
                if status.success:
                    count_success += 1
                    continue
                if status.waiting or status.running:
                    count_running += 1
                    continue
                count_inactive += 1


            success_bar.n = count_success
            active_bar.n = count_running
            inactive_bar.n = count_inactive

            for bar in [success_bar, active_bar, inactive_bar]:
                bar.refresh()  # needed to change the displayed value to bar.n

            if count_running == 0:
                break

            time.sleep(5)  # delay before next acquisition of statuses

        success_bar.close()
        active_bar.close()
        inactive_bar.close()

    @staticmethod
    def delete_all_job_groups():
        """
        Delete all existing groups on disk
        """
        jgrp_dir_path = JobGroup._DIR_PATH
        list_groups = JobGroup.list_existing()
        for each_file in list_groups:
            JobGroup._PERSISTENT_DATA.delete_file(os.path.join(jgrp_dir_path, each_file + '.' + FILE_EXT_JGRP))

    @staticmethod
    def delete_job_group(name: str):
        """
        Delete a single group by name

        :param name: name of the JobGroup to delete
        """
        file_path = os.path.join(JobGroup._DIR_PATH, name + '.' + FILE_EXT_JGRP)
        JobGroup._PERSISTENT_DATA.delete_file(file_path)

    @staticmethod
    def delete_job_groups_date(del_before_date: datetime):
        """
        Delete all saved groups created before a date.

        :param del_before_date: datetime of the oldest job group to keep. Anterior groups will be deleted.
        """
        existing_groups = JobGroup.list_existing()
        files_to_del = []  # list of files before date to delete
        for jg_name in existing_groups:
            jg = JobGroup(jg_name)
            jg_datetime = datetime.strptime(jg.created_date, DATE_TIME_FORMAT)
            if jg_datetime < del_before_date:
                files_to_del.append(jg_name)

        if not files_to_del:
            get_logger().warn(f'No files found to delete before {del_before_date}', channel.user)

        # delete files
        for f in files_to_del:
            JobGroup.delete_job_group(f)

    def _list_jobs_status_type(self, statuses: list[str]) -> list[RemoteJob]:
        remote_jobs = []
        self._update_job_statuses()
        for job in self._jobs:
            if job._job_status in statuses:
                remote_jobs.append(self._recreate_remote_job(job.to_dict()))
        return remote_jobs

    def list_successful_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that have run successfully on the cloud.
        """
        return self._list_jobs_status_type(['SUCCESS'])

    def list_active_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that are currently active on the cloud - those with a Running or
        Waiting status.
        """
        return self._list_jobs_status_type(['RUNNING', 'WAITING'])

    def list_unsuccessful_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that have run unsuccessfully on the cloud - errored or canceled
        """
        return self._list_jobs_status_type(['ERROR', 'CANCELED'])

    def list_unsend_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that have not been send to the cloud
        """
        return [job for job in self._jobs if not job.has_been_send]

    def _launch_jobs(self, rerun: bool, delay: int = None):
        """
        Launches or reruns jobs in the group on Cloud in a parallel/sequential manner.

        :param rerun: if True rerun failed jobs or run unsent jobs
        :param delay: number of seconds to wait between the launch of to jobs on cloud
        """

        jobs = None
        if rerun:
            jobs = self.list_unsuccessful_jobs()
        else:
            jobs = self.list_unsend_jobs()
        job_nmb = len(jobs)

        if delay is not None:
            # Use tqdm to track progress if sequential
            bar_format = '{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}|{desc}'

            prog = tqdm(total=job_nmb, bar_format=bar_format, desc="Successful: 0, Failed: 0")

        count_success = 0
        count_fail = 0
        for job in self._jobs:
            if rerun:
                self._jobs.append(job.rerun())
            else:
                job.execute_async()

            self._write_to_file()  # save that we sent the job

            if delay is not None:
                while not job.is_complete:
                    time.sleep(1)

                if job.status.success:
                    count_success += 1
                else:
                    count_fail += 1

                prog.update(1)
                prog.set_description_str(f"Successful: {count_success}, Failed: {count_fail}")

                time.sleep(delay)  # add delay before launching next job

        if delay is not None:
            prog.close()

        self._write_to_file()

    def run_sequential(self, delay: int):
        """
        Launches the unsent jobs in the group on Cloud in a sequential manner with a
        user-specified delay between the completion of one job and the start of the next.

        :param delay: number of seconds to wait between launching jobs on cloud
        """
        self._launch_jobs(rerun=False, delay=delay)

    def rerun_failed_sequential(self, delay: int):
        """
        Reruns Failed jobs in the group on the Cloud in a sequential manner with a
        user-specified delay between the completion of one job and the start of the next.

        :param delay: number of seconds to wait between re-launching jobs on cloud
        """
        self._launch_jobs(rerun=True, delay=delay)

    def run_parallel(self):
        """
        Launches all the unsent jobs in the group on Cloud, running them in parallel.

        If the user lacks authorization to send multiple jobs to the cloud or exceeds
        the maximum allowed limit, an exception is raised, terminating the launch process.
        Any remaining jobs in the group will not be sent.
        """
        self._launch_jobs(rerun=False)

    def rerun_failed_parallel(self):
        """
        Restart all failed jobs in the group on the Cloud, running them in parallel.

        If the user lacks authorization to send multiple jobs at once or exceeds the maximum allowed limit, an exception
        is raised, terminating the launch process. Any remaining jobs in the group will not be sent.
        """
        self._launch_jobs(rerun=True)

    def get_results(self) -> list[dict]:
        """
        Retrieve results for all jobs in the group. It aggregates results by calling the `get_results()`
        method of each job object that have completed successfully.
        """
        self._update_job_statuses()
        results = []
        for job in self._jobs:
            if job.maybe_completed:
                try:
                    results.append(job.get_results())
                except RuntimeError:
                    results.append(None)
            else:
                results.append(None)
        return results
