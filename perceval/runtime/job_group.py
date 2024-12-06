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

from perceval.runtime import RemoteJob, RunningStatus
from perceval.runtime.rpc_handler import RPCHandler
from perceval.utils import PersistentData, FileFormat
from perceval.utils.logging import get_logger, channel
import os
import json
from datetime import datetime
from tqdm import tqdm


FILE_EXT_JGRP = 'jgrp'
JGRP_DIR_NAME = "job_group"

DATE_TIME_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_MAX_SAMPLES = 10000


class JobGroup:
    """
    A JobGroup handles a collection of Jobs saved on disk using perceval
    persistent data. It can perform various tasks such as
    - Saving information for a collection of jobs, whether they have been sent to the cloud or not.
    - Running jobs within the group either in parallel or sequentially.
    - Rerunning failed jobs within the group.

    :param name: a name for the group of jobs (also, the filename used to save JobGroup on disk)
    """
    _JGRP_PERSISTENT_DATA = PersistentData()  # Persistent data object for the job group class
    _JGRP_DIR_PATH = os.path.join(_JGRP_PERSISTENT_DATA.directory, JGRP_DIR_NAME)

    def __init__(self, name: str):
        self._name = name
        self._group_info = dict()
        self._file_path =  os.path.join(JobGroup._JGRP_DIR_PATH, f"{self._name}.{FILE_EXT_JGRP}")

        if self._job_group_exists(name):
            get_logger().info(f'Job Group with name {name} exists; subsequent jobs will be appended to it',
                              channel.user)
            self._load_job_group()
        else:
            self._create_job_group()

    @property
    def name(self) -> str:
        """
        Name of the job Group [also the filename (without extension) on disk]
        """
        return self._name

    @property
    def list_remote_jobs(self) -> list[RemoteJob]:
        """
        Returns a chronologically ordered list of RemoteJobs in the group.
        Jobs never sent to the cloud will be represented by None.
        """
        list_rj = []
        for job_entry in self._group_info['job_group_data']:
            if job_entry['status'] != None:
                rj = self._recreate_remote_job_from_stored_data(job_entry)
                list_rj.append(rj)
            else:
                list_rj.append(None)

        return  list_rj

    @staticmethod
    def _get_current_datetime() -> str:
        # returns current datetime as string in format DATE_TIME_FORMAT
        return datetime.now().strftime(DATE_TIME_FORMAT)

    def _create_job_group(self):
        """
        Saves information for a new job group on disk using PersistentData()
        """
        JobGroup._JGRP_PERSISTENT_DATA.create_sub_directory(JGRP_DIR_NAME)  # create directory and validate permissions

        current_time = JobGroup._get_current_datetime()
        self._group_info['created_date'] = current_time
        self._group_info['modified_date'] = current_time
        self._group_info['job_group_data'] = []
        # write to disk
        self._write_job_group_to_disk()

    def _read_job_group_from_disk(self) -> dict:
        """
        Returns the Job group data stored on disk
        """
        group_data = json.loads(JobGroup._JGRP_PERSISTENT_DATA.read_file(self._file_path, FileFormat.TEXT))
        return group_data

    def _write_job_group_to_disk(self):
        """
        Writes job group data to disk
        """
        JobGroup._JGRP_PERSISTENT_DATA.write_file(self._file_path,
                                    json.dumps(self._group_info), FileFormat.TEXT)

    def _load_job_group(self):
        """
        Creates a Job Group by loading an existing one from file
        """
        group_data = self._read_job_group_from_disk()

        self._group_info['created_date'] = group_data['created_date']
        self._group_info['modified_date'] = JobGroup._get_current_datetime()
        self._group_info['job_group_data'] = group_data['job_group_data']
        # Write to disk (necessary to update modification time)
        self._write_job_group_to_disk()

    @staticmethod
    def list_existing() -> list[str]:
        """
        Returns a list of filenames of all JobGroups saved to disk
        """
        jgrp_path = JobGroup._JGRP_DIR_PATH
        files = [os.path.splitext(f)[0] for f in os.listdir(jgrp_path) if f.endswith(FILE_EXT_JGRP)]
        return files

    @staticmethod
    def _job_group_exists(name: str) -> bool:
        """
        Returns True if a JobGroup with an identical name is already saved on disk
        """
        return JobGroup._JGRP_PERSISTENT_DATA.has_file(os.path.join(JobGroup._JGRP_DIR_PATH, name + '.' + FILE_EXT_JGRP))

    def add(self, job_to_add: RemoteJob, **kwargs):
        """
        Adds information of the new RemoteJob to an existing Group.
        Saves the data in a chronological order in the group (each entry is
        a dictionary of necessary information - status, id, body, metadata)

        :param job_to_add: a remote job to add to the list of existing job group
        """
        if not isinstance(job_to_add, RemoteJob):
            raise TypeError('Only a RemoteJob can be added to the group')

        # Reject adding a duplicate RemoteJob
        curr_grp_data = self._group_info['job_group_data']
        curr_job_ids = [data['id'] for data in curr_grp_data if data['id'] is not None]

        if job_to_add.id in curr_job_ids:
            raise ValueError(f"Duplicate job detected : job id {job_to_add.id} exists in the group.")

        job_info = dict()
        job_info['id'] = job_to_add.id

        if job_to_add.id is None:
            job_info['status'] = None  # set status to None for Jobs not sent to cloud
            job_info['body'] = job_to_add._create_payload_data(**kwargs)  # Save job payload to launch later on cloud
        else:
            job_info['status'] = job_to_add.status()

        # save metadata to recreate remote jobs
        job_info['metadata'] = {'headers': job_to_add._rpc_handler.headers,
                                'platform': job_to_add._rpc_handler.name,
                                'url': job_to_add._rpc_handler.url}

        self._group_info['job_group_data'].append(job_info)  # save changes in object
        self._modify_job_dataset(job_info)  # include the added job's info to dataset in the group

    def _modify_job_dataset(self, updated_info: dict):
        """
        Modifies the recorded information when a new job is added
        """
        group_data = self._read_job_group_from_disk()

        saved_job_info = group_data['job_group_data']
        saved_job_info.append(updated_info)

        self._write_job_group_to_disk()  # save changes to disk

    def _collect_job_statuses(self) -> list:
        """
        Lists through all jobs in the group and returns a list of status
        If a status is changed from existing in file, that entry is
        updated with new information
        """
        status_jobs_in_group = []

        for job_entry in self._group_info['job_group_data']:
            if job_entry['status'] not in ['SUCCESS', None]:
                rj = self._recreate_remote_job_from_stored_data(job_entry)
                job_entry['status'] = rj.status()  # update with current status

            status_jobs_in_group.append(job_entry['status'])

        self._write_job_group_to_disk()  # rewrites the group information on disk

        return status_jobs_in_group

    @staticmethod
    def _map_job_status_category(status_entry: str):
        # status categories
        status_success = [RunningStatus.SUCCESS]
        status_sent = [RunningStatus.RUNNING, RunningStatus.WAITING, RunningStatus.CANCEL_REQUESTED]
        status_other = [RunningStatus.ERROR, RunningStatus.CANCELED, RunningStatus.SUSPENDED, RunningStatus.UNKNOWN]

        if status_entry is None:
            return 'UNFIN_NOT_SENT'
        elif RunningStatus[status_entry] in status_sent:
            return 'UNFIN_SENT'
        elif RunningStatus[status_entry] in status_success:
            return 'FIN_SUCCESS'
        elif RunningStatus[status_entry] in status_other:
            return 'FIN_OTHER'
        else:
            raise ValueError(f"Unspecified status of job in group with value {status_entry}. Cannot categorize")

    def progress(self) -> dict:
        """
        Iterates over all jobs in the group to create a dictionary of display
        the current status of jobs in a tabular form. Jobs in the group are
        categorized as follows (depending on their RunningStatus on the Cloud)
            - Finished
                -- successful {'SUCCESS'}
                -- unsuccessful {'CANCELED', 'ERROR', 'UNKNOWN', 'SUSPENDED'}
            - Unfinished
                -- sent {'WAITING', 'RUNNING', 'CANCEL_REQUESTED'}
                -- not sent {None}
        """
        success_job_cnt = 0
        other_job_cnt = 0
        sent_job_cnt = 0
        unsent_job_cnt = 0

        job_statuses = self._collect_job_statuses()

        for status_entry in job_statuses:
            job_category = JobGroup._map_job_status_category(status_entry)
            if job_category == 'UNFIN_NOT_SENT':
                unsent_job_cnt += 1
            elif job_category == 'UNFIN_SENT':
                sent_job_cnt += 1
            elif job_category == 'FIN_SUCCESS':
                success_job_cnt += 1
            elif job_category == 'FIN_OTHER':
                other_job_cnt += 1

        fin_job_prog = {'successful': success_job_cnt, 'unsuccessful': other_job_cnt}
        unfin_job_prog = {'sent': sent_job_cnt, 'not sent': unsent_job_cnt}

        progress = dict()
        progress['Total'] = len(self._group_info['job_group_data'])
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
        tot_jobs = len(self._group_info['job_group_data'])

        # define tqdm bars
        bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
        success_bar = tqdm(total=tot_jobs, bar_format=bar_format, desc="Successful Jobs", position=0, leave=True)
        active_bar = tqdm(total=tot_jobs, bar_format=bar_format,desc ="Running/Waiting Jobs", position=1, leave=True)
        inactive_bar = tqdm(total=tot_jobs, bar_format=bar_format, desc="Inactive/Unsuccessful Jobs", position=2, leave=True)

        while True:
            status_list = self._collect_job_statuses()  # list of statuses for jobs
            group_categories = []

            for job_index in range(tot_jobs):
                job_category = JobGroup._map_job_status_category(status_list[job_index])
                group_categories.append(job_category)

                if job_category == 'FIN_SUCCESS':
                    success_bar.update(1)
                elif job_category == 'UNFIN_SENT':
                    active_bar.update(1)
                elif job_category in ['FIN_OTHER', 'UNFIN_NOT_SENT']:
                        inactive_bar.update(1)

            # category list from status
            if not any(category == 'UNFIN_SENT' for category in group_categories):
                # exit if no jobs running/waiting on cloud
                break

        success_bar.close()
        active_bar.close()
        inactive_bar.close()

    def _recreate_remote_job_from_stored_data(self, job_entry) -> RemoteJob:
        """
        Returns a RemoteJob object recreated using its id and platform metadata
        """
        platform_metadata = job_entry['metadata']
        user_token = platform_metadata['headers']['Authorization'].split(' ')[1]

        rpc_handler = RPCHandler(platform_metadata['platform'],
                                 platform_metadata['url'], user_token)

        return RemoteJob.from_id(job_entry['id'], rpc_handler)

    @staticmethod
    def delete_all_job_groups():
        """
        To delete all existing Job groups on disk
        """
        jgrp_dir_path = JobGroup._JGRP_DIR_PATH
        list_groups = JobGroup.list_existing()
        for each_file in list_groups:
            JobGroup._JGRP_PERSISTENT_DATA.delete_file(os.path.join(jgrp_dir_path, each_file + '.' + FILE_EXT_JGRP))

    @staticmethod
    def delete_job_group(group_name: str):
        """
        Delete a single JobGroup file by its name
        :param group_name: a JobGroup name to delete
        """
        jgrp_dir_path = JobGroup._JGRP_DIR_PATH

        JobGroup._JGRP_PERSISTENT_DATA.delete_file(os.path.join(jgrp_dir_path, group_name + '.' + FILE_EXT_JGRP))

    @staticmethod
    def delete_job_groups_date(del_before_date: datetime):
        """
        Delete all saved Job Groups created before a date (not included).
        :param del_before_date: integer (form - YYYYMMDD) files created before this date deleted
        """
        existing_groups = JobGroup.list_existing()
        files_to_del = []  # list of files before date to delete
        for jg_name in existing_groups:
            jg = JobGroup(jg_name)
            jg_datetime = datetime.strptime(jg._group_info['created_date'], DATE_TIME_FORMAT)
            if jg_datetime < del_before_date:
                files_to_del.append(jg_name)

        if not files_to_del:
            get_logger().warn(f'No files found to delete before{del_before_date}', channel.user)

        # delete files
        for f in files_to_del:
            JobGroup.delete_job_group(f)

    def _list_jobs_status_type(self, statuses: list[str] = None) -> list[RemoteJob]:
        remote_jobs = []
        if statuses is None:
            # handles unsent jobs with given statuses
            for job_entry in self._group_info['job_group_data']:
                if job_entry['status'] is None:
                    remote_jobs.append(self._recreate_unsent_remote_job(job_entry))
            return remote_jobs
        else:
            # handles jobs sent to cloud
            for job_entry in self._group_info['job_group_data']:
                if job_entry['status'] in statuses:
                    remote_jobs.append(self._recreate_remote_job_from_stored_data(job_entry))
            return remote_jobs

    def list_successful_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that have run successfully on the cloud.
        """
        return self._list_jobs_status_type(['SUCCESS'])

    def list_active_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all RemoteJobs in the group that are currently active on the cloud - those
        with RUNNINGSTATUS - RUNNING or WAITING.
        """
        return self._list_jobs_status_type(['RUNNING', 'WAITING'])

    def list_failed_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all the failed RemoteJobs in the group - includes those with
         RUNNINGSTATUS -> ERROR or CANCELED.
        """
        return self._list_jobs_status_type(['ERROR', 'CANCELED'])

    @staticmethod
    def _recreate_unsent_remote_job(job_entry):
        platform_metadata = job_entry['metadata']
        user_token = platform_metadata['headers']['Authorization'].split(' ')[1]

        rpc_handler = RPCHandler(platform_metadata['platform'],
                                 platform_metadata['url'], user_token)
        return RemoteJob(request_data=job_entry['body'], rpc_handler=rpc_handler,
                         job_name=job_entry['body']['job_name'])

    def list_unsent_jobs(self) -> list[RemoteJob]:
        """
        Returns a list of all the RemoteJobs in the group that were not sent to the cloud.
        """
        return self._list_jobs_status_type()
