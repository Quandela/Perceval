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
import warnings

from perceval.runtime import RemoteJob
from perceval.runtime.rpc_handler import RPCHandler
from perceval.utils import PersistentData, FileFormat
from perceval.utils.logging import get_logger, channel
import os
import json
import datetime

FILE_EXT_JGRP = 'jgrp'

class JobGroup:
    """
    A JobGroup handles a collection of Jobs saved on disk using perceval
    persistent data. It can perform various tasks such as
    - Saving information for a collection of jobs, whether they have been sent to the cloud or not.
    - Running jobs within the group either in parallel or sequentially.
    - Rerunning failed jobs within the group.
    """
    def __init__(self, name: str):
        self._name = name
        self._group_info = dict()
        self._file_name = f"{self._name}.{FILE_EXT_JGRP}"

        if self._job_group_exists(name):
            get_logger().info(f'Job Group with name {name} exists, loading from disk', channel.general)
            self._load_job_group()
        else:
            self._create_job_group()


    @property
    def name(self):
        """
        Name of the job Group [also the filename (without extension) on disk]
        """
        return self._name

    @property
    def job_group_data(self):
        """
        List of necessary information (id,status,metadata) for each job within the group
        """
        return self._group_info['job_group_data']

    @staticmethod
    def _get_current_datetime():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _get_job_group_dir():
        return os.path.join(PersistentData().directory, "job_group")

    @staticmethod
    def _setup_job_group_dir(dir_path):
        # creates a sub folder in persistent data directory if non-existent
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError as exc:
            warnings.warn(UserWarning(f"{exc}"))

        # validate read/write permissions for the directory
        if not os.access(dir_path, os.W_OK) or not os.access(dir_path, os.R_OK):
            warnings.warn(UserWarning(f"Cannot read or write in {dir_path}"))

    def _create_job_group(self):
        # Saves information for a new job group on disk using PersistentData(),
        jgrp_dir_path = JobGroup._get_job_group_dir()
        JobGroup._setup_job_group_dir(jgrp_dir_path)  # create directory and validat permissions

        self._group_info['created_date'] = JobGroup._get_current_datetime()
        self._group_info['modified_date'] = JobGroup._get_current_datetime()
        self._group_info['job_group_data'] = []
        # write to disk
        self._write_job_group_to_disk()

    def _read_job_group_from_disk(self) -> dict:
        # returns the Job group data stored on disk
        jgrp_dir = JobGroup._get_job_group_dir()
        ps = PersistentData()
        group_data = json.loads(ps.read_file(os.path.join(jgrp_dir, self._file_name),
                                             FileFormat.TEXT))
        return group_data

    def _write_job_group_to_disk(self):
        # writes job group data to disk
        jgrp_dir = JobGroup._get_job_group_dir()
        PersistentData().write_file(os.path.join(jgrp_dir, self._file_name),
                                    json.dumps(self._group_info), FileFormat.TEXT)

    def _load_job_group(self):
        # Creates a Job Group by loading an existing one from file
        group_data = self._read_job_group_from_disk()

        self._group_info['created_date'] = group_data['created_date']
        self._group_info['modified_date'] = JobGroup._get_current_datetime()
        self._group_info['job_group_data'] = group_data['job_group_data']
        # Write to disk (necessary to update modification time)
        self._write_job_group_to_disk()

    @staticmethod
    def list_saved_job_groups():
        """
        Returns a list of filenames of all JobGroups saved to disk
        """
        jgrp_path = JobGroup._get_job_group_dir()
        files = [f for f in os.listdir(jgrp_path) if f.endswith(FILE_EXT_JGRP)]
        return files

    @staticmethod
    def _job_group_exists(name: str):
        # returns True if a JobGroup with an identical name is already saved on disk
        saved_job_groups = JobGroup.list_saved_job_groups()
        return any([filename.split('.')[0] == name for filename in saved_job_groups])

    def add(self, job_to_add: RemoteJob, **kwargs):
        """
        Adds information of the new RemoteJob to an existing Group.
        Saves the data in a chronological order in the group (each entry is
        a dictionary of necessary information - status, id, body, metadata)
        """
        if not isinstance(job_to_add, RemoteJob):
            raise TypeError('Only a RemoteJob can be added to the group')

        job_info = dict()
        job_info['id'] = job_to_add.id

        if job_to_add.id is None:
            # set status to None for Jobs not sent to cloud
            job_info['status'] = None
            # Save information inside body (circuit, payload, etc.) to send job later
            job_info['body'] = job_to_add._request_data
            max_samples = kwargs.get('max_samples', 10000)
            job_info['args'] = {"max_samples": max_samples}
        else:
            job_info['status'] = job_to_add.status()

        if not job_info['status'] == 'SUCCESS':
            # save metadata for any job that did not run succesfully
            job_info['metadata'] = {'headers': job_to_add._rpc_handler.headers,
                                    'platform': job_to_add._rpc_handler.name,
                                    'url': job_to_add._rpc_handler.url}

        # include the added job's info to dataset in the group
        self._modify_job_dataset(job_info)

    def _modify_job_dataset(self, updated_info: dict):
        # modifies the recorded information when a new job is added
        group_data = self._read_job_group_from_disk()

        saved_job_info = group_data['job_group_data']
        saved_job_info.append(updated_info)

        self._group_info['job_group_data'] = saved_job_info  # save changes in object
        self._write_job_group_to_disk()  # save changes to disk

    def progress(self):
        """
        Lists through all jobs in the group and returns a list of status
        If a status is changed from existing in file, that entry is
        updated with new information
        """
        status_jobs_in_group = []

        for job_entry in self._group_info['job_group_data']:
            if not job_entry['status'] in ['SUCCESS', None]:
                rj = self._recreate_remote_job_from_stored_data(job_entry)
                rj_status = rj.status()
                if rj_status == 'SUCCESS':
                    # remove metadata information if job ran successfully
                    del job_entry['metadata']
                job_entry['status'] = rj_status  # update current status

            status_jobs_in_group.append(job_entry['status'])

        self._write_job_group_to_disk()  # rewrites the group information on disk

        # replace None with 'Not_sent' - more readable
        status_list = [status if status is not None else "Not_sent" for status in status_jobs_in_group]
        return status_list

    def _recreate_remote_job_from_stored_data(self, job_entry) -> RemoteJob:
        # returs a RemoteJob object recreated using its id and platform metadata
        platform_metadata = job_entry['metadata']
        user_token = platform_metadata['headers']['Authorization'].split(' ')[1]
        rpc_handler = RPCHandler(platform_metadata['platform'],
                                 platform_metadata['url'], user_token)
        rj_recreated = RemoteJob.from_id(job_entry['id'], rpc_handler)
        return rj_recreated

    def run_parallel(self, **kwargs):
        # todo: refine and implement
        pass

    def run_sequential(self, delay: float, **kwargs):
        # todo : refine and implement
        # if delay:
            # run each after delay
        pass

    def rerun_failed(self):
        pass
        # todo: implement
        # will be based on the API from Cloud side

    @staticmethod
    def delete_all_job_groups():
        """
        To delete all existing Job groups on disk
        """
        jgrp_dir_path = JobGroup._get_jgrp_dir()
        list_groups = JobGroup.list_saved_job_groups()
        ps = PersistentData()
        for each_file in list_groups:
            ps.delete_file(os.path.join(jgrp_dir_path, each_file))

    @staticmethod
    def delete_job_group(filename: str):
        """
        Delete a single JobGroup file by its name
        :param filename: a JobGroup filename with its extenstion to delete
        """
        jgrp_dir_path = JobGroup._get_job_group_dir()

        PersistentData().delete_file(os.path.join(jgrp_dir_path, filename))

    @staticmethod
    def delete_job_groups_date(del_before_date: int):
        """
        Delete all saved Job Groups created before a date (not included).
        :param del_before_date: integer (form - YYYYMMDD) files created before this date deleted
        """
        existing_groups = JobGroup.list_saved_job_groups()
        files_to_del = []  # list of files before date to delete
        for file in existing_groups:
            jg_name = file.split('.')[0]  # remove extension
            jg = JobGroup(jg_name)
            jg_datetime = jg._group_info['created_date']
            jg_date = int(jg_datetime.split('_')[0])
            if jg_date < del_before_date:
                files_to_del.append(file)

        if not files_to_del:
            warnings.warn(UserWarning, f'No files found to delete before{del_before_date}')

        # delete files
        for f in files_to_del:
            JobGroup.delete_job_group(f)
