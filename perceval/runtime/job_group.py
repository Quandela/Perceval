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
from perceval.utils.logging import logger, channel
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
            logger.info(f'Job Group with name {name} exists, loading from disk', channel.general)
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
    def jobs_record(self):
        """
        List of necessary information (id,status,metadata) for each job within the group
        """
        return self._group_info['jobs_record']

    @staticmethod
    def _get_current_datetime():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _get_jgrp_dir():
        return os.path.join(PersistentData().directory, "job_group")

    @staticmethod
    def _setup_jgrp_dir(dir_path):
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
        # Saves the records of a new job group on disk using PersistentData(),
        jgrp_dir_path = JobGroup._get_jgrp_dir()
        JobGroup._setup_jgrp_dir(jgrp_dir_path)  # create directory and validat permissions

        self._group_info['created_date'] = JobGroup._get_current_datetime()
        self._group_info['modified_date'] = JobGroup._get_current_datetime()
        self._group_info['jobs_record'] = []
        # write to disk
        self._write_job_group_to_disk()

    def _read_job_group_from_disk(self) -> dict:
        # returns the Job group data stored on disk
        jgrp_dir = JobGroup._get_jgrp_dir()
        ps = PersistentData()
        group_data = json.loads(ps.read_file(os.path.join(jgrp_dir, self._file_name),
                                             FileFormat.TEXT))
        return group_data

    def _write_job_group_to_disk(self):
        # writes job group data to disk
        jgrp_dir = JobGroup._get_jgrp_dir()
        PersistentData().write_file(os.path.join(jgrp_dir, self._file_name),
                                    json.dumps(self._group_info), FileFormat.TEXT)

    def _load_job_group(self):
        # Creates a Job Group by loading an existing one from file
        group_data = self._read_job_group_from_disk()

        self._group_info['created_date'] = group_data['created_date']
        self._group_info['modified_date'] = JobGroup._get_current_datetime()
        self._group_info['jobs_record'] = group_data['jobs_record']
        # Write to disk (necessary to update modification time)
        self._write_job_group_to_disk()

    @staticmethod
    def list_saved_job_groups():
        """
        Returns a list of filenames of all JobGroups saved to disk
        """
        jgrp_path = JobGroup._get_jgrp_dir()
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
        Saves the data in a choronological order in the group record (each entry is
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
        # include the added job's info to job records
        self._modify_job_record(job_info)

    def _modify_job_record(self, updated_record: dict):
        # modifies the recorded information when a new job is added
        group_data = self._read_job_group_from_disk()

        saved_job_record = group_data['jobs_record']
        saved_job_record.append(updated_record)
        self._group_info['jobs_record'] = saved_job_record
        # save changes to disk
        self._write_job_group_to_disk()

    def progress(self):
        """
        Lists through all jobs in the group and returns a list of status
        If a status is changed from existing in file, job record is updated
        """
        new_job_record = self._group_info['jobs_record']
        status_all_jobs = []

        for index, each_job in enumerate(self._group_info['jobs_record']):
            # todo : implement something like the following to update status/info
            # if not each_job['status'] in ['SUCCESS', None]:
            #     new_job = RemoteJob().from_id(each_job['id'], rpc_handler='')
            #     new_job_record[index] = info_to_save_from_new_job(new_job)
            if not each_job['status'] in ['SUCCESS', None]:
                rj = self._recreate_remote_job_from_record(each_job)
            status_all_jobs.append(each_job['status'])

        self._modify_job_record(new_job_record)
        self._job_records = new_job_record
        return status_all_jobs

    def _recreate_remote_job_from_record(self, each_job):
        platform_metadata = each_job['metadata']
        _token = platform_metadata['headers']['Authorization'].split(' ')[1]
        rpc_handler = RPCHandler(platform_metadata['platform'], platform_metadata['url'], _token)
        rj_again = RemoteJob().from_id(each_job['id'], rpc_handler)
        return rj_again

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
        # warning ; there is an erase method in PersistentData() -> that can
        # todo : add "jgrp" extension to exclude it
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
        jgrp_dir_path = JobGroup._get_jgrp_dir()

        PersistentData().delete_file(os.path.join(jgrp_dir_path, filename))

    def delete_job_groups_date(self, date: str):
        # erase all files with date before given date
        # todo : implement
        # warning ; there is an erase method in PersistentData() -> that can
        # erase all -> add jobgroup to exclude it?
        pass
