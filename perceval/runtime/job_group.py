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
from perceval.utils import PersistentData, FileFormat
from os import path, listdir
import json
import datetime

FILE_EXT_JGRP = 'jgrp'

class JobGroup:
    """
    A JobGroup handles a collection of Jobs saved on disk using perceval
    persistent data. It can perform various tasks, including -
    - Saving a collection of jobs, whether they have been sent to the cloud or not.
    - Running jobs within the group either in parallel or sequentially.
    - Rerunning failed jobs within the group.
    """
    def __init__(self, name: str, load_from_file: bool=False):
        self._file_name = None
        self._name = name
        self.job_records = []
        # list of information (status/id/payload/metadata) for each job in the group
        # todo: rework init - not a good solution
        if not load_from_file:
            if JobGroup._job_group_exists(name):
                raise FileExistsError()
            else:
                self._create_job_group(name)  # creates and saves a job group on disk


    @property
    def name(self):
        return self._name

    def _generate_file_name(self, name):
        # creates file name for the job group with creation date-time stamp added
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_name = f"{name}-{current_timestamp}.{FILE_EXT_JGRP}"

    @staticmethod
    def _get_name_from_filename(filename: str):
        # extracts name of JobGroup from its filename
        name_no_ext = filename.split('.')[0]  # remove extension from filename
        jg_name = name_no_ext.split('-')  # splits name into 'filename' + 'datetime stamp'
        return jg_name[0]

    def _create_job_group(self, name):
        # Saves a job group file on disk, using pcvl.PersistentData(), when a new
        # job group object is instantiated
        if self._file_name is None:
            self._generate_file_name(name)
        ps = PersistentData()
        ps.write_file(self._file_name, json.dumps(self.job_records), FileFormat.TEXT)

    @staticmethod
    def list_saved_job_groups():
        """
        Returns a list of filenames of all JobGroups saved to disk
        """
        saved_jg_files = []
        for filenames in listdir(PersistentData().directory):
            if filenames.endswith(FILE_EXT_JGRP):
                saved_jg_files.append(filenames)
        return saved_jg_files

    @staticmethod
    def _job_group_exists(name: str):
        # returns True if a JobGroup with an identical name is already saved on disk
        jg_files = JobGroup.list_saved_job_groups()

        existing_names = []
        for file in jg_files:
            jg_name = JobGroup._get_name_from_filename(file)
            existing_names.append(jg_name)

        return name in existing_names

    @classmethod
    def load_job_group(cls, filename: str):
        """
        Loads an existing JobGroup from file to create a
        JobGroup object to resume working on it.

        :param filename: Filename of the existing JobGroup to be resumed
        :return: JobGroup object with information of the saved Jobs
        """
        jg_name = JobGroup._get_name_from_filename(filename)
        # initiates an instance of job group using name from the file
        resumed_group = cls(jg_name, load_from_file=True)
        resumed_group._file_name = filename

        # load job records from file and populate the re-created job group with it
        group_data = json.loads(PersistentData().read_file(filename, FileFormat.TEXT))
        resumed_group.job_records = group_data
        return resumed_group

    def _add_job_record(self, job_info: dict):
        # reads existing job records for the current Job Group
        # and modifies it whenever a new job is added
        pd = PersistentData()
        data = json.loads(pd.read_file(self._file_name,
                                       FileFormat.TEXT))
        data.append(job_info)
        pd.write_file(self._file_name, json.dumps(data), FileFormat.TEXT)

    def add(self, job_to_add: RemoteJob, **kwargs):
        """
        Creates a list of Remote Jobs with each entry a dictionary of necessary information
        (status, id, body) of the jobs
        """
        if not isinstance(job_to_add, RemoteJob):
            raise TypeError('Only a RemoteJob can be added to the group')

        job_info = dict()
        job_info['id'] = job_to_add.id

        if job_to_add.id is None:
            # set status to None for Jobs not sent to cloud
            job_info['status'] = None
            job_info['body'] = job_to_add._request_data
            job_info['metadata'] = {'headers': job_to_add._rpc_handler.headers,
                                    'platform': job_to_add._rpc_handler.name,
                                    'url': job_to_add._rpc_handler.url}
            # WARNING: Scaleway RPCHandler members are named _headers, _name and _url todo - find solution

            max_samples = kwargs.get('max_samples', 10000)
            job_info['args'] = {"max_samples": max_samples}
            # todo: do we need to have something else?
        else:
            job_info['status'] = job_to_add.status()
            if not job_to_add.status() == 'SUCCESS':
                # if a job is added that is sent to cloud immediately - this will still be waiting todo - find if we want to change
                job_info['metadata'] = {'headers': job_to_add._rpc_handler.headers,
                                        'platform': job_to_add._rpc_handler.name,
                                        'url': job_to_add._rpc_handler.url}
            # Todo : find if i need metadata for rerun if status=='error'

        self.job_records.append(job_info)
        self._add_job_record(job_info)
        # todo: maybe this can be now replaced with the modify record method i added used by progress

    def _modify_job_record(self, updated_record):
        # called from progress if any job record needs updating
        pd = PersistentData()
        pd.write_file(self._file_name, json.dumps(updated_record), FileFormat.TEXT)

    def progress(self):
        """
        Lists through all jobs in the group and returns a list of status
        If a status is changed from existing in file, job record is updated
        """
        new_job_record = self.job_records
        status_all_jobs = []

        for index, each_job in enumerate(self.job_records):
            # todo : implement something like the following to update status/info
            # if not each_job['status'] in ['SUCCESS', None]:
            #     new_job = RemoteJob().from_id(each_job['id'], rpc_handler='')
            #     new_job_record[index] = info_to_save_from_new_job(new_job)

            status_all_jobs.append(each_job['status'])

        self._modify_job_record(new_job_record)
        self.job_records = new_job_record
        return status_all_jobs

    def run_parallel(self, **kwargs):
        pass
        # todo: refine and implement

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
        list_groups = JobGroup.list_saved_job_groups()
        ps = PersistentData()
        for each_file in list_groups:
            ps.delete_file(each_file)

    @staticmethod
    def delete_job_group(filename: str):
        """
        Delete a single JobGroup file by its name
        :param filename: a JobGroup filename with its extenstion to delete
        """
        ps = PersistentData()
        ps.delete_file(filename)

    def delete_job_groups_date(self, date: str):
        # erase all files with date before given date
        # todo : implement
        # warning ; there is an erase method in PersistentData() -> that can
        # erase all -> add jobgroup to exclude it?
        pass
