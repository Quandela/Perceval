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

# from perceval.runtime import JobGroup, RemoteJob, RunningStatus
from perceval.runtime.rpc_handler import RPCHandler
from datetime import datetime

DATE_TIME_FORMAT = "%Y%m%d_%H%M%S"

def serialize_job_group(job_group) -> dict:
    from perceval.runtime import JobGroup
    group_data = {'created_date': job_group.created_date.strftime(DATE_TIME_FORMAT),
                  'modified_date': job_group.modified_date.strftime(DATE_TIME_FORMAT),
                  'job_group_data': []}
    for job in job_group._jobs:
        dict_job = _serialize_remote_job(job) #.to_dict()  # move rj serialize her in private method
        group_data['job_group_data'].append(dict_job)
    return group_data

def deserialize_job_group(job_group_data: dict, group_name: str):
    # TODO: what about name? this will actually cause a circular loop.
    #  recreating a jobgroup from name -> want to load from disk -> that will call deserialize to convert dict to jobgroup
    from perceval.runtime import JobGroup
    jg = JobGroup(group_name)
    jg.created_date = datetime.strptime(job_group_data['created_date'], DATE_TIME_FORMAT)
    jg.modified_date = datetime.strptime(job_group_data['modified_date'], DATE_TIME_FORMAT)
    for job_entry in job_group_data['job_group_data']:
        jg._jobs.append(_build_remote_job(job_entry))
#
#     return jg


def _serialize_remote_job(rj) -> dict:
    from perceval.runtime import RemoteJob
    job_info = dict()
    job_info['id'] = rj.id

    if rj.was_sent:
        job_info['status'] = str(rj._job_status)
    else:
        job_info['status'] = None  # set status to None for Jobs not sent to cloud
    job_info['body'] = rj._create_payload_data()  # Save job payload to launch later on cloud

    # save metadata to recreate remote jobs
    # TODO: from/to_dict directly in rpc_handler -> this is from Marion
    job_info['metadata'] = {'headers': rj._rpc_handler.headers,
                            'platform': rj._rpc_handler.name,
                            'url': rj._rpc_handler.url}

    return job_info

def _deserialize_remote_job(rj_data: dict, rpc_handler: RPCHandler):
    from perceval.runtime import RemoteJob, RunningStatus
    rj = RemoteJob(rj_data['body'], rpc_handler, rj_data['body']['job_name'])
    rj._id = rj_data['id']
    if rj_data['status'] is not None:
        rj._job_status.status = RunningStatus[rj_data['status']]
    return rj

def _build_remote_job(job_entry: dict):
    """
    Returns a RemoteJob object recreated using its id and platform metadata
    """
    platform_metadata = job_entry['metadata']
    user_token = platform_metadata['headers']['Authorization'].split(' ')[1]

    rpc_handler = RPCHandler(platform_metadata['platform'],
                             platform_metadata['url'], user_token)

    return _deserialize_remote_job(job_entry, rpc_handler)
    # return RemoteJob.from_dict(job_entry, rpc_handler)
