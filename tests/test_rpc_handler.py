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
"""module test rpc handler"""

from _mock_rpc_handler import get_rpc_handler


def test_build_endpoint(requests_mock):
    """test build endpoint function"""
    rpc = get_rpc_handler(requests_mock, url='https://example.org')

    wanted = 'https://example.org/endpoint'
    assert rpc.build_endpoint('/endpoint') == wanted

    wanted = 'https://example.org/endpoint/id'
    assert rpc.build_endpoint('/endpoint', 'id') == wanted

    wanted = 'https://example.org/endpoint/path/id'
    assert rpc.build_endpoint('/endpoint', 'path', 'id') == wanted

    wanted = 'https://example.org/endpoint/with/trailing/slash'
    assert rpc.build_endpoint('/endpoint/', 'with', 'trailing/slash') == wanted

    wanted = 'https://example.org/endpoint/path/with/slash'
    assert rpc.build_endpoint('/endpoint/', '/path/', 'with/', '/slash/') == wanted

    wanted = 'https://example.org/endpoint/with/number/1/8/789'
    assert rpc.build_endpoint('/endpoint/', '/with/number', 1, '/8/', 789) == wanted


def test_rpc_handler(requests_mock):
    """test rpc handler calls"""
    rpc = get_rpc_handler(requests_mock, url='https://example.org')
    resp_details = rpc.fetch_platform_details()
    assert 'name' in resp_details
    job_id = rpc.create_job({})
    assert isinstance(job_id, str)
    resp_status = rpc.get_job_status(job_id)
    assert resp_status['msg'] == 'ok'
    rpc.cancel_job(job_id)
    resp_result = rpc.get_job_results(job_id)
    assert 'results' in resp_result
    new_job_id = rpc.rerun_job(job_id)
    assert new_job_id is not None
    assert new_job_id != job_id
