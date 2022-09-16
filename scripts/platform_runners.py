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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time

from perceval.platforms.platform import *
import perceval as pcvl
import perceval.components as cp
import numpy as np
from perceval.algorithm import Sampler

cnot = pcvl.Circuit(6, name="Ralph CNOT")
cnot.add((0, 1), cp.GenericBS(R=1 / 3, phi_b=np.pi, phi_d=0))
cnot.add((3, 4), cp.GenericBS(R=1 / 2))
cnot.add((2, 3), cp.GenericBS(R=1 / 3, phi_b=np.pi, phi_d=0))
cnot.add((4, 5), cp.GenericBS(R=1 / 3))
cnot.add((3, 4), cp.GenericBS(R=1 / 2))

try:
    platform = get_platform('Toto')
except RuntimeError as e:
    print(e)

# local_pf_name = 'slos'
# local_pf_name = 'naive'
local_pf_name = 'CliffordClifford2017'


my_platform = get_platform(local_pf_name)
assert not my_platform.is_remote()

sampler = Sampler(my_platform, cnot)

res = sampler.samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
for r in res:
    print(str(r))

nsample = 100000
async_job = sampler.samples.execute_async(pcvl.BasicState([0, 1, 0, 1, 0, 0]), nsample)

while not async_job.is_completed():
    print(f"Waiting for job to finish. Status = {async_job.status()}")
    time.sleep(1)

print(f"Job status = {async_job.status()}")
results = async_job.get_results()
assert len(results) == nsample


# Brain storm 2022-09-15:
# rp = get_platform('SLOS', 'dummy-token')
# rb = rp.backend(cnot)
#
# rb.samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
#
# # runner
# class Sampler(Runner):
#     __init__(platform, cu):
#         self._backend = self._platform.backend(cu)
#
#     @property
#     def sample(self):
#         if self._platform.is_remote:
#             job = RemoteJob(self._backend.sample_async)
#         else:
#             job = LocalJob(self._backend.sample)
#         return job
#
#
# class SLOSSampler(Sampler):  # faux : la plateforme sait qu'elle est SLOS
#
#
# class QML(Runner):
#     __init__(platform)
#
#
# class SamplerFactory:
#     string => Sampler
#
# s = Sampler(platform)
# s.samples(input_state)
#
# my_platform = get_platform('NAIVE', token=pouet, endpoint=cloud)
# res = my_platform.backend(cnot).samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
