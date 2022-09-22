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
from tqdm import tqdm
from perceval.platforms.platform import *
import perceval as pcvl
import perceval.components as cp
import numpy as np
from perceval.algorithm import Sampler, Analyzer

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

local_pf_name = 'CliffordClifford2017'
my_platform = get_platform(local_pf_name)
assert not my_platform.is_remote()

sampler = Sampler(my_platform, cnot)

res = sampler.sample(pcvl.BasicState([0, 1, 0, 1, 0, 0]))
print(res)

nsample = 100000
async_job = sampler.samples.execute_async(pcvl.BasicState([0, 1, 0, 1, 0, 0]), nsample)

previous_prog = 0
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    tq.set_description(f'Get {nsample} samples from {cnot.name} using simulator backend {my_platform.name}')
    while not async_job.is_completed():
        tq.update(async_job.status.progress-previous_prog)
        previous_prog = async_job.status.progress
        time.sleep(.2)
    tq.update(1-previous_prog)
    tq.close()

print(f"Job status = {async_job.status()}")
results = async_job.get_results()
assert len(results) == nsample

chip_QRNG = pcvl.Circuit(4, name='QRNG')
# Parameters
phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
        pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
# Defining the LO elements of chip
(chip_QRNG
 .add((0, 1), cp.SimpleBS())
 .add((2, 3), cp.SimpleBS())
 .add((1, 2), cp.PERM([1, 0]))
 .add(0, cp.PS(phis[0]))
 .add(2, cp.PS(phis[2]))
 .add((0, 1), cp.SimpleBS())
 .add((2, 3), cp.SimpleBS())
 .add(0, cp.PS(phis[1]))
 .add(2, cp.PS(phis[3]))
 .add((0, 1), cp.SimpleBS())
 .add((2, 3), cp.SimpleBS())
 )
# Setting parameters value and see how chip specs evolve
phis[0].set_value(np.pi / 2)
phis[1].set_value(0.2)
phis[2].set_value(0)
phis[3].set_value(0.4)

print("Use circuit analyzer algorithm")
analyzer = Analyzer(get_platform('SLOS'), chip_QRNG, [pcvl.BasicState("[1,0,1,0]"), pcvl.BasicState("[0,1,1,0]")], "*")
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    def update_progress(p):
        tq.update(p-update_progress.prev)
        update_progress.prev = p

    tq.set_description("Analyzing QRNG circuit with SLOS")
    update_progress.prev = 0
    analyzer.compute(progress_callback=update_progress)
    tq.close()

pcvl.pdisplay(analyzer)


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
