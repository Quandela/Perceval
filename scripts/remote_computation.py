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

import perceval as pcvl
import perceval.lib.phys as phys
import sympy as sp
import numpy as np
from tqdm import tqdm
import time

from perceval.algorithm import Sampler

cnot = phys.Circuit(6, name="Ralph CNOT")
cnot.add((0, 1), phys.BS(R=1 / 3, phi_b=sp.pi, phi_d=0))
cnot.add((3, 4), phys.BS(R=1 / 2))
cnot.add((2, 3), phys.BS(R=1 / 3, phi_b=sp.pi, phi_d=0))
cnot.add((4, 5), phys.BS(R=1 / 3))
cnot.add((3, 4), phys.BS(R=1 / 2))
# pcvl.pdisplay(cnot)


# local_simulator_backend = pcvl.get_platform('local').get_backend("Naive")
# local_s_cnot = local_simulator_backend(cnot.U)
# results = local_s_cnot.samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 10000)
# for s in results:
#     print(str(s))

token_qcloud = '_T_eyJhbGciOiJIUzUxMiIsImlhdCI6MTY2MjQ3NjY3NSwiZXhwIjoxNjY1MDY4Njc1fQ.eyJpZCI6Mn0.wmBX9LU0T05XTpYhBcGRNeeusFh-Rmt4rk4g1Po0YGPoo_LMlUa4-l1nVD_jn9z2_LYqesO3RTw3DVR3w0EJhw'
platform_url = "http://127.0.0.1:5001"

naive_remote_platform = pcvl.get_platform("Naive", token_qcloud, platform_url)


sampler = Sampler(naive_remote_platform, cnot)

nsample = 10000
async_job = sampler.samples.execute_async(pcvl.BasicState([0, 1, 0, 1, 0, 0]), nsample)

previous_prog = 0
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    tq.set_description(f'Get {nsample} samples from {cnot.name} using simulator backend {naive_remote_platform.name}')
    while not async_job.is_completed():
        tq.update(async_job.status.progress-previous_prog)
        previous_prog = async_job.status.progress
        time.sleep(.2)
    tq.update(1-previous_prog)
    tq.close()

print(f"Job status = {async_job.status()}")
results = async_job.get_results()
assert len(results) == nsample

#
# results = job.get_results()
#
# results2 = s_cnot.samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
#
#
# states = {
#     pcvl.BasicState([0, 1, 0, 1, 0, 0]): "00",
#     pcvl.BasicState([0, 1, 0, 0, 1, 0]): "01",
#     pcvl.BasicState([0, 0, 1, 1, 0, 0]): "10",
#     pcvl.BasicState([0, 0, 1, 0, 1, 0]): "11"
# }
#
# ca = pcvl.CircuitAnalyser(s_cnot, states)
# ca.compute(expected={"00": "00", "01": "01", "10": "11", "11": "10"})
# pcvl.pdisplay(ca)
# print("performance=%s, error rate=%.3f%%" % (pcvl.simple_float(ca.performance)[1], ca.error_rate))
