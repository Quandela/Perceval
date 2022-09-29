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
import perceval.components.base_components as cp
import numpy as np
from tqdm import tqdm
import time

from perceval.algorithm import Sampler

theta_r13 = cp.BS.r_to_theta(1/3)
cnot = pcvl.Circuit(6, name="Ralph CNOT")
cnot.add((0, 1), cp.BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
cnot.add((3, 4), cp.BS.H())
cnot.add((2, 3), cp.BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
cnot.add((4, 5), cp.BS.H(theta=theta_r13))
cnot.add((3, 4), cp.BS.H())

token_qcloud = '_T_eyJhbGciOiJIUzUxMiIsImlhdCI6MTY2NDQ1Mzc0OSwiZXhwIjoxNjY3MDQ1NzQ5fQ.eyJpZCI6OH0.Y9kKZaZ9xX_pN89uE1Z3niRkqAK1EFeECjNtIeQK045KyHuGcWBKTIces2VAUoQNTAAwgfwgwcW9OSd9NUm14Q'
platform_url = "https://api.cloud.quandela.dev"

naive_remote_platform = pcvl.get_platform("AbaertLaptop", token_qcloud, platform_url)

sampler = Sampler(naive_remote_platform, cnot)

nsample = 10000
async_job = sampler.sample_count.execute_async(pcvl.BasicState([0, 1, 0, 1, 0, 0]), nsample)

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
print(results)
#assert len(results) == nsample

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
