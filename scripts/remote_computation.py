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
import perceval.components.unitary_components as cp
import numpy as np
from tqdm import tqdm
import time

from perceval import RemoteProcessor
from perceval.algorithm import Sampler

theta_r13 = cp.BS.r_to_theta(1/3)
cnot = pcvl.Circuit(6, name="Ralph CNOT")
cnot.add((0, 1), cp.BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
cnot.add((3, 4), cp.BS.H())
cnot.add((2, 3), cp.BS.H(theta=theta_r13, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
cnot.add((4, 5), cp.BS.H(theta=theta_r13))
cnot.add((3, 4), cp.BS.H())


[phi_0, phi_1, phi_2, phi_3] = [pcvl.P("p_{0}".format(i)) for i in range(4)]

c = (pcvl.Circuit(4, name="Q4R")
     .add(0, cp.BS())
     .add(2, cp.BS())
     .add(1, cp.PERM([1, 0]))
     .add(0, cp.PS(phi_0))
     .add(0, cp.BS())
     .add(0, cp.PS(phi_1))
     .add(0, cp.BS())
     .add(2, cp.PS(phi_2))
     .add(2, cp.BS())
     .add(2, cp.PS(phi_3))
     .add(2, cp.BS()))

phi_0.set_value(0)
phi_1.set_value(1)
phi_2.set_value(2)
phi_3.set_value(3)

U = c.compute_unitary(use_symbolic=False)


token_qcloud = 'YOUR_API_KEY'
platform_url = "https://api.cloud.quandela.dev"


remote_simulator = RemoteProcessor("qpu", token_qcloud, platform_url)
specific_circuit = remote_simulator.specs['specific_circuit']

remote_simulator.set_circuit(U)
remote_simulator.with_input(pcvl.BasicState([1, 0, 1, 0]))

sampler = Sampler(remote_simulator)

nsample = 10000
async_job = sampler.sample_count.execute_async(nsample)

previous_prog = 0
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    tq.set_description(f'Get {nsample} samples from {remote_simulator.name}')
    while not async_job.is_completed():
        tq.update(async_job.status.progress/100-previous_prog)
        previous_prog = async_job.status.progress/100
        time.sleep(1)
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
