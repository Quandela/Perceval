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
import time

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

token_qcloud = '_T_eyJhbGciOiJIUzUxMiIsImlhdCI6MTY2MDc0MzU0MCwiZXhwIjoxNjYzMzM1NTQwfQ.eyJpZCI6MX0.g3u2J1hogz3RRvoh-WTnkp4aD1KdHZw2q0tQh_HolX8nSDlTPlYMNiqNbj6rkFfd8nn7gamdjhd01xUoznwaNw'
credentials = pcvl.RemoteCredentials(url="http://127.0.0.1:5001", token=token_qcloud)

simulator_backend = pcvl.get_platform(credentials).get_backend("Naive")
s_cnot = simulator_backend(cnot.U)

# job = s_cnot.async_samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
#
# job_status = 'created'
# while job_status not in ['completed', 'error', 'canceled']:
#     print(f'job status : {job_status}')
#     time.sleep(2)
#     job_status = job.get_status()
#
# results = job.get_results()

results = s_cnot.samples(pcvl.BasicState([0, 1, 0, 1, 0, 0]), 1000)
for s in results:
    print(str(s))
print(f'Result of sample : {results}')
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
