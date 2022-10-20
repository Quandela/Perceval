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
import perceval as pcvl
import perceval.components.unitary_components as cp
from perceval.components import catalog
import numpy as np
from perceval.algorithm import Sampler, Analyzer
from perceval.components import Processor, Source


# Clifford & Clifford 2017 backend does not support probability computation
# However, the Sampler algorithm is able to estimate output probabilities, transparently, through sampling
local_simulator_name = 'CliffordClifford2017'
gate_name = "postprocessed cnot"
cnot_processor = catalog[gate_name].as_processor(local_simulator_name).build()
assert not cnot_processor.is_remote
cnot_processor.source = Source(brightness=0.5)
cnot_processor.set_postprocess(None)  # Remove post-processing function to retrieve more states
# You may use a logical state as an input, converted into the fock state |1,0,1,0> thanks to dual rail encoding ports
# defined in cnot_processor
cnot_processor.with_input(pcvl.LogicalState([0, 0]))

sampler = Sampler(cnot_processor)
output = sampler.probs()
pcvl.pdisplay(output['results'])
print(f"Physical performance: {output['physical_perf']}")
print(f"Performance on post process / heralding: {output['logical_perf']}")

# Let's sample asynchronously
nsample = 100000
cnot_processor.with_input(pcvl.BasicState([0, 1, 0, 1]))  # You can change the input of your processor anytime
async_job = sampler.samples.execute_async(nsample)

previous_prog = 0
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    tq.set_description(f'Get {nsample} samples from {gate_name} using simulator {local_simulator_name}')
    while not async_job.is_completed():
        tq.update(async_job.status.progress-previous_prog)
        previous_prog = async_job.status.progress
        time.sleep(.2)
    tq.update(1-previous_prog)
    tq.close()

print(f"Job status = {async_job.status()}")
output = async_job.get_results()
assert len(output['results']) == nsample

# Now, try an async sample_count with SLOS backend
local_simulator_name = 'SLOS'
cnot_circuit = catalog["postprocessed cnot"].as_circuit().build()
proc_slos = Processor(local_simulator_name, 6, pcvl.Source(brightness=0.9))
proc_slos.add(0, cnot_circuit)
proc_slos.with_input(pcvl.BasicState([1, 0, 1, 0, 1, 0]))

slos_sampler = Sampler(proc_slos)
job2 = slos_sampler.sample_count.execute_async(count=100000)

previous_prog = 0
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    tq.set_description(f'Count samples from {cnot_circuit.name} using simulator {local_simulator_name}')
    while not job2.is_completed():
        tq.update(job2.status.progress-previous_prog)
        previous_prog = job2.status.progress
        time.sleep(.2)
    tq.update(1-previous_prog)
    tq.close()

print(f"Job status = {job2.status()}")
output = job2.get_results()
for state, count in output['results'].items():
    print(f"{state}: {count}")
print(f"Physical performance: {output['physical_perf']}")


chip_QRNG = pcvl.Circuit(4, name='QRNG')
# Parameters
phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
        pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
# Defining the LO elements of chip
(chip_QRNG
 .add((0, 1), cp.BS())
 .add((2, 3), cp.BS())
 .add((1, 2), cp.PERM([1, 0]))
 .add(0, cp.PS(phis[0]))
 .add(2, cp.PS(phis[2]))
 .add((0, 1), cp.BS())
 .add((2, 3), cp.BS())
 .add(0, cp.PS(phis[1]))
 .add(2, cp.PS(phis[3]))
 .add((0, 1), cp.BS())
 .add((2, 3), cp.BS())
 )
# Setting parameters value and see how chip specs evolve
phis[0].set_value(np.pi / 2)
phis[1].set_value(0.2)
phis[2].set_value(0)
phis[3].set_value(0.4)

print("Use analyzer algorithm")
qrng_processor = Processor("SLOS", chip_QRNG, Source(brightness=0.8))

analyzer = Analyzer(qrng_processor, {pcvl.BasicState([1, 0, 1, 0]): '00', pcvl.BasicState([0, 1, 1, 0]): '10'}, '*')
with tqdm(total=1, bar_format='{desc}{percentage:3.0f}%|{bar}|') as tq:
    def update_progress(p):
        tq.update(p-update_progress.prev)
        update_progress.prev = p

    tq.set_description("Analyzing QRNG circuit with SLOS")
    update_progress.prev = 0
    res = analyzer.compute(progress_callback=update_progress, normalize=True)
    tq.close()

pcvl.pdisplay(analyzer)
print(f"Performance = {analyzer.performance}")
