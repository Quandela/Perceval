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

from perceval import NoiseModel, BasicState, Processor, Experiment, BS, Simulator, \
    SLOSBackend, Circuit, Source, FSIDistribution, FSCDistribution, Matrix, Unitary
from perceval.utils.statevector import FockStateIndex, FockStateCode, FockStateCodeInv, SVDistribution, StateVector, \
    FSCIDistribution

from perceval.utils import BasicState, post_select_distribution


def generate_distribution(source, input_state):
    # result = source.generate_state_distribution(input_state, None)
    result = source.generate_distribution(input_state)
    return result


import time

memory_mo = dict()
time_ms = dict()

FSD_MAP = {
    FockStateIndex : FSIDistribution,
    FockStateCode : FSCDistribution,
    FockStateCodeInv : FSCIDistribution
}

# for n in range(4,13):
for n in [9]:
    memory_mo[n] = dict()
    time_ms[n] = dict()

    # second test : at the end of simulator
    simulator = Simulator(SLOSBackend())
    circuit = Unitary(Matrix.random_unitary(n))

    simulator.set_circuit(circuit)

    state = [1 for k in range(n)]
    input_state = BasicState(state)
    source = Source(0.9, 0.03, 0.6)
    svd = generate_distribution(source, input_state)

    print('beginning loop over types of FS')
    for fs_type in [FockStateIndex, FockStateCode, FockStateCodeInv]:
        source.FS_TYPE = fs_type
        source.FSD_TYPE = FSD_MAP[fs_type]
        simulator.FS_TYPE = fs_type
        simulator.FSD_TYPE = FSD_MAP[fs_type]
        use_mem_maps = [False]

        time_ms[n][fs_type] = dict()
        if fs_type == FockStateIndex:
            # use_mem_maps = [True]
            use_mem_maps = [True,False]

        for mem_maps in use_mem_maps:
            FockStateIndex.use_memory_maps(mem_maps)

            # first test : source generation
            # source = NoiseModel(.9, 1, 0.03)
            # p = Processor("SLOS", Experiment(BS(), noise=source))
            #
            #
            # start = time.perf_counter()
            # result = generate_distribution(p.source, input_state)
            # stop = time.perf_counter()

            # second test : at the end of simulator
            start = time.perf_counter()
            res = simulator.probs_svd(svd)
            # input_state = BasicState("|{_:0},{_:0},{_:0},{_:0},{_:0},{_:0},{_:1},{_:2},{_:3},{_:4},{_:5},{_:6}>")
            # res = simulator.probs(input_state)
            stop = time.perf_counter()

            # input("Stop to check memory...")

            time_ms[n][fs_type][mem_maps] = (stop - start) * 1000
            print('time duration:', time_ms[n][fs_type][mem_maps])
            # print('results:', res)


with open('results_benchmark_fsi.csv', 'w') as f:
    f.write('n;FockStateIndex (with mem maps);FockStateIndex (without mem maps);FockStateCode;FockStateCodeInv\n')
    for n in time_ms:
        line = f'{n};'
        for fs_type in time_ms[n]:
            for mem_maps in time_ms[n][fs_type]:
                line += f'{time_ms[n][fs_type][mem_maps]};'
        f.write(line + '\n')
