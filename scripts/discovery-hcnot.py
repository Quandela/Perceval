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

import concurrent.futures
import logging
import random
import sys
import threading
import time

import numpy as np
from scipy.optimize import minimize

import perceval as pcvl
import perceval.components.unitary_components as comp
from perceval.rendering.pdisplay import pdisplay_analyzer


mapping = {
    "HH": pcvl.BasicState([1, 1, 0, 1, 1, 1, 0]),
    "HL": pcvl.BasicState([1, 1, 0, 1, 1, 0, 1]),
    "LH": pcvl.BasicState([1, 0, 1, 1, 1, 1, 0]),
    "LL": pcvl.BasicState([1, 0, 1, 1, 1, 0, 1])
}

istates = {v: k for k, v in mapping.items()}
simulator_backend = pcvl.BackendFactory().get_backend("Naive")


def f_to_minimize(circuit, params, params_value, prob=1):
    for idx, p in enumerate(params_value):
        params[idx].set_value(p)

    sim = simulator_backend(circuit.compute_unitary(use_symbolic=False), n=5, mask=["1  11  "])
    sim.compile(istates.keys())
    ca = pcvl.CircuitAnalyser(sim, istates, "*")
    ca.compute(expected={"LH": "LL", "LL": "LH", "HH": "HH", "HL": "HL"})
    loss = np.sqrt((0.1*(prob-ca.performance))**2+(0.9*ca.error_rate)**2)
    return loss


def discover(circuit, p, params=None, method=None, init_params=None, bounds=None):
    if init_params is None:
        init_params = np.random.randn(len(params))

    minimize(lambda x: f_to_minimize(circuit, params, x, p), init_params, method=method, bounds=bounds)

    sim = simulator_backend(circuit.compute_unitary(use_symbolic=False), n=5, mask=["1  11  "])
    ca = pcvl.CircuitAnalyser(sim, istates, "*")
    ca.compute(expected={"LH": "LL", "LL": "LH", "HH": "HH", "HL": "HL"})
    performance = ca.performance
    ber = ca.error_rate
    return ber, performance, pdisplay_analyzer(ca)


global_lock = threading.Lock()
result_file = sys.stdout  # open("result-discovery.txt", "w")


def run_a_discovery(name):
    logging.info("Discovery thread %s: starting", name)
    start = time.time()

    # generic interferometer
    n = 7
    gen_rect = pcvl.Circuit.generic_interferometer(
        n,
        lambda i: random.randint(0, 1)
                  and comp.BS.H(theta=pcvl.P("theta%d" % i), phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2)
                  or comp.BS.H(theta=pcvl.P("theta%d" % i)),
        shape="rectangular",
        depth=4
    )

    params = gen_rect.get_parameters()
    bounds = [p.bounds for p in params]

    (end_ber, end_performance, cadisplay) = discover(gen_rect,
                                                     0.5,
                                                     params=params,
                                                     init_params=[random.random() for _ in params],
                                                     bounds=bounds)
    with global_lock:
        result_file.write("time=%f, performance=%f, ber=%f\n" % (time.time()-start, end_performance, end_ber))
        result_file.write(cadisplay+"\n")
        result_file.write(str(gen_rect.get_parameters())+"\n\n")

    logging.info("Discovery thread %s: ending in %f seconds", name, time.time()-start)


log_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO,
                    datefmt="%H:%M:%S")

# max_works is the number of parallel workers that will be launched
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # launch N iterations
    executor.map(run_a_discovery, range(10))
