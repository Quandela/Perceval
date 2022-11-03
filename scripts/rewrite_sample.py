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
from perceval.components.unitary_components import BS, PS
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import fidelity, frobenius
import time
import random

pattern1 = pcvl.Circuit(2, name="pattern1") // (0, PS(pcvl.P("phi1"))) // (0, BS.H()) //\
           (0, PS(pcvl.P("phi2"))) // (0, BS.H())
rewrite1 = pcvl.Circuit(2, name="rewrite1") // (0, PS(pcvl.P("phi1"))) // (0, BS.H(theta=BS.r_to_theta(0.42))) //\
           (0, PS(pcvl.P("phi2"))) // (0, BS.H(theta=BS.r_to_theta(0.42))) // (0, PS(pcvl.P("phi3"))) //\
           (1, PS(pcvl.P("phi4")))

pattern2 = pcvl.Circuit(1, name="pattern2") // PS(pcvl.P("phi1")) // PS(pcvl.P("phi2"))
rewrite2 = pcvl.Circuit(1, name="rewrite2") // PS(pcvl.P("phi"))

pattern3 = pcvl.Circuit(2, name="pattern3") // (1, PS(pcvl.P("phip"))) // (0, BS.H(theta=BS.r_to_theta(0.42)))
rewrite3 = pcvl.Circuit(2, name="rewrite3") // (0, PS(pcvl.P("phi1"))) // (0, BS.H(theta=BS.r_to_theta(0.42))) //\
           (0, PS(pcvl.P("phi2"))) // (1, PS(pcvl.P("phi3")))

a = pcvl.Circuit.generic_interferometer(8,
                                        lambda idx: pcvl.Circuit(2) // PS(phi=random.random()) // BS.H()
                                                    // PS(phi=random.random()) // BS.H(),
                                        shape="rectangle")
u = a.compute_unitary(use_symbolic=False)

current = time.time()


def tick(description):
    global current
    dt = time.time()-current
    print("%f\t%s" % (dt, description))
    current = time.time()


rules = [(pattern1, rewrite1, "lightgreen"), (pattern2, rewrite2, "pink"), (pattern3, rewrite3, "lightgray")]

while True:
    found_match = False
    for pattern, rewrite, color in rules:
        start_pos = 0
        matched = a.match(pattern, browse=True, pos=start_pos)
        if matched is None:
            break
        found_match = True
        idx = a.isolate(list(matched.pos_map.keys()), color=color)
        for k, v in matched.v_map.items():
            pattern.param(k).set_value(v)
        v = pattern.compute_unitary(False)
        res = optimize(rewrite, v, frobenius, sign=-1)
        subc = rewrite.copy()
        found = True
        a.replace(idx, subc, merge=False)
        a.replace(idx, subc, merge=True)
        pattern.reset_parameters()
        rewrite.reset_parameters()
        start_pos = idx
        print(pattern.name, res.fun, fidelity(u, a.compute_unitary(False)))
    if not found_match:
        break
