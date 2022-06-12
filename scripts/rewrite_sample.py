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
from perceval.algorithm.optimize import optimize
from perceval.algorithm import frobenius
import time
import random

pattern1=pcvl.Circuit(3, name="pattern1")//(0,phys.PS(pcvl.P("phi0")))//(0,phys.BS(theta=pcvl.P("theta1")))//(0,phys.PS(pcvl.P("phi2")))//(1,phys.PS(pcvl.P("phi1")))//(1,phys.BS(theta=pcvl.P("theta2")))//(0,phys.BS(theta=pcvl.P("theta3")))
pattern1._color = "lightgreen"
pcvl.pdisplay(pcvl.Circuit(3).add(0,pattern1,False), recursive=True)

rewrite1=pcvl.Circuit(3, name="rewrite1")//(0,phys.PS(pcvl.P("beta2")))//(1,phys.PS(pcvl.P("beta1")))//(1,phys.BS(theta=pcvl.P("alpha1")))//(0,phys.BS(theta=pcvl.P("alpha2")))//(1,phys.PS(pcvl.P("beta3")))//(1,phys.BS(theta=pcvl.P("alpha3")))//(0,phys.PS(pcvl.P("beta4")))//(1,phys.PS(pcvl.P("beta5")))//(2,phys.PS(pcvl.P("beta6")))
rewrite1._color = "pink"
pcvl.pdisplay(pcvl.Circuit(3).add(0,rewrite1,False), recursive=True)

pattern2=pcvl.Circuit(1, name="pattern2")//phys.PS(pcvl.P("phi1"))//phys.PS(pcvl.P("phi2"))
rewrite2=pcvl.Circuit(1, name="rewrite2")//phys.PS(pcvl.P("phi"))

a=pcvl.Circuit.generic_interferometer(8,
                                      lambda idx:pcvl.Circuit(2)//phys.PS(phi=random.random())//phys.BS(theta=random.random()),
                                      shape="rectangle")
#pcvl.pdisplay(a, recursive=True, render_size=1)

current = time.time()
def tick(description):
    global current
    dt = time.time()-current
    print("%f\t%s" % (dt, description))
    current = time.time()

rules = [(pattern1, rewrite1, "lightgreen"), (pattern2, rewrite2, "pink")]
while True:
    found = False
    for pattern, rewrite, color in rules:
        start_pos = 0
        while True:
            matched = a.match(pattern, browse=True, pos=start_pos)
            tick("match: %s - %s" % (pattern._name, matched and "ok" or "nok"))
            if matched is None:
                break
            idx = a.isolate(list(matched.pos_map.keys()), color=color)
            tick("└─ isolate match")
            for k, v in matched.v_map.items():
                pattern[k].set_value(v)
            tick("└─ set values")
            v = pattern.compute_unitary(False)
            tick("└─ compute unitary")
            res = optimize(rewrite, v, frobenius, sign=-1)
            tick("└─ optimize %s - %f" % (rewrite._name, res.fun))
            subc = rewrite.copy()
            tick("└─ rewrite")
            found = True
            a.replace(idx, subc, merge=True)
            tick("└─ replace")
            pattern.reset_parameters()
            rewrite.reset_parameters()
            tick("└─ reset parameters")
            start_pos = idx
    if not found:
        break
