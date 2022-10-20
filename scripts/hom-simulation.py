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
import perceval.components.unitary_components as comp
import quandelibc as qc


dt = pcvl.Parameter("Δt")

c = pcvl.Circuit(2)
c //= comp.BS()
c //= (1, comp.TD(dt))
c //= comp.BS()

pcvl.pdisplay(c)


def photon_length_fn(t):
    length = 0.2e-9
    h = 2 / length
    if t > length:
        return length, 1, 0
    return length, 1-(length-t)*h*(length-t)/2/length, 0


st0 = pcvl.BasicState([1, 0])
backend = pcvl.BackendFactory().get_backend("Stepper")

sim = backend(c)


def f(x):
    dt.set_value(x)
    return sim.prob(st0, qc.FockState([2, 0]))+sim.prob(st0, qc.FockState([0, 2]))


for i in range(100):
    x = i * 2e-9/50
    print("f(%g)=%g" % (x, f(x)))
