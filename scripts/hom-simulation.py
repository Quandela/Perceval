import perceval as pcvl
import perceval.lib.symb as symb
import quandelibc as qc
import numpy as np


dt = pcvl.Parameter("Δt")

c = pcvl.Circuit(2)
c //= symb.BS()
c //= ((1), symb.DT(dt))
c //= symb.BS()

pcvl.pdisplay(c)

def photon_length_fn(t):
    length = 0.2e-9
    h = 2/length
    if t > length:
        return (length, 1, 0)
    return (length, 1-(length-t)*h*(length-t)/2/length, 0)

st0 = pcvl.AnnotatedBasicState([1, 0], time=2e-9, time_gen_fn=photon_length_fn)
backend = pcvl.BackendFactory().get_backend("Stepper")

sim = backend(c)

def f(x):
    dt.set_value(x)
    return sim.prob(st0, qc.FockState([2, 0]))+sim.prob(st0, qc.FockState([0, 2]))

for i in range(100):
    x = i * 2e-9/50
    print("f(%g)=%g" % (x, f(x)))
