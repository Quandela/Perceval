import perceval as pcvl
import numpy as np
import matplotlib.pyplot as plt
import perceval.lib.symb as symb

N = 100
ind = np.arange(0, 1, 1/N)
o = np.zeros((100,))
simulator_backend = pcvl.BackendFactory().get_backend('Naive')

for i in range(N):
    source = pcvl.Source(brightness=1, purity=1, indistinguishability=ind[i])
    qpu = pcvl.Processor({0: source, 1: source}, symb.BS())
    all_p, sv_out = qpu.run(simulator_backend)
    o[i] = sv_out[pcvl.StateVector("|1,1>")]

plt.plot(ind, o)
plt.ylabel("$p(|1,1>)$")
plt.xlabel("indistinguishability")
source = pcvl.Source(brightness=1, purity=1, indistinguishability=0.5)
qpu = pcvl.Processor({0: source, 1: source}, symb.BS())

all_p, sv_out = qpu.run(simulator_backend)
print("INPUT\n", qpu.source_distribution.pdisplay())
print("OUTPUT\n", sv_out.pdisplay())

plt.show()
