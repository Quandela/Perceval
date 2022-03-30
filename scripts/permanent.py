import perceval as pcvl
import quandelibc as qc
import numpy as np
import time

m = 16
n = 12

u = pcvl.Matrix.random_unitary(m)

fsms = [[]]
fsas = [qc.FSArray(m, 0)]
coefs = [[1]]

for i in range(1, m+1):
    fsas.append(qc.FSArray(m, i))
    coefs.append(np.zeros(fsas[-1].count(), dtype=complex))
    fsms.append(qc.FSiMap(fsas[-1], fsas[-2], True))

compute = 0
def permanent(idx_current, k):
    global compute
    if k == 0:
        return 1
    if not coefs[k][idx_current]:
        m = 0
        while m < k:
            index_parent, mode = fsms[k].get(idx_current, m)
            if index_parent == qc.npos:
                break
            compute += 1
            coefs[k][idx_current] += permanent(index_parent, k-1)*u[k-1, mode]
            m += 1
    return coefs[k][idx_current]

start_slos_1 = time.time()
for idx in range(fsas[-1].count()):
    permanent(idx, n)
end_slos_1 = time.time()
time_total_slos = end_slos_1-start_slos_1
print("slos", time_total_slos)

start_qc_1 = time.time()
for idx in range(fsas[-1].count()):
    qc.permanent_cx(u, 1)
end_qc_1 = time.time()

print("qc", end_qc_1-start_qc_1)
