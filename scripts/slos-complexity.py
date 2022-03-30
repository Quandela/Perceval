import quandelibc as qc
import numpy as np

for m in range(6, 25):
    for n in range(1, 13):
        fsa = qc.FSArray(m, n)

        allop = 0
        count = 0
        worst = 0
        best = None
        for fs in fsa:
            nop = np.prod([s+1 for s in fs if s])
            if nop > worst:
                worst = nop
            if best is None or nop < best:
                best = nop
            allop += nop
            count += 1

        print("m=", m, "n=", n, "Mn=", count, "best=", best, "worst=", worst, "avg=", allop/count, "ref=", n*2**n)
