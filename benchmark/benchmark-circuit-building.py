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
from perceval.components.unitary_components import BS, PS, PERM
import random
import time

t_build = 0
t_set = 0
t_compute = 0
t_get = 0

for _ in range(1000):
    start = time.time()
    #List of the parameters φ1,φ2,...,φ8
    List_Parameters = []

    # VQE is a 6 optical mode circuit
    VQE = pcvl.Circuit(6)

    VQE.add((1, 2), BS.H())
    VQE.add((3, 4), BS.H())
    List_Parameters.append(pcvl.Parameter("φ1"))
    VQE.add((2,), PS(phi=List_Parameters[-1]))
    List_Parameters.append(pcvl.Parameter("φ3"))
    VQE.add((4,), PS(phi=List_Parameters[-1]))
    VQE.add((1, 2), BS.H())
    VQE.add((3, 4), BS.H())
    List_Parameters.append(pcvl.Parameter("φ2"))
    VQE.add((2,), PS(phi=List_Parameters[-1]))
    List_Parameters.append(pcvl.Parameter("φ4"))
    VQE.add((4,), PS(phi=List_Parameters[-1]))


    # CNOT ( Post-selected with a success probability of 1/9)
    VQE.add([0,1,2,3,4,5], PERM([0,1,2,3,4,5]))#Identity PERM (permutation) for the purpose of drawing a nice circuit
    VQE.add((3, 4), BS.H())
    VQE.add([0,1,2,3,4,5], PERM([0,1,2,3,4,5]))#Identity PERM (permutation) for the same purpose
    VQE.add((0, 1), BS.H(theta=BS.r_to_theta(1/3)))
    VQE.add((2, 3), BS.H(theta=BS.r_to_theta(1/3)))
    VQE.add((4, 5), BS.H(theta=BS.r_to_theta(1/3)))
    VQE.add([0,1,2,3,4,5], PERM([0,1,2,3,4,5]))#Identity PERM (permutation) for the same purpose
    VQE.add((3, 4), BS.H())
    VQE.add([0,1,2,3,4,5], PERM([0,1,2,3,4,5]))#Identity PERM (permutation) for the same purpose

    List_Parameters.append(pcvl.Parameter("φ5"))
    VQE.add((2,), PS(phi=List_Parameters[-1]))
    List_Parameters.append(pcvl.Parameter("φ7"))
    VQE.add((4,), PS(phi=List_Parameters[-1]))
    VQE.add((1, 2), BS.H())
    VQE.add((3, 4), BS.H())
    List_Parameters.append(pcvl.Parameter("φ6"))
    VQE.add((2,), PS(phi=List_Parameters[-1]))
    List_Parameters.append(pcvl.Parameter("φ8"))
    VQE.add((4,), PS(phi=List_Parameters[-1]))
    VQE.add((1, 2), BS.H())
    VQE.add((3, 4), BS.H())

    t_build += time.time()-start
    start = time.time()
    init_param = [random.random() for _ in List_Parameters]

    for idx, p in enumerate(List_Parameters):
        p.set_value(init_param[idx])

    t_set += time.time()-start
    start = time.time()

    VQE.compute_unitary(use_symbolic = False)

    t_compute += time.time()-start
    start = time.time()

    for i in range(len(List_Parameters)):
        init_param[i] = VQE.get_parameters()[i]._value

    t_get += time.time()-start

print("TOTAL=", t_build+t_set+t_compute+t_get, "DETAIL=", t_build, t_set, t_compute, t_get)
