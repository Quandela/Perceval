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

from perceval.components import Circuit, PredefinedCircuit
from perceval.components.base_components import *


R1 = 0.228
R2 = 0.758
theta1 = BS.r_to_theta(R1)
theta2 = BS.r_to_theta(R2)


c_hcnot = (Circuit(8, name="Heralded CNOT")
           .add((0, 1, 2), PERM([1, 2, 0]))
           .add((4, 5), BS.H())
           .add((5, 6, 7), PERM([1, 2, 0]))
           .add((3, 4), BS.H())
           .add((2, 3), BS.H(theta=theta1, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
           .add((4, 5), BS.H(theta=theta1))
           .add((3, 4), BS.H())
           .add((5, 6, 7), PERM([2, 1, 0]))
           .add((1, 2), PERM([1, 0]))
           .add((2, 3), BS.H(theta=theta2))
           .add((4, 5), BS.H(theta=theta2, phi_bl=np.pi, phi_tr=np.pi/2, phi_tl=-np.pi/2))
           .add((5, 6), PERM([1, 0]))
           .add((4, 5), BS.H())
           .add((4, 5), PERM([1, 0]))
           .add((0, 1, 2), PERM([2, 1, 0])))


heralded_cnot = PredefinedCircuit(c_hcnot,
                                  "heralded cnot",
                                  description="https://doi.org/10.1073/pnas.1018839108",
                                  heralds={0: 0, 1: 1, 6: 0, 7: 1})
