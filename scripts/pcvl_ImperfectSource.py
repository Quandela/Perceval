# -*- coding: utf-8 -*-
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
r"""

@Author: Mathias Pont (mathias.pont@c2n.upsaclay.fr)
@Affiliation:
Centre for Nanosciences and Nanotechnology, CNRS, Universite Paris-Saclay, UMR 9001,
10 Boulevard Thomas Gobert,
91120 Palaiseau, France

Computes the output state probabilities of a Mach-Zehnder interferometer and compare to a theoretical model [1]

[1] https://doi.org/10.1103/PhysRevLett.126.063602

"""
import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.lib.symb as symb
import pytest


def outputstate_to_2outcome(output):
    """
    :param output: an output of the chip
    :return: a measurable outcome
    """
    state = []
    for m in output:
        if m.isdigit():
            state.append(m)

    if int(state[0]) == 0 and int(state[1]) == 0:
        return '|0,0>'
    if int(state[0]) == 0 and int(state[1]) > 0:
        return '|0,1>'
    if int(state[0]) > 0 and int(state[1]) == 0:
        return '|1,0>'
    if int(state[0]) > 0 and int(state[1]) > 0:
        return '|1,1>'


class QPU:

    def __init__(self):
        # Set up Perceval
        self.simulator_backend = pcvl.BackendFactory().get_backend('Naive')

        # Create a MZI interferometer
        self.mzi_chip = pcvl.Circuit(m=2, name="MZI")

        self.phase_shifters = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

        (self.mzi_chip
         .add(0, symb.PS(self.phase_shifters[0]))
         .add((0, 1), symb.BS())
         .add(0, symb.PS(self.phase_shifters[1]))
         .add((0, 1), symb.BS())
         )

        # Initial phase set to zero
        self.phase_shifters[0].set_value(0)
        # Internal phase set to pi/2
        self.phase_shifters[1].set_value(np.pi / 2)


def compute(qpu, beta, g2, M, multiphoton_model="distinguishable"):
    # Find out all the unput states that must be considered depending on the characteristics of the source
    sps = pcvl.Source(brightness=beta,
                      multiphoton_component=g2,
                      multiphoton_model=multiphoton_model,
                      indistinguishability=M,
                      indistinguishability_model="homv",
                      overall_transmission=1)  # "homv", or "linear"

    outcome = {'|0,0>': 0,
               '|1,0>': 0,
               '|1,1>': 0,
               '|0,1>': 0
               }

    p = pcvl.Processor({0: sps, 1: sps, }, qpu.mzi_chip)

    #input_states_dict = {str(k): v for k, v in p.source_distribution.items()}
    #print(input_states_dict)
    #print(sum(p.source_distribution.values()))
    #assert(pytest.approx(sum(p.source_distribution.values())) == 1)

    all_p, sv_out = p.run(qpu.simulator_backend)

    for output_state in sv_out:
        # print(str(output_state), str(sv_out[output_state]))
        # Each output is mapped to an outcome
        result = outputstate_to_2outcome(str(output_state))
        # The probability of an outcome is added, weighted by the probability of this input
        outcome[result] += sv_out[output_state]

    visibility = 1 - 2 * outcome['|1,1>'] / (outcome['|0,0>'] + outcome['|1,0>'] + outcome['|1,1>'] + outcome['|0,1>'])
    assert (pytest.approx(outcome['|0,0>'] + outcome['|1,0>'] + outcome['|1,1>'] + outcome['|0,1>']) == 1)

    return visibility


if __name__ == '__main__':
    qpunit = QPU()

    mu = 1
    Nb = 100
    X = np.array(np.linspace(0.0, 0.3, Nb))
    Y = np.array(np.linspace(0.5, 1, Nb))
    beta_axis = np.array(np.linspace(0.5, 1, Nb))

    # V_HOM vs g2
    z11 = [compute(qpunit, mu, g2, 1, multiphoton_model="indistinguishable") for g2 in X]
    z12 = [compute(qpunit, mu, g2, 1, multiphoton_model="distinguishable") for g2 in X]
    z1_th1 = [1 - g2 for g2 in X]
    z1_th2 = [1 - 2 * g2 for g2 in X]

    # Plot the result
    plt.figure()

    plt.plot(X, z11, 's', label='simulation indistinguishable', color='firebrick')
    plt.plot(X, z1_th1, label='model indistinguishable', color='firebrick')

    plt.plot(X, z12, 's', label='simulation distinguishable', color='seagreen')
    plt.plot(X, z1_th2, label='model distinguishable', color='seagreen')

    plt.grid()
    plt.legend(fontsize=20)
    plt.xlabel(r'$g^{(2)}(0)$', fontsize=30)
    plt.ylabel(r'$V_{HOM}$', fontsize=30)
    plt.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=28)

    plot_all = False
    if plot_all:
        # V_HOM vs M
        z2 = [compute(qpunit, mu, 0, m) for m in Y]
        z2_th = Y
        # V_HOM vs beta
        z3 = [compute(qpunit, beta, 0.016, 1) for beta in beta_axis]
        # V_HOM vs g2 & M
        Z = np.array([[compute(qpunit, mu, g2, M) for g2 in X] for M in Y])
        Z2 = np.array([[compute(qpunit, mu, g2, 1) for g2 in X] for mu in Y])

        Z_th = np.array([[M - (1 + M) * g2 for g2 in X] for M in Y])
        plt.figure()
        plt.plot(Y, z2, 's', label='simulation')
        plt.plot(Y, z2_th, label='model')
        plt.grid()
        plt.legend(fontsize=20)
        plt.xlabel('M', fontsize=20)
        plt.ylabel(r'$V_{HOM}$', fontsize=20)

        plt.figure()
        plt.plot(beta_axis, z3, 's', label='simulation')
        plt.grid()
        plt.legend(fontsize=20)
        plt.xlabel(r'$\beta$', fontsize=20)
        plt.ylabel(r'$V_{HOM}$', fontsize=20)

        fig, ax = plt.subplots()
        cf = ax.pcolormesh(X, Y, (Z - Z_th) * 100, shading='auto', cmap='GnBu')
        ax.set_title('Simulation - Model', fontsize=24)
        ax.set_xlabel(r'$g^{(2)}(0)$', fontsize=24)
        ax.set_ylabel('Indistinguishability', fontsize=24)
        fig.colorbar(cf, label=r'$V_{simu}-V_{model}$ [%]')
        ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)

        fig, ax = plt.subplots()
        cf = ax.pcolormesh(X, Y, Z2, shading='auto', cmap='GnBu')
        ax.set_title('Simulation', fontsize=24)
        ax.set_xlabel(r'$g^{(2)}(0)$', fontsize=24)
        ax.set_ylabel('Losses', fontsize=24)
        fig.colorbar(cf, label=r'$V_{simu}$ [%]')
        ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)
