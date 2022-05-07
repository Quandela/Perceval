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

"""
@Author: Mathias Pont (mathias.pont@c2n.upsaclay.fr)
@Affiliation:
Centre for Nanosciences and Nanotechnology, CNRS, Universite Paris-Saclay, UMR 9001,
10 Boulevard Thomas Gobert, 
91120 Palaiseau, France

Computes the output state probabilities of a Mach-Zehnder interferometer

"""

import numpy as np
import matplotlib.pyplot as plt

import perceval as pcvl
import perceval.lib.symb as symb
from tqdm import tqdm


def outputstate_to_2outcome(state_ket):
    state = []
    for m in state_ket:
        if m.isdigit():
            state.append(m)

    if int(state[0]) == 0 and int(state[1]) == 0:
        return None
    if int(state[0]) == 0 and int(state[1]) > 0:
        return '|0,1>'
    if int(state[0]) > 0 and int(state[1]) == 0:
        return '|1,0>'
    if int(state[0]) > 0 and int(state[1]) > 0:
        return '|1,1>'



def mzi_BasicState(input_state):
    """
    :param input_state: ['|1,1>',]
    :return:
    """
    # Set up Perceval
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="MZI")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, symb.PS(phases[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phases[1]))
     .add((0, 1), symb.BS())

     )

    # Initial phase set to zero
    phases[0].set_value(0)

    scan_range = np.arange(0, np.pi, 0.1)

    outcome_theta = {}
    outcome = []

    for theta in scan_range:

        output_prob = dict()

        phases[1].set_value(theta)
        sim = simulator_backend(mzi_chip.U)

        for output_state, probability in sim.allstateprob_iterator(pcvl.BasicState(input_state)):

            output_state = str(output_state)

            if output_state not in outcome:
                outcome.append(output_state)

            if output_state in output_prob:
                output_prob[output_state] += probability
            else:
                output_prob[output_state] = probability

        outcome_theta[theta] = output_prob

    fig, ax = plt.subplots()
    for measured_state in outcome:
        ax.plot(outcome_theta.keys(),
                 [outcome_theta[angle][measured_state] for angle in outcome_theta.keys()],
                 '-s',
                 label=measured_state,
                 alpha=0.9)
    ax.set_title('Output state distribution', fontsize=24)
    ax.set_ylabel('Probability', fontsize=24)
    ax.set_xlabel('Internal phase of the MZI [rad]', fontsize=24)
    ax.legend(fontsize=18)
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)

    return outcome_theta



def mzi_AnySource(input_states_dict):
    """
    :param input_states_dict: ['|1,1>',]
    :return:
    """
    # Set up Perceval
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="mzi")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, symb.PS(phases[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phases[1]))
     .add((0, 1), symb.BS())

     )

    # Initial phase set to zero
    phases[0].set_value(0)


    scan_range = np.arange(0, np.pi, 0.1)

    to_plot = [[], [], []]
    label_outcomes = ['|1,0>', '|1,1>', '|0,1>']

    for theta in scan_range:

        output = {'|1,0>': 0,
                  '|1,1>': 0,
                  '|0,1>': 0
                  }

        phases[1].set_value(theta)
        sim = simulator_backend(mzi_chip.U)

        for input_n in input_states_dict:
            for output_state, probability in sim.allstateprob_iterator(pcvl.BasicState(input_n)):
                result = outputstate_to_2outcome(str(output_state))
                output[result] += input_states_dict[input_n] * probability

        to_plot[0].append(output['|1,0>'])
        to_plot[1].append(output['|1,1>'])
        to_plot[2].append(output['|0,1>'])

    plt.figure()
    for i, out in enumerate(to_plot):
        plt.plot(scan_range, out, '-o', label=label_outcomes[i])
    plt.xlabel('Phase [rad]', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.grid()
    plt.legend()

    # Compute this ouput distribution for np.pi/2

    output = {'|1,0>': 0,
              '|1,1>': 0,
              '|0,1>': 0
              }

    phases[1].set_value(np.pi / 2)
    sim = simulator_backend(mzi_chip.U)

    for input_n in input_states_dict:

        for output_state, probability in sim.allstateprob_iterator(pcvl.BasicState(input_n)):
            result = outputstate_to_2outcome(str(output_state))
            output[result] += input_states_dict[input_n] * probability

    v = 1 - 2 * output['|1,1>']

    print(v)

    return v


def mzi_ImperfectSource(beta, g2, M, plotit=False):
    """
    Here we suppose that we only have access to click detectors. All the outputs of the chip will be either a single
    click on one of the detector or a double click. We compute the probability of each of these outcomes.
    output -> what come out of the chip (|2,2>, |2,1>, |3,1>, ...)
    outcome -> what we can measure with our SNSPD detectors
    :param beta: source brightness
    :param g2: 1-single-photon purity
    :param M: Indistinguishability
    :return: plot and visibility of the HOM
    """

    # Set up Perceval
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="mzi")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, symb.PS(phases[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phases[1]))
     .add((0, 1), symb.BS())

     )

    # Initial phase set to zero
    phases[0].set_value(0)
    # Internal phase set to pi/2
    phases[1].set_value(np.pi / 2)

    # Find out all the unput states that must be considered depending on the characteristics of the source
    source = pcvl.Source(brightness=beta,
                         purity=1 - g2,
                         indistinguishability=M,
                         purity_model="random",  # "random", or "indistinguishable"
                         indistinguishability_model="homv")  # "homv", or "linear"
    # indistinguishability_model:
    # `homv` defines indistinguishability as HOM visibility,
    # `linear` defines indistinguishability as ratio of indistinguishable photons

    p = pcvl.Processor({0: source,
                        1: source,
                        },
                       mzi_chip)

    input_states_dict = {str(k): v for k, v in p.source_distribution.items()}

    del p, source

    if plotit:
        # Scan phi2 over the range scan_range to show the HOM dip.
        scan_range = np.arange(0, np.pi, 0.1)
        to_plot = [[], [], []]
        label = ['|1,0>', '|1,1>', '|0,1>']

        for theta in scan_range:

            # All outcome are initialize to 0 probability
            outcome = {'|1,0>': 0,
                       '|1,1>': 0,
                       '|0,1>': 0
                       }

            phases[1].set_value(theta)
            sim = simulator_backend(mzi_chip.U)

            for input_n in input_states_dict:

                for output_state, probability in sim.allstateprob_iterator(pcvl.AnnotatedBasicState(input_n)):
                    # Each output is mapped to an outcome
                    result = outputstate_to_2outcome(str(output_state))
                    # The probability of an outcome is added, weighted by the probability of this input
                    if result:
                        outcome[result] += input_states_dict[input_n] * probability

            to_plot[0].append(outcome['|1,0>'])
            to_plot[1].append(outcome['|1,1>'])
            to_plot[2].append(outcome['|0,1>'])

        plt.figure()
        for idx2, out in enumerate(to_plot):
            plt.plot(scan_range, out, '-o', label=label[idx2])
        plt.xlabel('Phase [rad]', fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.grid()
        plt.legend()

    # Compute the outcome distribution specifically for ph2=pi/2

    outcome = {'|1,0>': 0,
               '|1,1>': 0,
               '|0,1>': 0
               }

    for input_n in input_states_dict:
        sim = simulator_backend(mzi_chip.U)

        for output_state, probability in sim.allstateprob_iterator(pcvl.AnnotatedBasicState(input_n)):
            result = outputstate_to_2outcome(str(output_state))
            if result:
                outcome[result] += input_states_dict[input_n] * probability

    visibility = 1 - 2 * outcome['|1,1>']

    return visibility


def mzi_ImperfectSource_qpu(beta, g2, M, plotit=False):
    """
    Using the processor of Perceval. Here we suppose that we only have access to click detectors. All the outputs of the chip will be either a single
    click on one of the detector or a double click. We compute the probability of each of these outcomes.
    output -> what come out of the chip (|2,2>, |2,1>, |3,1>, ...)
    outcome -> what we can measure with our SNSPD detectors
    :param beta: source brightness
    :param g2: 1-single-photon purity
    :param M: Indistinguishability
    :return: plot and visibility of the HOM
    """

    # Set up Perceval
    simulator_backend = pcvl.BackendFactory().get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="mzi")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, symb.PS(phases[0]))
     .add((0, 1), symb.BS())
     .add(0, symb.PS(phases[1]))
     .add((0, 1), symb.BS())

     )

    # Initial phase set to zero
    phases[0].set_value(0)
    # Internal phase set to pi/2
    phases[1].set_value(np.pi / 2)

    # Find out all the unput states that must be considered depending on the characteristics of the source
    source = pcvl.Source(brightness=beta,
                         purity=1 - g2,
                         indistinguishability=M,
                         purity_model="random",  # "random", or "indistinguishable"
                         indistinguishability_model="homv")  # "homv", or "linear"
    # indistinguishability_model:
    # `homv` defines indistinguishability as HOM visibility,
    # `linear` defines indistinguishability as ratio of indistinguishable photons


    if plotit:
        # Scan phi2 over the range scan_range to show the HOM dip.
        scan_range = np.arange(0, np.pi, 0.1)
        to_plot = [[], [], []]
        label = ['|1,0>', '|1,1>', '|0,1>']

        for theta in scan_range:

            # All outcome are initialize to 0 probability
            outcome = {'|1,0>': 0,
                       '|1,1>': 0,
                       '|0,1>': 0
                       }

            phases[1].set_value(theta)

            qpu = pcvl.Processor({0: source,
                                  1: source,
                                  },
                                 mzi_chip)

            all_p, sv_out = qpu.run(simulator_backend)

            for output_state in sv_out:
                # Each output is mapped to an outcome
                result = outputstate_to_2outcome(str(output_state))
                # The probability of an outcome is added, weighted by the probability of this input
                if result:
                    outcome[result] +=  sv_out[output_state]

            to_plot[0].append(outcome['|1,0>'])
            to_plot[1].append(outcome['|1,1>'])
            to_plot[2].append(outcome['|0,1>'])

        plt.figure()
        for idx2, out in enumerate(to_plot):
            plt.plot(scan_range, out, '-o', label=label[idx2])
        plt.xlabel('Phase [rad]', fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.grid()
        plt.legend()

    # Compute the outcome distribution specifically for ph2=pi/2

    outcome = {'|1,0>': 0,
               '|1,1>': 0,
               '|0,1>': 0
               }

    phases[1].set_value(np.pi / 2)

    qpu = pcvl.Processor({0: source,
                        1: source,
                        },
                       mzi_chip)

    all_p, sv_out = qpu.run(simulator_backend)


    for output_state in sv_out:
        # Each output is mapped to an outcome
        result = outputstate_to_2outcome(str(output_state))
        # The probability of an outcome is added, weighted by the probability of this input
        if result:
            outcome[result] += sv_out[output_state]

    visibility = 1 - 2 * outcome['|1,1>']

    return visibility


def compare_simu_mzi(Nb):
    
    X = np.array(np.linspace(0.0, 0.1, Nb))
    Y = np.array(np.linspace(0.8, 1.0, Nb))

    Z = np.array([[mzi_ImperfectSource_qpu(1, g2, M) for g2 in X] for M in tqdm(Y)])

    Z_th = np.array([[M * (1 - g2) - g2 for g2 in X] for M in Y])

    fig, ax = plt.subplots()
    cf = ax.pcolormesh(X, Y, Z, vmin=0.5, vmax=1.01)
    ax.set_xlabel(r'$g^{(2)}(0)$', fontsize=24)
    ax.set_ylabel('Indistinguishability', fontsize=24)
    ax.set_title('Simulation', fontsize=24)
    fig.colorbar(cf, ticks=np.arange(0, 1.01, 0.1), label='Visibility')
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)

    fig1, ax1 = plt.subplots()
    cf1 = ax1.pcolormesh(X, Y, Z_th, vmin=0.5, vmax=1.01)
    ax1.set_xlabel(r'$g^{(2)}(0)$', fontsize=24)
    ax1.set_title('Model', fontsize=24)
    ax1.set_ylabel('Indistinguishability', fontsize=24)
    fig1.colorbar(cf1, ticks=np.arange(0, 1.01, 0.1), label='Visibility')
    ax1.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)

    fig2, ax2 = plt.subplots()
    cf2 = ax2.pcolormesh(X, Y, (Z - Z_th) * 100)
    ax2.set_title('Simulation - Model', fontsize=24)
    ax2.set_xlabel(r'$g^{(2)}(0)$', fontsize=24)
    ax2.set_ylabel('Indistinguishability', fontsize=24)
    fig2.colorbar(cf2, label=r'$V_{simu}-V_{model}$ [%]')
    ax2.tick_params(direction='in', bottom=True, top=True, left=True, right=True, labelsize=16)


if __name__ == '__main__':
    mzi_BasicState('|1,1>')

    mzi_AnySource({'|1,2>': 0.8, '|2,1>': 0.2})

    mzi_ImperfectSource_qpu(beta=1, g2=0, M=0.9)
