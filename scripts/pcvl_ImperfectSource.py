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
@Author: Mathias Pont
"""

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components.unitary_components as comp
import perceval.algorithm as algo


def outputstate_to_outcome(state_ket):
    """
    :param state_ket: an output of the chip
    :return: a measurable outcome
    """
    state_list = []
    for m in state_ket:
        if m.isdigit():
            state_list.append(m)

    state = tuple(state_list)

    if int(state[0]) == 0 and int(state[1]) == 0:
        return None
    if int(state[0]) == 0 and int(state[1]) > 0:
        return '|0,1>'
    if int(state[0]) > 0 and int(state[1]) == 0:
        return '|1,0>'
    if int(state[0]) > 0 and int(state[1]) > 0:
        return '|1,1>'


def mzi_BasicState_pcvl(input_state):
    """
    :param input_state: a ket in string. For example '|1,1>'
    :return: plot
    """
    # Set up Perceval
    naive_backend = pcvl.BackendFactory.get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="mzi")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, comp.PS(phases[0]))
     .add((0, 1), comp.BS())
     .add(0, comp.PS(phases[1]))
     .add((0, 1), comp.BS())
     )
    pcvl.pdisplay(mzi_chip)

    # Initial phase set to zero
    phases[0].set_value(0)

    # The phase of the MZI
    phases[1].set_value(np.pi)

    # We run the simulator once with any phase (here pi) to get all the possible outputs.
    ca = algo.Analyzer(naive_backend, mzi_chip,
                       [pcvl.BasicState(input_state)],
                       "*")
    ca.compute()
    nb_of_outputs = len(ca.output_states_list)

    scan_range = np.arange(0, np.pi, 0.1)

    # We initialise a list to store each output probability as a function of the phase phi2
    output = [[] for i in range(nb_of_outputs)]

    for theta in scan_range:

        # Set the phase of the MZI
        phases[1].set_value(theta)

        # Run the analyser
        ca = algo.Analyzer(naive_backend, mzi_chip,
                           [pcvl.BasicState(input_state)],
                           "*")

        ca.compute()

        # Append the result to the ouput to plot
        for i in range(len(output)):
            output[i].append(ca.distribution[0][i])

    plt.figure()
    for i, out in enumerate(output):
        plt.plot(scan_range, out, '-o', label=ca.output_states_list[i])
    plt.xlabel('Phase [rad]', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.grid()
    plt.legend()
    plt.show()


def mzi_ImperfectSource_pcvl(beta, g2, V):
    """
    Here we suppose that we only have access to click detectors. All the outputs of the chip will be either a single
    click on one of the detector or a double click. We compute the probability of each of these outcomes.
    output -> what come out of the chip (|2,2>, |2,1>, |3,1>, ...)
    outcome -> what we can measure with our SNSPD detectors

    :param beta: source brightness
    :param g2: 1-single-photon purity
    :param V: Indistinguishability
    :return: plot and visibility of the HOM
    """

    # Set up Perceval
    naive_backend = pcvl.BackendFactory.get_backend('Naive')

    # Create a MZI interferometer
    mzi_chip = pcvl.Circuit(m=2, name="mzi")

    phases = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2")]

    (mzi_chip
     .add(0, comp.PS(phases[0]))
     .add((0, 1), comp.BS())
     .add(0, comp.PS(phases[1]))
     .add((0, 1), comp.BS())
     )

    # Initial phase set to zero
    phases[0].set_value(0)

    # Find out all the unput states that must be considered depending on the characteristics of the source
    source = pcvl.Source(brightness=beta,
                         purity=1 - g2,
                         indistinguishability=V)

    p = pcvl.Processor({0: source,
                        1: source
                        },
                       mzi_chip)

    input_states_dict = {str(k): v for k, v in p.source_distribution.items()}

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

        for input_n in input_states_dict:

            phases[1].set_value(theta)
            ca = algo.Analyzer(naive_backend, mzi_chip,
                               [pcvl.BasicState(input_n)],
                               "*")
            ca.compute()

            for idx1, output_state in enumerate(ca.output_states_list):
                # Each output is mapped to an outcome
                result = outputstate_to_outcome(str(output_state))
                # The probability of an outcome is added, weighted by the probability of this input
                if result:
                    outcome[result] += input_states_dict[input_n] * ca.distribution[0][idx1]

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
        print(input_n, input_states_dict[input_n])
        phases[1].set_value(np.pi / 2)
        ca = algo.Analyzer(naive_backend, mzi_chip,
                           [pcvl.BasicState(input_n)],
                           "*")

        ca.compute()
        pcvl.pdisplay(ca)

        for idx1, output_state in enumerate(ca.output_states_list):
            result = outputstate_to_outcome(str(output_state))
            if result:
                outcome[result] += input_states_dict[input_n] * ca.distribution[0][idx1]

    to_plot[0].append(outcome['|1,0>'])
    to_plot[1].append(outcome['|1,1>'])
    to_plot[2].append(outcome['|0,1>'])

    v = 1 - 2 * outcome['|1,1>'] / (outcome['|1,0>'] + outcome['|0,1>'])

    print(f'Visibility = {round(v, 4)}')
    plt.show()


if __name__ == '__main__':
    mzi_ImperfectSource_pcvl(1, 0, 0.9)
