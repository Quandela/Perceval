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
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from exqalibur.exqalibur import FockState

from perceval import SimulatedComputer, PayloadGenerator, Computation, Experiment, NoiseModel, ComputationIterator, PS, \
    P, PostSelect, Circuit, BS, PayloadUpdater, Processor, CommandFactory
from .._test_utils import assert_experiment_equals
from perceval.utils.constants import KEY_COMPUTATION, KEY_EXPERIMENT

computer = SimulatedComputer("SLOS")
noise = NoiseModel(0.8)
N_SHOTS = 1000
N_SAMPLES = 100

def get_experiment(use_parameter=False):
    e = Experiment(2, name="test")
    e.add(0, PS(P("phi") if use_parameter else 2))
    e.add_herald(0, 1)
    e.set_postselection(PostSelect("[0] < 2"))
    e.min_detected_photons_filter(1)
    e.with_input(FockState([3]))
    return e


def get_computation(use_parameter=False) -> Computation:
    comp = Computation(computer.get_command("sample_count"), get_experiment(use_parameter))
    comp.add_params(max_shots=N_SHOTS, max_samples=N_SAMPLES)

    return comp

def get_expected() -> dict:
    comp = get_computation()
    parameters = {"compute_physical_logical_perf": True}

    # In any case, the old format doesn't allow mitigations
    return PayloadGenerator.from_computation(comp, parameters=parameters, noise=noise)


def get_expected_iterations() -> dict:
    comp = get_computation(True)

    comp = ComputationIterator(comp)
    comp.add_iteration(circuit_params = {"phi" : 2})
    comp.add_iteration(circuit_params = {"phi" : 3})

    parameters = {"compute_physical_logical_perf": True}

    # In any case, the old format doesn't allow mitigations
    return PayloadGenerator.from_computation(comp, parameters=parameters, noise=noise)


def compare_computations(left: Computation | ComputationIterator, right: Computation | ComputationIterator):
    # We don't care about the noise here
    left.experiment.noise = None
    right.experiment.noise = None
    # We don't care about the experiment name as well
    left.experiment.name = "name"
    right.experiment.name = "name"
    assert_experiment_equals(left.experiment, right.experiment)
    assert left.command == right.command
    assert left.parameters == right.parameters
    assert left.job_name == right.job_name
    assert left.job_group_name == right.job_group_name

    if isinstance(left, ComputationIterator):
        assert isinstance(right, ComputationIterator)
        assert left.iterations == right.iterations
        assert left.parameters == right.parameters
        compare_computations(left.base_computation, right.base_computation)


def compare_payloads(left: dict, right: dict):
    assert left.keys() == right.keys()

    if KEY_COMPUTATION in left:
        comp_left = left.pop(KEY_COMPUTATION)
        comp_right = right.pop(KEY_COMPUTATION)
        compare_computations(comp_left, comp_right)

    if KEY_EXPERIMENT in left:
        exp_left = left.pop(KEY_EXPERIMENT)
        exp_right = right.pop(KEY_EXPERIMENT)
        assert_experiment_equals(exp_left, exp_right)

    assert left == right


def test_payload_update_from_v0():
    # No changes in 0.11, 0.12
    # Empty payload
    res = PayloadUpdater.update_payload({"toto": 1}, None, target_payload_version=1)
    assert res == {"toto": 1}  # Experiment should not be added here

    # Minimum payload for experiment
    minimum_payload = {"toto": 2,
                       "circuit": Circuit(3) // BS(),
                       "input_state": FockState([0, 1, 2])}

    target_experiment = Experiment(3)
    target_experiment.set_circuit(minimum_payload["circuit"])
    target_experiment.with_input(minimum_payload["input_state"])

    res = PayloadUpdater.update_payload(minimum_payload, None, target_payload_version=1)

    assert minimum_payload["toto"] == 2
    assert_experiment_equals(target_experiment, res[KEY_EXPERIMENT])

    # Full payload
    target_experiment = get_experiment()

    full_payload = {'command': 'probs',
                    'job_context': {'result_mapping': ['perceval.utils', 'probs_to_sample_count'],
                                    'mapping_delta_parameters': {'max_samples': N_SAMPLES, 'max_shots': N_SHOTS}},
                    "circuit": target_experiment.unitary_circuit(),
                    "input_state": target_experiment.input_state,
                    "noise": noise,
                    "postselect": target_experiment.post_select_fn,
                    "heralds": target_experiment.heralds,
                    "parameters": {
                        "min_detected_photons": target_experiment.min_photons_filter,
                        "compute_physical_logical_perf": True
                        },

                    }

    res = PayloadUpdater.update_payload(full_payload, None, target_payload_version=1)
    res["experiment"].name = target_experiment.name
    target_experiment.noise = noise
    assert_experiment_equals(target_experiment, res["experiment"])

    res = PayloadUpdater.update_payload(full_payload, computer)
    compare_payloads(res, get_expected())


def test_translation_from_v1():
    # No change in perceval 1.1 and 1.2
    e = get_experiment()
    e.noise = noise
    payload = {'command': 'samples',
               'experiment': e,
               'job_context': {'result_mapping': ['perceval.utils', 'samples_to_sample_count']},
               'max_samples': N_SAMPLES,
               'max_shots': N_SHOTS,
               'parameters': {"compute_physical_logical_perf": True}}
    new_payload = PayloadUpdater.update_payload(payload, computer)
    compare_payloads(new_payload, get_expected())

    e = get_experiment()
    e.noise = noise
    payload = {'command': 'probs',
               'experiment': e,
               'job_context': {'result_mapping': ['perceval.utils', 'probs_to_sample_count'],
                               'mapping_delta_parameters': {'max_samples': N_SAMPLES, 'max_shots': N_SHOTS}},
               'parameters': {"compute_physical_logical_perf": True}
               }

    new_payload = PayloadUpdater.update_payload(payload, computer)
    compare_payloads(new_payload, get_expected())

    e = get_experiment(True)
    e.noise = noise
    payload_iterations = {'command': 'sample_count',
                          'experiment': e,
                          'iterator': [{"circuit_params": {"phi" : 2}}, {"circuit_params": {"phi" : 3}}],
                          'job_context': None,
                          'max_samples': N_SAMPLES,
                          'max_shots': N_SHOTS,
                          'parameters': {"compute_physical_logical_perf": True}}

    new_payload_iterations = PayloadUpdater.update_payload(payload_iterations, computer)
    compare_payloads(new_payload_iterations, get_expected_iterations())


def test_downgrade_to_v1():
    e = get_experiment()
    p = Processor("CliffordClifford2017")  # Only has "samples" exposed

    comp = Computation(CommandFactory.sample_count, e)
    comp.add_params(max_shots = N_SHOTS, max_samples = N_SAMPLES)

    parameters = {"compute_physical_logical_perf": True}
    expected = {'command': 'samples',
               'experiment': e,
               'job_context': {'result_mapping': ['perceval.utils', 'samples_to_sample_count']},
               'max_samples': N_SAMPLES,
               'max_shots': N_SHOTS,
               'parameters': parameters}

    payload = PayloadGenerator.from_computation(comp, parameters=parameters)
    payload = PayloadUpdater.update_payload(payload, p, target_payload_version=1)

    compare_payloads(payload, expected)
