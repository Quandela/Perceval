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

# Parameters
KEY_MAX_SHOTS = "max_shots"
KEY_MAX_SAMPLES = "max_samples"
KEY_COMPILATION_SEED = "compilation_seed"

# Results keys
KEY_RESULTS = "results"
KEY_SHOTS_USED = "nb_shots_used"
KEY_RESULTS_LIST = "results_list"
KEY_ITERATION = "iteration"
KEY_GLOBAL_PERF = "global_perf"
KEY_PHYSICAL_PERF = "physical_perf"
KEY_LOGICAL_PERF = "logical_perf"

# Result legacy keys
KEY_JOB_CONTEXT = "job_context"
KEY_RESULT_MAPPING = "result_mapping"
KEY_MAPPING_PARAMETERS = "mapping_delta_parameters"

# Payload keys
KEY_COMPUTATION = "computation"
KEY_MITIGATIONS = "mitigations"
KEY_PARAMETERS = "parameters"
KEY_NOISE = "noise"

# Payload legacy keys
KEY_COMMAND = "command"
KEY_ITERATOR = "iterator"
KEY_CIRCUIT = "circuit"
KEY_CIRCUIT_PARAMS = "circuit_params"
KEY_INPUT_STATE = "input_state"
KEY_COUNT = "count"
KEY_EXPERIMENT = "experiment"
KEY_POSTSELECT = "postselect"
KEY_HERALDS = "heralds"
KEY_MIN_DETECTED_PHOTONS = "min_detected_photons"

# Global data keys
KEY_VERSION = "pcvl_version"
KEY_PROCESS_ID = "process_id"
KEY_PAYLOAD = "payload"
KEY_PLATFORM_NAME = "platform_name"
KEY_JOB_NAME = "job_name"
KEY_JOB_GROUP_NAME = "job_group_name"
