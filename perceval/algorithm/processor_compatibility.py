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

from abc import ABCMeta
from typing import Any

from perceval.runtime import AbstractComputer, SimulatedComputer, RemoteComputer
from perceval.runtime.legacy import AProcessor, Processor, RemoteProcessor
from perceval.providers.rpc_based_communication_layer import RPCBasedCommunicationLayer
from perceval.providers.kipu import KipuRPCHandler, KipuCommunicationLayer
from perceval.providers.quandela import QuandelaCommunicationLayer
from perceval.providers.quandela.rpc_handler import RPCHandler as QuandelaRPCHandler
from perceval.providers.scaleway import ScalewayCommunicationLayer, RPCHandler as ScalewayRPCHandler


# TODO: remove this file when removing the Processor support

def computer_from_processor(processor: AProcessor) -> AbstractComputer:
    # This method is a patch for places where a Processor is needed.
    # It should not be documented as it is intended for internal purpose only
    # This baseline should be adapted to specific cases if needed
    if isinstance(processor, Processor):
        computer = SimulatedComputer(processor.backend)

    elif isinstance(processor, RemoteProcessor):
        rpc = processor.get_rpc_handler()
        if isinstance(rpc, QuandelaRPCHandler):
            computer = RemoteComputer(QuandelaCommunicationLayer.from_rpc(rpc))
        elif isinstance(rpc, KipuRPCHandler):
            computer = RemoteComputer(KipuCommunicationLayer.from_rpc(rpc))
        elif isinstance(rpc, ScalewayRPCHandler):
            computer = RemoteComputer(ScalewayCommunicationLayer.from_rpc(rpc))
        else:
            computer = RemoteComputer(RPCBasedCommunicationLayer(rpc))

    else:
        raise NotImplementedError(f'No Computer can be obtained for the given processor of type "{type(processor).__name__}".')

    computer.noise = processor.experiment.noise
    computer.parameters = processor.parameters
    return computer


def adapt_arguments_with_processor(args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
    idx = None
    for i, arg in enumerate(args):
        if isinstance(arg, AProcessor):
            idx = i
            break

    if idx is not None:
        args = args[:idx] + (computer_from_processor(args[idx]), args[idx].experiment) + args[idx + 1:]
        return args, kwargs

    found = False
    for key, val in kwargs.items():
        if "processor" in key and isinstance(val, AProcessor):
            kwargs["computer"] = computer_from_processor(val)
            kwargs["experiment"] = val.experiment
            found = True
            break
    if found:
        del kwargs[key]

    return args, kwargs


class ProcessorCompatibilityMeta(type):
    def __call__(cls, *args, **kwargs):
        args, kwargs = adapt_arguments_with_processor(args, kwargs)
        return super().__call__(*args, **kwargs)


class AProcessorCompatibilityMeta(ABCMeta, ProcessorCompatibilityMeta):
    pass
