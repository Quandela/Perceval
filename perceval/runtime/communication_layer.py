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

from abc import ABC, abstractmethod
from typing import TypeVar

from .execution_status import ExecutionStatus
from .command import Command
from .platform_specs import PlatformSpecs

RemoteId = TypeVar("RemoteId")


class CommunicationLayer(ABC):
    """
    This class is responsible for the communication with the distant computer, with .
    Implementations of this class must remain const, except for potential non-job-related cache
    """

    @abstractmethod
    def get_specs(self) -> PlatformSpecs:
        """
        :return: The specs of the target platform
        """
        pass

    @abstractmethod
    def send(self, payload: dict) -> RemoteId:
        pass

    @abstractmethod
    def get_results(self, remote_id: RemoteId) -> dict:
        pass

    @abstractmethod
    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> ExecutionStatus | None:
        """
        :param remote_id:
        :param refresh_errors: The number of times in a row where this method returned None
        :return: The Job Status if it was available, None otherwise
        """
        pass

    @abstractmethod
    def get_remote_status(self) -> str:
        pass

    @abstractmethod
    def get_performances(self) -> dict:
        pass

    @abstractmethod
    def get_commands(self) -> list[Command]:
        pass

    @abstractmethod
    def cancel(self, remote_id: RemoteId) -> None:
        pass

    @abstractmethod
    def get_availability(self) -> int:
        """Returns the number of concurrent jobs currently available to be sent"""
        pass

    def start_session(self) -> None:
        """May be used to start a non-interrupted session. May do nothing on stateless implementations"""
        pass

    def stop_session(self) -> None:
        """May be used to stop a non-interrupted session. May do nothing on stateless implementations"""
        pass

    def delete_session(self) -> None:
        """May be used to delete a non-interrupted session. May do nothing on stateless implementations"""
        pass
