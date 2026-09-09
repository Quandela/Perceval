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
    """Provider-facing interface between :class:`RemoteComputer` and a remote platform.

    A communication layer translates Perceval payloads and lifecycle operations into calls to a
    provider service. Implementations should not retain mutable state for individual jobs: the
    remote identifier returned by :meth:`send` is passed to all later status, result, and
    cancellation calls. Provider-wide caches and session state are allowed.

    Implementations may be used concurrently by several executions. They should therefore avoid
    shared per-job state and make any mutable cache or session state concurrency-safe.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the remote platform."""
        pass

    @abstractmethod
    def get_specs(self) -> PlatformSpecs:
        """Return the capabilities and constraints of the target platform.

        :return: The target platform specifications.
        """
        pass

    @abstractmethod
    def send(self, payload: dict) -> RemoteId:
        """Submit a Perceval payload to the remote platform.

        The payload is produced by :class:`PayloadGenerator` and contains a
        :class:`Computation` or :class:`ComputationIterator`, plus optional noise, mitigation, and
        computer-specific parameters. An implementation is responsible for translating or
        serializing it into its provider's request format.

        :param payload: The payload to submit.
        :return: A provider-defined, stable job identifier accepted by the other job methods.
        """
        pass

    @abstractmethod
    def get_results(self, remote_id: RemoteId) -> dict:
        """Retrieve the results of a completed remote job.

        :param remote_id: Identifier returned by :meth:`send`.
        :return: A Perceval result dictionary, normally containing ``"results"`` or
            ``"results_list"`` and any associated performance data.
        :raises Exception: If the provider cannot retrieve the results.
        """
        pass

    @abstractmethod
    def get_job_status(self, remote_id: RemoteId, refresh_errors: int = 0) -> ExecutionStatus | None:
        """Retrieve the current status of a remote job.

        Returning ``None`` signals a transient refresh failure. :class:`RemoteComputer` keeps the
        previous status and passes the number of consecutive failures back on the next call, so an
        implementation can eventually raise a permanent error.

        :param remote_id: Identifier returned by :meth:`send()`.
        :param refresh_errors: Number of consecutive previous calls that returned ``None``.
        :return: The current execution status, or ``None`` after a recoverable refresh failure.
        """
        pass

    @abstractmethod
    def get_remote_status(self) -> str:
        """Return the provider's current platform status as a human-readable string."""
        pass

    @abstractmethod
    def get_performances(self) -> dict:
        """Return the platform performance characterization.

        The mapping is exposed as :attr:`RemoteComputer.performance` and may also be used to derive
        its noise model.
        """
        pass

    @abstractmethod
    def get_commands(self) -> list[Command]:
        """Return the commands implemented by the remote platform.

        Each command name must identify an operation that the provider can execute, and its
        signature must describe the parameters accepted by that operation.
        """
        pass

    @abstractmethod
    def cancel(self, remote_id: RemoteId) -> None:
        """Request cancellation of a remote job.

        The request need not make cancellation immediate; later calls to :meth:`get_job_status`
        communicate the final state.

        :param remote_id: Identifier returned by :meth:`send()`.
        """
        pass

    @abstractmethod
    def get_availability(self) -> int:
        """Return the number of concurrent jobs that can currently be submitted by the user.

        Return ``0`` when the provider has no free capacity or availability cannot be established.
        Always returns ``1`` if there is no API call to get this number.
        The value must not be negative.
        """
        pass

    @abstractmethod
    def get_platform_details(self) -> dict:
        """Return any kind of information that the platform could provide.
        No particular field is guaranteed to exist here, nor do their types.
        This should only be used for information purpose or on specific computers,
        and always check that the read field exists before accessing it.
        """
        pass

    def start_session(self) -> None:
        """Start or acquire a provider session.

        Stateless implementations may keep the default no-op implementation.
        """
        pass

    def stop_session(self) -> None:
        """Stop or release the current provider session without deleting it.

        Stateless implementations may keep the default no-op implementation.
        """
        pass

    def delete_session(self) -> None:
        """Permanently delete the current provider session.

        Stateless implementations may keep the default no-op implementation.
        """
        pass
