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

import time
from datetime import datetime

from tqdm import tqdm

from perceval.serialization import InputArchive, OutputArchive, Serialization
from perceval.utils import FileFormat, PersistentData
from perceval.utils.logging import channel, get_logger

from .execution import Execution
from .execution_status import RunningStatus


FILE_EXT_EGRP = "egrp"


class ExecutionGroup:
    """A named, persistent collection of :class:`Execution` objects.

    An existing group is loaded when its name is reused. Every mutation is
    automatically stored with the archive serialization system.

    The ExecutionGroup class can perform various tasks such as:
    - Saving information for a collection of jobs, whether they have been sent to the cloud or not.
    - Running jobs within the group either in parallel or sequentially.
    - Rerunning failed jobs within the group.
    - Retrieving all results at once.

    :param name: Name uniquely identifying the group on disk.
    """

    STATUS_REFRESH_DELAY = 5

    def __init__(self, name: str, folder_path: str = "./execution_groups"):
        name = name.removesuffix(f".{FILE_EXT_EGRP}")

        if not isinstance(name, str) or len(name) == 0:
            raise TypeError("An execution group name must be a non-empty string")

        self._persistent_data = PersistentData(folder_path)  # class used as utils provider - Makes the folder
        self._folder_path = self._persistent_data.directory
        self._name = name
        self._filename = f"{self._name}.{FILE_EXT_EGRP}"

        if self._persistent_data.has_file(self._filename):
            get_logger().info(
                f"Execution Group with name {name} exists; subsequent executions will be appended to it",
                channel.user,
            )
            text = self._persistent_data.read_file(self._filename, FileFormat.TEXT)
            archive = InputArchive.from_text(text)
            self.created_date = datetime.fromtimestamp(Serialization.deserialize(archive))
            self.modified_date = datetime.fromtimestamp(Serialization.deserialize(archive))
            self._executions = Serialization.deserialize(archive)
        else:
            now = datetime.now()

            self.created_date = now
            self.modified_date = now
            self._executions: list[Execution] = []

            self._write_to_file()

    def __len__(self):
        return len(self._executions)

    def __getitem__(self, index):
        return self._executions[index]

    @property
    def name(self) -> str:
        return self._name

    @property
    def folder_path(self) -> str:
        return self._folder_path

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def executions(self) -> list[Execution]:
        """Return the executions in insertion order."""
        return list(self._executions)

    def _write_to_file(self) -> None:
        """
        Writes execution group data to disk
        """
        self.modified_date = datetime.now()
        archive = OutputArchive()
        Serialization.serialize(self.created_date.timestamp(), archive)
        Serialization.serialize(self.modified_date.timestamp(), archive)
        Serialization.serialize(self._executions, archive)
        # TODO: use JSON format so it is "more" readable? Compress? Leave the choice to the user?
        self._persistent_data.write_file(self._filename, archive.to_text(compress=True), FileFormat.TEXT)

    def add(self, execution: Execution, **kwargs) -> None:
        """Add an execution then saves the group.

        :param execution: an execution to add to the list of current execution group
        :param kwargs: parameters to pass to the execution, that will be used when it is launched.
        """
        if not isinstance(execution, Execution):
            raise TypeError(f"Only an Execution can be added to an ExecutionGroup (got {type(execution).__name__})")
        if execution in self._executions:
            raise ValueError(f"Duplicate execution {execution.name} detected")
        if kwargs:
            execution.computation.add_params(**kwargs)  # Raises an error with ComputationIterator
        self._executions.append(execution)
        self._write_to_file()

    def _update_execution_statuses(self) -> None:
        changed = False
        for execution in self._executions:
            if execution.was_sent and not execution._status.completed:
                old_status = execution._status.status
                current_status = execution.status.status
                changed |= old_status != current_status
        if changed:
            self._write_to_file()

    def progress(self) -> dict:
        """Summarize finished and unfinished executions.

        return format: {"Total": int, "Finished": [int, {"successful": int, "unsuccessful": int}], "Unfinished": [int, {"sent": int, "not sent": int}]}
        """
        self._update_execution_statuses()
        unsent = successful = unsuccessful = active = 0
        for execution in self._executions:
            if not execution.was_sent:
                unsent += 1
            elif execution._status.success:
                successful += 1
            elif execution._status.waiting or execution._status.running:
                active += 1
            else:
                unsuccessful += 1

        return {
            "Total": len(self._executions),
            "Finished": [successful + unsuccessful,
                         {"successful": successful, "unsuccessful": unsuccessful}],
            "Unfinished": [active + unsent, {"sent": active, "not sent": unsent}],
        }

    def track_progress(self) -> None:
        """
        Display progress bars until no execution is active using 'tqdm'.
        The bars represent the number of Successful, Active (i.e. launched and not finished), and Unsuccessful executions
        """
        total = len(self._executions)

        # define tqdm bars
        bar_format = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}"
        success_bar = tqdm(total=total, bar_format=bar_format, desc="Successful Executions", position=0, leave=True)
        active_bar = tqdm(total=len(self.list_active_executions()), bar_format=bar_format,
                          desc="Running/Waiting Executions", position=1, leave=True)
        inactive_bar = tqdm(total=total, bar_format=bar_format, desc="Inactive/Unsuccessful Executions",
                            position=2, leave=True)
        try:
            while True:
                current = self.progress()
                success_bar.n = current["Finished"][1]["successful"]
                active_bar.n = current["Unfinished"][1]["sent"]
                inactive_bar.n = current["Finished"][1]["unsuccessful"] + current["Unfinished"][1]["not sent"]
                for bar in (success_bar, active_bar, inactive_bar):
                    bar.refresh()
                if active_bar.n == 0:
                    break
                time.sleep(self.STATUS_REFRESH_DELAY)
        finally:
            success_bar.close()
            active_bar.close()
            inactive_bar.close()

    def _filter_by_running_status(self, statuses: list[RunningStatus]) -> list[Execution]:
        self._update_execution_statuses()
        return [
            execution for execution in self._executions
            if execution.was_sent and execution._status.status in statuses
        ]

    def list_successful_executions(self) -> list[Execution]:
        """
        Returns a list of all Executions in the group that have run successfully.
        """
        return self._filter_by_running_status([RunningStatus.SUCCESS])

    def list_active_executions(self) -> list[Execution]:
        """
        Returns a list of all Executions in the group that are currently active - those with a Running or
        Waiting status.
        """
        return self._filter_by_running_status([RunningStatus.RUNNING, RunningStatus.WAITING])

    def list_unsuccessful_executions(self) -> list[Execution]:
        """
        Returns a list of all Executions in the group that have run unsuccessfully - errored or canceled
        """
        return self._filter_by_running_status([RunningStatus.ERROR, RunningStatus.CANCELED])

    def list_unsent_executions(self) -> list[Execution]:
        """
        Returns a list of all Executions in the group that have not been launched
        """
        return [execution for execution in self._executions if not execution.was_sent]

    def _launch_wait_executions(self, delay: float, rerun: bool,
                                replace_failed_executions: bool = False, sequential: bool = False) -> None:
        """
        Launches or reruns jobs in the group on Cloud in a parallel/sequential manner.

        :param delay: number of seconds to wait between the launch of two consecutive executions
        :param rerun: if True rerun failed executions or run unsent executions
        :param replace_failed_executions: replace the rerun executions in the ExecutionGroup,
         else keep the failed in addition of the rerun ones
        :param sequential: if True, only one execution is run at a time, including if several tokens have job availability
        """
        executions_to_run = (self.list_unsuccessful_executions() if rerun else self.list_unsent_executions())
        awaited = set()
        success_count = failure_count = 0
        progress_bar = tqdm(total=len(executions_to_run),
                            bar_format="{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}|{desc}",
                            desc="Successful: 0, Failed: 0")
        try:
            while executions_to_run or awaited:
                launch_candidates = [] if sequential and awaited else list(executions_to_run)
                for execution in launch_candidates:
                    if execution.computer.available_jobs <= 0:
                        continue
                    if delay:
                        time.sleep(delay)

                    executions_to_run.remove(execution)
                    execution.job_group_name = self.name

                    if rerun:
                        rerun_execution = execution.rerun()
                        if replace_failed_executions:
                            index = self._executions.index(execution)
                            self._executions[index] = rerun_execution
                        else:
                            self._executions.append(rerun_execution)
                        execution = rerun_execution
                    else:
                        execution.execute_async()

                    awaited.add(execution)
                    self._write_to_file()
                    if sequential:
                        break

                finished = set()
                for execution in awaited:
                    if execution.status.completed:
                        if execution.status.success:
                            success_count += 1
                        else:
                            failure_count += 1
                        finished.add(execution)

                        self._write_to_file()

                        progress_bar.update(1)
                        progress_bar.set_description_str(f"Successful: {success_count}, Failed: {failure_count}")

                awaited.difference_update(finished)
                if executions_to_run or awaited:
                    time.sleep(self.STATUS_REFRESH_DELAY)
        finally:
            progress_bar.close()

    def run_sequential(self, delay: float) -> None:
        """
        Launches the unsent executions in the group in a sequential manner with a
        user-specified delay between the completion of one execution and the start of the next.

        :param delay: number of seconds to wait between launching jobs on cloud
        """
        self._launch_wait_executions(delay, rerun=False, sequential=True)

    def rerun_failed_sequential(self, delay: float, replace_failed_executions: bool = True, **kwargs) -> None:
        """
        Reruns Failed executions in the group in a sequential manner with a user-specified delay between the
        completion of one execution and the start of the next.

        :param delay: number of seconds to wait between re-launching jobs on cloud
        :param replace_failed_executions: Indicates whether a new job created from a rerun should replace the previously
                                    failed job (defaults to True).
        """
        # backward compatibility
        replace_failed_executions = kwargs.pop("replace_failed_jobs", replace_failed_executions)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        self._launch_wait_executions(delay, rerun=True,
                                     replace_failed_executions=replace_failed_executions, sequential=True)

    def run_parallel(self) -> None:
        """
        Launches all the unsent executions in the group, running them in parallel.
        The number of concurrent executions is determined by the capabilities of the computers.
        """
        self._launch_wait_executions(0, rerun=False)

    def rerun_failed_parallel(self, replace_failed_executions: bool = True, **kwargs) -> None:
        """
        Restart all failed executions in the group, running them in parallel.
        The number of concurrent executions is determined by the capabilities of the computers.

        :param replace_failed_executions: Indicates whether a new execution created from a rerun should replace the
                                          previously failed execution (defaults to True).
        """
        # backward compatibility
        replace_failed_executions = kwargs.pop("replace_failed_jobs", replace_failed_executions)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        self._launch_wait_executions(0, rerun=True,
                                     replace_failed_executions=replace_failed_executions)

    def launch_async_executions(self, concurrent_execution_count: int | None = None) -> None:
        """
        Launches up to concurrent_execution_count executions and returns without waiting for execution completion.

        Beware that local executions that are not finished will have to be started from scratch again if the scripts stops.
        In that case, use ``track_progress()`` to block, or launch your executions using ``run_sequential()`` or ``run_parallel()``.

        :param concurrent_execution_count: maximum number of concurrent executions.
         If not specified, the maximum number allowed by the computers is launched.
        """
        executions = self.list_unsent_executions()
        launched = 0
        for execution in executions:
            if execution.computer.available_jobs <= 0:
                continue

            execution.job_group_name = self.name
            execution.execute_async()
            launched += 1
            self._write_to_file()

            if launched == concurrent_execution_count:
                break

        if not launched and len(executions):
            get_logger().warn(f"{self.name}: no execution will be run as there is no slot available", channel.user)
        get_logger().info(
            f"{self.name}: {launched} executions launched / {len(self.list_unsent_executions())} unsent executions remaining",
            channel.user,
        )

    def relaunch_async_failed_executions(self, replace_failed_executions: bool = True,
                                         concurrent_execution_count: int | None = None) -> None:
        """
        Relaunches up to concurrent_execution_count failed executions and returns without waiting for execution completion.


        Beware that local executions that are not finished will have to be started from scratch again if the scripts stops.
        In that case, use ``track_progress()`` to block,
        or launch your executions using ``rerun_failed_sequential()`` or ``rerun_failed_parallel()``.

        :param concurrent_execution_count: maximum number of concurrent executions.
         If not specified, the maximum number allowed by the computers is used.
        :param replace_failed_executions: replace the rerun executions in the execution group,
         else keep the failed ones in addition of the rerun ones
        """

        executions = self.list_unsuccessful_executions()
        launched = 0
        for execution in executions:
            if execution.computer.available_jobs <= 0:
                continue

            execution.job_group_name = self.name
            index = self._executions.index(execution)
            rerun_execution = execution.rerun()
            if replace_failed_executions:
                self._executions[index] = rerun_execution
            else:
                self._executions.append(rerun_execution)
            launched += 1
            self._write_to_file()

            if launched == concurrent_execution_count:
                break

        if not launched and len(executions):
            get_logger().warn(f"{self.name}: no execution will be run as there is no slot available", channel.user)
        get_logger().info(
            f"{self.name}: {launched} executions launched / {len(self.list_unsent_executions())} unsent executions remaining",
            channel.user,
        )

    # JobGroup-compatible spellings ease migration of calling code.
    def launch_async_jobs(self, concurrent_job_count: int | None = None) -> None:
        self.launch_async_executions(concurrent_job_count)

    def relaunch_async_failed_jobs(self, replace_failed_jobs: bool = True,
                                   concurrent_job_count: int | None = None) -> None:
        self.relaunch_async_failed_executions(replace_failed_jobs, concurrent_job_count)

    def cancel_all(self) -> None:
        """
        Cancels all started and not completed executions in the group.
        """
        for execution in self._executions:
            if execution.was_sent and not execution.is_complete:
                try:
                    execution.cancel()
                except RuntimeError:
                    pass
                self._write_to_file()

    def get_results(self) -> list[dict | None]:
        """
        Retrieve results for all completed executions in the group.
        Non-completed executions will add None to the resulting list.
        """
        self._update_execution_statuses()
        results = []
        for execution in self._executions:
            if execution._status.maybe_completed:
                try:
                    results.append(execution.get_results())
                    self._write_to_file()
                except RuntimeError:
                    results.append(None)
            else:
                results.append(None)
        return results
