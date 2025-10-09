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

from packaging.version import Version

def keep_latest_versions(versions: list[str], mini: str = None) -> list[str]:
    """
    Keep the highest patch for all major-minor versions. Drops all pre-release patches.

    :param versions: A list of version strings 'vM.m.p'
    :param mini: (optional) The minimum version to keep. All versions below this are dropped.
    :return: The ordered list of version strings with the highest patch number.
    """
    if mini is not None:
        mini = Version(mini)

    version_dict: dict[tuple[int, ...], Version] = {}

    for one_version in versions:
        try:
            version = Version(one_version)
        except AttributeError:
            continue
        if not version.is_prerelease and (mini is None or version >= mini):  # filter alpha,beta...
            major_minor = version.release[:2]
            if major_minor not in version_dict or version > version_dict[major_minor]:
                version_dict[major_minor] = version

    latest_versions = sorted(version_dict.values())
    return list(map(lambda v: f"v{v}", latest_versions))
