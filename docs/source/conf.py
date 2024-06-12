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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from git import Repo

sys.path.insert(0, os.path.relpath("../"))

from source import build_catalog
from perceval import PMetadata


def version_highter_then(v1, v2):
    """compare two version"""
    v1 = list(map(int, re.findall(r"\d+", v1)[0:3]))
    v2 = list(map(int, re.findall(r"\d+", v2)[0:3]))
    for i, charac in enumerate(v1):
        if i < len(v2):
            if v2[i] > charac:
                return False
            elif v2[i] == charac:
                continue
    return True


def keep_latest_versions(versions, mini=None):
    """keep latest version"""
    version_dict = {}

    for one_version in versions:
        # major_version = re.match(r"v\d+", one_version).group()
        try:
            major_version = re.match(r"v\d+\.(\d+)", one_version).groups()
        except AttributeError:
            major_version = '0.0.0'
        if (
            major_version not in version_dict
            or one_version > version_dict[major_version]
        ) and (mini is not None and version_highter_then(one_version, mini)):
            version_dict[major_version] = one_version

    latest_versions = list(version_dict.values())
    return sorted(latest_versions, key=lambda x: tuple(map(int, re.findall(r"\d+", x))))


REPO_PATH = Path(__file__).parent.parent.parent.resolve()

build_directory = os.path.join(REPO_PATH, "docs", "build")
if not os.path.exists(build_directory):
    os.makedirs(build_directory)
build_catalog.build_catalog_rst(os.path.join(build_directory, "catalog.rst"))

repo = Repo(REPO_PATH)
tags = [tag.name for tag in repo.tags]
versions = keep_latest_versions(tags, "v0.6")
versions_string = "".join([f"({one_version})|" for one_version in versions])[:-1]
versions_regex = re.compile(f"^{versions_string}$")

# -- Project information -----------------------------------------------------

project = PMetadata.name()
copyright = f"{datetime.now().year}, {PMetadata.author()[0].capitalize() + PMetadata.author()[1:]}"
author = PMetadata.author()
release = PMetadata.short_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinx_multiversion",
]

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = versions_regex

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = None

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None

# Pattern for released versions
smv_released_pattern = r".*"

smv_regex_name = r"(.*)\..*"

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 2,
    "titles_only": False,
    "display_version": True,
}

html_style = "css/style.css"
html_logo = "_static/img/Perceval logo white 160X160.png"
html_favicon = "_static/img/Perceval icon white 32x32.ico"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
