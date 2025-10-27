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

import re
import pytest
import time
import warnings

from functools import wraps
from pathlib import Path
from unittest.mock import MagicMock

from perceval import BasicState, Parameter, Expression
from perceval.components import ACircuit, PS, IDetector, AFFConfigurator, FFConfigurator, FFCircuitProvider, \
    BSLayeredPPNR, Detector, Experiment, Circuit, AProcessor, AComponent
from perceval.components.abstract_component import AParametrizedComponent
from perceval.utils import StateVector, SVDistribution
from perceval.utils.logging import channel
from perceval.rendering import Format, pdisplay_to_file
from perceval.rendering.circuit import ASkin, PhysSkin
from perceval.algorithm import AProcessTomography


TEST_IMG_DIR = Path(__file__).resolve().parent / 'imgs'


@pytest.fixture(scope="session")
def save_figs(pytestconfig):
    return pytestconfig.getoption("save_figs")


def _norm(svg):
    svg = svg.replace(" \n", "\n")
    svg = re.sub(r'url\(#.*?\)', 'url(#staticClipPath)', svg)
    svg = re.sub(r'<clipPath id=".*?">', '<clipPath id="staticClipPath">', svg)
    svg = re.sub(r'<dc:date>(.*)</dc:date>', '<dc:date></dc:date>', svg)
    svg = re.sub(r'<dc:title>(.*)</dc:title>', '<dc:title></dc:title>', svg)
    return svg


def _check_svg(test_path, ref_path, collection: str):
    with open(test_path) as f_test:
        test = _norm("".join(f_test.readlines()))
    with open(ref_path) as f_ref:
        ref = _norm("".join(f_ref.readlines()))
    m_test = re.search(rf'<g id="{collection}Collection.*?>((.|\n)*?)</g>', test)
    m_ref = re.search(rf'<g id="{collection}Collection.*?>((.|\n)*?)</g>', ref)
    if not m_test:
        return False, "cannot find patch in test"
    if not m_ref:
        return False, "cannot find patch in ref"
    m_test_patch = re.sub(r'url\(#.*?\)', "url()", m_test.group(1))
    m_ref_patch = re.sub(r'url\(#.*?\)', "url()", m_ref.group(1))
    if m_test_patch != m_ref_patch:
        return False, "test and ref are different"
    return True, "ok"


def _check_circuit(test_path, ref_path):
    return _check_svg(test_path, ref_path, 'Patch')


def _check_qpt(test_path, ref_path):
    return _check_svg(test_path, ref_path, 'Line3D')


def _save_or_check(c, tmp_path, circuit_name, save_figs, recursive=False, compact=False,
                   skin_type: type[ASkin] = PhysSkin) -> None:
    img_path = (TEST_IMG_DIR if save_figs else tmp_path) / \
        Path(circuit_name + ".svg")
    skin = skin_type(compact)

    if isinstance(c, AProcessTomography):
        pdisplay_to_file(c, img_path, output_format=Format.MPLOT)
    elif isinstance(c, (AComponent, AProcessor)):
        pdisplay_to_file(c, img_path, output_format=Format.MPLOT,
                              recursive=recursive, skin=skin)
    else:
        raise NotImplementedError(f"_save_or_check not implemented for {type(c)}")

    if save_figs:
        with open(img_path) as f_saved:
            saved = "".join(f_saved.readlines())
        saved = _norm(saved)
        with open(img_path, "w") as fw_saved:
            fw_saved.write(saved)
    else:
        if isinstance(c, AProcessTomography):
            ok, msg = _check_qpt(img_path, TEST_IMG_DIR /
                                 Path(circuit_name + ".svg"))
        elif isinstance(c, (AComponent, AProcessor)):
            ok, msg = _check_circuit(img_path, TEST_IMG_DIR /
                                     Path(circuit_name + ".svg"))
        else:
            raise NotImplementedError(f"_save_or_check not implemented for {type(c)}")
        assert ok, msg
