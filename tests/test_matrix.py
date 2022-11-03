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

from pathlib import Path

import perceval as pcvl
from perceval.rendering.pdisplay import pdisplay_matrix

import numpy as np
import sympy as sp

TEST_DATA_DIR = Path(__file__).resolve().parent / 'data'


def test_new_np():
    u = sp.eye(3)
    M = pcvl.Matrix(u)
    assert not(M is u) and np.array_equal(M, u)
    assert not M.is_symbolic()


def test_new_textarray():
    M = pcvl.Matrix("1 2 3\n4 5 6\n7 8 9")
    assert np.array_equal(M, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert not M.is_symbolic()


def test_new_shape0():
    M = pcvl.Matrix((3,))
    assert M.shape == (3, 1)


def test_new_shape1():
    M = pcvl.Matrix(3)
    assert M.shape == (3, 3)


def test_new_shape2():
    M = pcvl.Matrix((1, 3))
    assert M.shape == (1, 3)


def test_eye_n():
    M = pcvl.Matrix.eye(3)
    assert isinstance(M, np.ndarray)
    assert M.shape == (3, 3)
    assert pdisplay_matrix(M) == "⎡1  0  0⎤\n⎢0  1  0⎥\n⎣0  0  1⎦"


def test_eye_s():
    M = pcvl.Matrix.eye(3, use_symbolic=True)
    assert pdisplay_matrix(M) == "⎡1  0  0⎤\n⎢0  1  0⎥\n⎣0  0  1⎦"
    assert isinstance(M, sp.Matrix)
    assert M.shape == (3, 3)


def test_zero_n():
    M = pcvl.Matrix.zeros((3, 3))
    assert isinstance(M, np.ndarray)
    assert M.shape == (3, 3)
    assert pdisplay_matrix(M) == "⎡0  0  0⎤\n⎢0  0  0⎥\n⎣0  0  0⎦"


def test_zero_s():
    M = pcvl.Matrix.zeros((3, 3), use_symbolic=True)
    assert pdisplay_matrix(M) == "⎡0  0  0⎤\n⎢0  0  0⎥\n⎣0  0  0⎦"
    assert isinstance(M, sp.Matrix)
    assert M.shape == (3, 3)
    assert M.is_symbolic()


def test_read_fromfile1():
    with open(TEST_DATA_DIR / 'u_hom', "r") as f:
        M = pcvl.Matrix(f)
        assert M.shape == (2, 2)


def test_read_fromfile_complex():
    with open(TEST_DATA_DIR / 'u_random_8', "r") as f:
        M = pcvl.Matrix(f)
        assert M.shape == (8, 8)
        assert float(abs((M[0, 0] - (-0.3233639242889934+0.10358205117585266*sp.I))**2)) < 1e-10
        assert M.is_unitary()


def test_check_unitary_numeric():
    with open(TEST_DATA_DIR / 'u_hom', "r") as f:
        M = pcvl.Matrix(f)
        assert M.is_unitary()


def test_check_unitary_numeric_sym():
    with open(TEST_DATA_DIR / 'u_hom_sym', "r", encoding="utf-8") as f:
        M = pcvl.Matrix(f)
        assert M.is_unitary()


def atest_str_1():
    M = pcvl.Matrix([1, "2*x", 3])
    assert M.shape == (3, 1)
    assert str(M) == "Matrix([[1], [2*x], [3]])"


def atest_repr():
    M = pcvl.Matrix([[1, "sqrt(2)"], ["-cos(x)", "-1"]])
    assert sp.pretty(M) == "⎡   1     √2⎤\n⎢           ⎥\n⎣-cos(x)  -1⎦"


def atest_repr_1():
    M = pcvl.Matrix([1, "2*x", 3])
    assert sp.pretty(M) == "⎡ 1 ⎤\n⎢   ⎥\n⎢2⋅x⎥\n⎢   ⎥\n⎣ 3 ⎦"


def atest_repr_2():
    M = pcvl.Matrix([[1, "2*x", 3]])
    assert pdisplay_matrix(M) == "[1  2*x  3]"


def test_repr_3():
    M = pcvl.Matrix([[1, 0], [0, 1]])
    assert pdisplay_matrix(M) == "⎡1  0⎤\n⎣0  1⎦"


def atest_str_2():
    M = pcvl.Matrix([[1, "sqrt(2)"], ["-cos(x)", "-1"]])
    assert M.shape == (2, 2)
    assert str(M) == "Matrix([[1, sqrt(2)], [-cos(x), -1]])"


def atest_tonumpy():
    M = pcvl.Matrix([["1/sqrt(2)", "-1/sqrt(2)"], ["1/sqrt(2)", "1/sqrt(2)"]])
    assert np.allclose(M.tonp(), np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]]))


def test_keeppcvlcls():
    M = pcvl.Matrix([3])
    N = pcvl.Matrix([4])
    assert isinstance(M, pcvl.Matrix)
    assert isinstance(N, pcvl.Matrix)
    MN = M*N
    assert MN == pcvl.Matrix([12])
    assert isinstance(MN, pcvl.Matrix)


def test_keeppcvlcls_s():
    M = pcvl.Matrix([3], use_symbolic=True)
    N = pcvl.Matrix([4], use_symbolic=True)
    assert isinstance(M, pcvl.Matrix)
    assert isinstance(N, pcvl.Matrix)
    MN = M*N
    assert MN == pcvl.Matrix([12], use_symbolic=True)
    assert isinstance(MN, pcvl.Matrix)


def test_genunitary():
    M = pcvl.Matrix.random_unitary(3)
    assert isinstance(M, pcvl.Matrix)
    assert M.shape == (3, 3)
    assert M.is_unitary()
