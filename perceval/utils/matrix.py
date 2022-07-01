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

from __future__ import annotations
from abc import ABC, abstractmethod
import io
import re
from typing import Literal, Iterator, Optional, Union

import numpy as np
import sympy as sp

from perceval.utils.format import simple_complex
from .mlstr import mlstr


class Matrix(ABC):
    """
        This parent class is the gateway :class:`MatrixN` or :class:`MatrixS` - based on ``use_symbolic``,
        and checking if input contains any parameter, it will create an instance of one or the other class.

        * :class:`MatrixS` is a subclass of :class:`sympy.Matrix` with slight API augmentations for compatibility with numpy
        * :class:`MatrixN` is a subclass of :class:`numpy.ndarray`

        Both classes have additional utility functions - while Matrix class is also presenting
        additional static utility functions
    """

    @staticmethod
    def __new__(cls, source, use_symbolic=None):
        """Constructor for Matrix class

        :param source: can be a string, a file, a list, a ndarray, another Matrix, or a integer
        :param use_symbolic: True to force use of symbolic, False to force use of numeric,
                             None to select based on source
        """
        if isinstance(source, str):
            source = Matrix._read(source.split("\n"))
        elif isinstance(source, io.TextIOWrapper):
            source = Matrix._read(source)
        elif isinstance(source, tuple):
            assert len(source) == 1 or len(source) == 2, "Matrix only supports 1/2D-Matrices"
            if len(source) == 1:
                source = np.ndarray((source[0], 1))
            else:
                source = np.ndarray(source)
        elif isinstance(source, sp.Matrix) or isinstance(source, np.ndarray):
            pass
        elif isinstance(source, list):
            if use_symbolic:
                source = sp.Matrix(source)
            else:
                source = np.asarray(source)
        elif isinstance(source, int):
            source = np.ndarray((source, source))
        else:
            raise NotImplementedError("no implemented input parser")
        if use_symbolic is None and isinstance(source, sp.Matrix) and len(source.free_symbols):
            use_symbolic = True
        if use_symbolic:
            return MatrixS(source)
        else:
            if isinstance(source, sp.Matrix):
                if len(source.free_symbols):
                    raise ValueError("cannot use MatrixN for matrix with parameters")
                source = np.array(source, dtype=complex)
            return MatrixN(source)

    @staticmethod
    def eye(n: int, use_symbolic: bool = False) -> Matrix:
        """Returns an identity matrix

        :param n: size of the matrix
        :param use_symbolic: defines if matrix will be symbolic or numeric
        :return: an identity matrix
        """
        if use_symbolic:
            return MatrixS(sp.eye(n))
        return MatrixN(np.eye(n, dtype=complex))

    @staticmethod
    def zeros(shape: tuple[int, int], use_symbolic: bool = False) -> Matrix:
        """Generate an empty matrix

        :param shape: 2D shape of the matrix
        :param use_symbolic: defines if matrix will be symbolic or numeric
        :return: an empty matrix
        """
        if use_symbolic:
            return MatrixS(sp.zeros(rows=shape[0], cols=shape[1]))
        return MatrixN(np.zeros(shape))

    def is_square(self) -> bool:
        return len(self.shape) == 2 and self.shape[0] == self.shape[1]

    @abstractmethod
    def is_unitary(self) -> bool:
        """check if matrix is unitary"""

    @abstractmethod
    def is_symbolic(self) -> bool:
        """check if matrix is symbolic or numeric"""

    @property
    @abstractmethod
    def defined(self):
        pass

    @abstractmethod
    def tonp(self):
        pass

    @abstractmethod
    def tosp(self):
        pass

    @staticmethod
    def random_unitary(n: int, parameters: Optional[Union[np.ndarray,list]] = None) -> MatrixN:
        r"""static method generating a random unitary matrix

        :param n: size of the Matrix
        :param parameters: :math:`2n^2` random parameters to use a generator
        :return: a numeric Matrix
        """
        if parameters is not None:
            assert len(parameters) == 2*n**2, "parameters has not the right size: should be %d, and is %d" % (
                2*n**2, len(parameters)
            )
            a = np.reshape(parameters[:n**2], (n, n))
            b = np.reshape(parameters[n**2:], (n, n))
            u = a + 1j * b
        else:
            u = np.random.randn(n, n) + 1j*np.random.randn(n, n)
        (q, r) = np.linalg.qr(u)
        r_diag = np.sign(np.diagonal(np.real(r)))
        n_u = np.zeros((n, n))
        np.fill_diagonal(n_u, val=r_diag)
        return Matrix(np.matmul(q, n_u))

    def simp(self):
        """Simplify the matrix - only implemented for symbolic matrix"""
        return self

    def pdisplay(self, precision: float = None, output_format: Literal["text", "mplot", "html", "latex"] = "text") -> str:
        """Generates representation of the matrix

        :param precision:
        :param output_format:
        :return:
        """
        def simp(v):
            if isinstance(v, complex) or isinstance(v, int) or isinstance(v, float) or\
               isinstance(v, sp.Number) or (isinstance(v, sp.Expr) and len(v.free_symbols) == 0):
                return simple_complex(complex(v))[1]
            else:
                return v.__repr__()
        if output_format != "text":
            marker = output_format == "html" and "$" or ""
            if isinstance(self, sp.Matrix):
                return marker+sp.latex(self)+marker
            rows = []
            for j in range(self.shape[0]):
                row = []
                for v in self[j, :]:
                    row.append(sp.S(simp(v)))
                rows.append(row)
            return marker+sp.latex(Matrix(rows, use_symbolic=True))+marker
        if self.shape[0] == 1:
            return (mlstr("[")+mlstr("  ").join([simp(v) for v in self[0, :]])+"]")._s
        else:
            s = mlstr("")
            for j in range(self.shape[1]):
                if j:
                    s += "  "
                s += "\n".join([simp(v) for v in self[:, j]])
            h = s.height
            left_bracket = "⎡\n"+"⎢\n"*(h-2)+"⎣"
            right_bracket = "⎤\n"+"⎥\n"*(h-2)+"⎦"
            return (mlstr(left_bracket)+s+right_bracket)._s

    def _read(seqline: Iterator[str]) -> Matrix:
        """read a matrix from file or a string sequence"""
        rows = []
        n = None
        for line in seqline:
            line = line.strip("\n ⎡⎤⎥⎢⎣⎦|[]")
            if not line or line.startswith("#"):
                continue
            row = [sp.S(s) for s in re.split(r"[\t ]+", line) if s]
            if n:
                if len(row) != n:
                    raise ValueError("invalid matrix")
            else:
                n = len(row)
            rows.append(row)

        return sp.Matrix(rows)

    @abstractmethod
    def fill(self, _: float):
        pass

    @abstractmethod
    def __getitem__(self, k):
        pass


class MatrixS(Matrix, sp.Matrix):

    def __new__(cls, obj):
        return sp.Matrix.__new__(cls, obj)

    def is_symbolic(self):
        return True

    @property
    def defined(self):
        return not self.free_symbols

    def tonp(self):
        return MatrixN(np.array(self, dtype=complex))

    def tosp(self):
        return self

    @property
    def T(self):
        """
            use numpy language
        """
        return self.transpose()

    def fill(self, f):
        return sp.Matrix.fill(self, f)

    def __getitem__(self, k):
        return sp.Matrix.__getitem__(self, k)

    def conj(self):
        """
            use numpy language
        """
        return self.conjugate()

    def simp(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i, j] = self[i, j].simplify()
        return self

    @property
    def ndim(self):
        return 2

    def is_unitary(self):
        """check if a matrix is squary and unitary"""
        if not self.is_square():
            return False
        if self.free_symbols:
            # use sympi only if we really have to...
            p_trans = self * self.T.conj()-sp.eye(self.shape[0])
            if p_trans.free_symbols:
                # cannot decide
                return None
            return np.allclose(np.array(p_trans).astype(complex), np.zeros(self.shape))
        else:
            return np.allclose(self.tonp().dot(self.tonp().T.conj()), np.eye(self.shape[0]))


class MatrixN(np.ndarray, Matrix):

    def __new__(cls, obj):
        array = super().__new__(cls, shape=obj.shape, dtype=complex)
        np.copyto(array, obj, casting='safe')
        return array

    @property
    def defined(self):
        return True

    def is_symbolic(self):
        return False

    def tonp(self):
        return self

    def tosp(self):
        return MatrixS(self)

    def fill(self, f):
        return np.ndarray.fill(self, f)

    def __getitem__(self, k):
        return np.ndarray.__getitem__(self, k)

    def is_unitary(self):
        """check if a matrix is square and unitary"""
        if not self.is_square():
            return False
        return np.allclose(self.dot(self.T.conj()), np.eye(self.shape[0]))

    def inv(self) -> MatrixN:
        """returns inverse of the Matrix

        :return:
        """
        return np.linalg.inv(self)
