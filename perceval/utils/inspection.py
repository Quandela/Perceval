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

from inspect import signature
from typing import Callable


def has_kwargs(func: Callable):
    """Check if a function can be called with any number of keyword arguments (i.e. has **kwargs)"""
    sig = signature(func)

    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True

    return False


def has_arguments(func: Callable):
    """Check if a function can be called with at least one non-named argument (i.e. signature is not empty or only **kwargs)"""
    sig = signature(func)

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:  # *args
            return True
        elif param.kind == param.VAR_KEYWORD:  # **kwargs
            continue
        else:
            return True

    return False


def parse_signature(func: Callable) -> tuple[list[tuple[str, type | None, bool]], type | None]:
    """
    Returns the signature of the given function as a list of (name, type, is_positional_arg) tuples,
    and the type of the expected returned value.
    For types, if several values are given, or none is given, None is returned instead.

    *args and **kwargs are ignored.
    """

    sig = signature(func)

    res = []

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            continue

        # First, we get the class. If the class is not given, or several classes are given, we don't use one
        cls = param.annotation
        if cls is param.empty or not isinstance(cls, type):
            cls = None

        # Now we get if there is a default argument, in which case this argument is considered as optional
        positional = param.default is param.empty

        res.append((param.name, cls, positional))

    cls = sig.return_annotation
    if cls is sig.empty or not isinstance(cls, type):
        cls = None

    return res, cls
