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
from copy import copy
from perceval.utils import BasicState, PostSelect

import pytest


def test_postselect_init():
    ps_empty = PostSelect()  # Should work
    ps1 = PostSelect("([0]==0 & [1,2]==1 & [3,4]==1) | [5]==0")
    ps2 = PostSelect("([     0 ]   ==0 and [1 , 2]== 1 & [3,4] == 1)OR[5]==0")

    assert ps2 == ps1


def test_postselect_init_invalid():
    with pytest.raises(RuntimeError):
        PostSelect("[0]==0 & [1,2,]==1 & [3,4]==1 & [5]==0")  # Too many commas

    with pytest.raises(RuntimeError):
        PostSelect("[0]==0 & [1,,,,2]==1 & [3,4]==1 & [5]==0")  # Too many commas

    with pytest.raises(RuntimeError):
        PostSelect("[0]==0 & (1,2)==1 & [3,4]==1 & [5]==0")  # Tuple syntax is not supported

    with pytest.raises(RuntimeError):
        PostSelect("[2] < 4 + [1] > 2")  # Invalid separator

    with pytest.raises(RuntimeError):
        PostSelect("[2] == 4 | [1] > 2 & [0] < 1")  # Invalid use of different separators

    with pytest.raises(RuntimeError):
        PostSelect("[2] >> 4")  # Invalid operator

    with pytest.raises(RuntimeError):
        PostSelect("[0]==0 & [1,2]]==1 & [3,4]==1 & [5]==0")  # Too many brackets => Invalid operator


def test_postselect_usage():
    ps_empty = PostSelect()
    for bs in [BasicState(), BasicState([0, 1]), BasicState([8, 8, 8, 8, 8, 8, 8, 8, 8])]:
        assert ps_empty(bs)

    ps_cnot = PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0")
    for bs in [BasicState([0, 1, 0, 1, 0, 0]), BasicState([0, 0, 1, 0, 1, 0]), BasicState([0, 1, 0, 0, 1, 0]),
               BasicState([0, 0, 1, 1, 0, 0])]:
        assert ps_cnot(bs)
    for bs in [BasicState([1, 1, 0, 1, 0, 0]), BasicState([0, 0, 1, 0, 1, 1]), BasicState([0, 1, 1, 0, 1, 0]),
               BasicState([0, 0, 1, 2, 0, 0]), BasicState([0, 0, 1, 0, 0, 0])]:
        assert not ps_cnot(bs)
    assert ps_cnot(BasicState("|0,{_:0},0,{_:1},0,0>"))


def test_postselect_usage_advanced_ge_le():
    ps1 = PostSelect("[1,2]>=1")
    ps2 = PostSelect("[1,2]>0")
    ps3 = PostSelect("[1,2]<1")
    ps4 = PostSelect("[1,2]<=0")
    for bs in [BasicState([0, 1, 0, 1, 0, 0]), BasicState([0, 0, 1, 0, 1, 0]), BasicState([0, 1, 0, 0, 1, 0]),
               BasicState([0, 0, 1, 1, 0, 0])]:
        assert ps1(bs)
        assert ps2(bs)
        assert not ps3(bs)
        assert not ps4(bs)

def test_postselect_operators():
    ps = PostSelect("([1,2]>=1 & [3] < 2) | ([4] == 1 xor [5] > 0)")
    assert ps(BasicState([0, 1, 0, 1, 0, 0]))  # and True
    assert not ps(BasicState([0, 1, 0, 2, 0, 0]))
    assert ps(BasicState([0, 0, 0, 0, 1, 0]))
    assert not ps(BasicState([0, 0, 0, 0, 1, 1]))   # xor False
    assert ps(BasicState([0, 0, 0, 0, 1, 0]))   # xor True
    assert ps(BasicState([0, 0, 0, 0, 0, 1]))   # xor True
    assert not ps(BasicState([0, 0, 0, 0, 2, 0]))   # xor False


def test_postselect_str():
    ps1 = PostSelect("[0]==0 & [1, 2 ]>0 & [3, 4]==1 & [5]<1")
    ps2 = PostSelect(str(ps1))
    assert ps1 == ps2


def test_postselect_apply_permutation():
    ps = PostSelect("[0,1]==1 & [2,3]>2 & [4,5]<3")

    perm_vector = [1, 2, 0]  # Corresponds to PERM(perm_vector)
    ps.apply_permutation(perm_vector)
    assert ps == PostSelect("[1,2]==1 & [0,3]>2 & [4,5]<3")

    ps = PostSelect("[0,1]==1 & [2,3]>2 & [4,5]<3")
    ps.apply_permutation(perm_vector, 1)
    assert ps == PostSelect("[0,2]==1 & [3,1]>2 & [4,5]<3")



def test_postselect_shift_modes():
    initial_ps = PostSelect("[0,1]==1 & [2,3]>2 & [4,5]<3")
    ps = copy(initial_ps)
    ps.shift_modes(0)
    assert ps == initial_ps

    ps.shift_modes(2)
    assert ps == PostSelect("[2,3]==1 & [4,5]>2 & [6,7]<3")

    ps.shift_modes(-2)
    assert ps == initial_ps

    with pytest.raises(RuntimeError):
        ps.shift_modes(-1)


def test_postselect_can_compose_with():
    ps = PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0")
    assert ps.can_compose_with([0])
    assert ps.can_compose_with([1, 2])
    assert ps.can_compose_with([3, 4])
    assert ps.can_compose_with([5])

    # Special cases
    assert ps.can_compose_with([1])  # You can compose with a subset of a condition (here [1,2]==1)
    assert ps.can_compose_with([6, 7])  # You can compose if the modes are not even in the post-selection conditions

    assert not ps.can_compose_with([2, 3])
    assert not ps.can_compose_with([5, 6])

    ps = PostSelect("[0]>0 & [1,2,3]<1 & [2,3,4]<1 & [5]>0")
    assert ps.can_compose_with([2])
    assert ps.can_compose_with([2, 3])
    assert not ps.can_compose_with([2, 3, 4])
    assert not ps.can_compose_with([3, 4])


def test_postselect_merge():
    ps1 = PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0")
    ps2 = PostSelect("[5,6,7] == 1")
    ps1.merge(ps2)
    assert ps1 == PostSelect("[0]==0 & [1,2]==1 & [3,4]==1 & [5]==0 & [5,6,7] == 1")

    ps_empty = PostSelect()
    ps_empty.merge(ps1)
    assert ps_empty == ps1


def test_postselect_independent():
    ps1 = PostSelect("[0]==0 & [1,2]==1")
    assert not ps1.is_independent_with(ps1)
    ps2 = PostSelect("[2,3] == 1")
    assert not ps1.is_independent_with(ps2)
    ps3 = PostSelect("[3]<1 & [4]<1 & [5]>0")
    assert ps1.is_independent_with(ps3)
