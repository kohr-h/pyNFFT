# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 PyNFFT developers and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy as np
import pytest

from pynfft.util import empty_aligned, simd_alignment, simd_align_offset

# --- Test fixtures --- #


dtype_params = ["float32", "float64", "float128", "complex64", "complex128", "complex256"]
dtype_ids = [" dtype={!r} ".format(p) for p in dtype_params]
@pytest.fixture(scope="module", params=dtype_params, ids=dtype_ids)
def dtype(request):
    return request.param


shape_params = [1, 9, (8,), [4, 1], (4, 5, 6)]
shape_ids = [" shape={!r} ".format(p) for p in shape_params]
@pytest.fixture(scope="module", params=shape_params, ids=shape_ids)
def shape(request):
    return request.param


# --- Tests --- #


def test_simd_alignment(dtype):
    """Check if ``simd_alignment`` produces reasonable output.

    Output depends on the host machine, thus we only check that the
    function runs and gives a reasonable answer.
    """
    assert simd_alignment(dtype) in (4, 16, 32)


def test_empty_aligned(shape, dtype):
    """Check if ``empty_aligned`` produces a properly shaped aligned array."""
    arr = empty_aligned(shape, dtype)
    arr_npy = np.empty(shape, dtype, order="C")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == arr_npy.shape
    assert arr.dtype == arr_npy.dtype

    # For technical reasons, `OWNDATA` is currently False
    for flag in ["C_CONTIGUOUS", "WRITEABLE", "ALIGNED"]:
        assert getattr(arr.flags, flag.lower())

    print(arr.ctypes.data)
    assert simd_align_offset(arr) == 0

