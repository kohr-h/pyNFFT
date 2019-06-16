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

import numpy as np

from ._nfft import _simd_alignment, _empty_aligned_impl

# fmt: off
__all__ = (
    "empty_aligned",
    "simd_align_offset",
    "simd_alignment",
    "random_unit_complex",
    "random_unit_shifted",
)
#fmt: on

_VALID_DTYPES = (
        "float32",
        "float64",
        "float128",
        "complex64",
        "complex128",
        "complex256",
    )


def empty_aligned(shape, dtype):
    """Return an SIMD-aligned empty array.

    This function does the same as ``numpy.empty(shape, dtype, order="C")``,
    but additionally ensures that the allocated memory is aligned for optimal
    SIMD instruction performance. It should be preferred over regular array
    creation for use with NFFT.

    :param shape: Number of entries per axis.
    :type shape: int or sequence of int

    :param dtype: Specifier for the type of each entry in the array, as
    understood by the ``numpy.dtype`` constructor. Must be one of the
    following types::

             "float32", "float64", "float128",
             "complex64", "complex128", "complex256"

    :type dtype: data-type

    :returns: A new SIMD-aligned array.
    :rtype: numpy.ndarray
    """
    shape_in = shape
    try:
        shape = (int(shape),)
    except TypeError:
        shape = tuple(int(n) for n in shape)
    if any(n <= 0 for n in shape):
        raise ValueError("`shape` must be all positive, got {}".format(shape_in))

    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype not in _VALID_DTYPES:
        raise ValueError("unsupported `dtype` {!r}".format(dtype_in))

    _empty_aligned = _empty_aligned_impl[dtype.name]
    return _empty_aligned(shape)


def simd_align_offset(arr):
    """Return offset relative to SIMD alignment byte boundaries.

    The return value is 0 for an SIMD-aligned array, as returned
    by :func:`empty_aligned`. The alignment in bytes can be retrieved
    with :func:`simd_alignment`.

    :param arr: Array whose alignment should be checked.
    :type arr: numpy.ndarray

    :returns: Alignment offset of ``arr``.
    :rtype: int
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("`arr` must be a numpy.ndarray, got {!r}".format(arr))
    align = simd_alignment(arr.dtype)
    return arr.ctypes.data % align


def simd_alignment(dtype):
    """Return the SIMD alignment for a dtype in number of bytes.

    The alignment depends on the CPU capabilities, with larger alignment
    factors for wider vector instructions.
    See the implementation in the `cpu.h header
    <https://github.com/pynfft/pynfft/blob/master/pynfft/cpu.h>`_
    for details.

    Note that certain SIMD instructions are not supported for extended
    precision types.
    """
    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype not in _VALID_DTYPES:
        raise ValueError("unsupported `dtype` {!r}".format(dtype_in))
    align = _simd_alignment()
    if dtype in ("float128", "complex256"):
        align = min(16, align)
    return align


def random_unit_complex(shape, dtype):
    """Return an array of random complex numbers in ``[0, 1) + [0, 1) * i``.

    Used for testing :attr:`pynfft.NFFT.f` and
    :attr:`pynfft.NFFT.f_hat`.

    :param shape: Number of entries per axis.
    :type shape: int or sequence of int

    :param dtype: Complex data type specifier.
    :type dtype: data-type

    :returns: A new complex array filled with random noise in
        ``[0, 1) + [0, 1) * i``.
    :rtype: numpy.ndarray
    """
    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype.kind != "c":
        raise ValueError(
            "`dtype` must be a complex data type, got {!r}".format(dtype_in)
        )

    arr = empty_aligned(shape, dtype)
    arr.real[:] = np.random.uniform(0, 1, size=shape)
    arr.imag[:] = np.random.uniform(0, 1, size=shape)
    return arr


def random_unit_shifted(shape, dtype):
    """Return a vector of random real numbers in ``[-0.5, 0.5)``.

    Used for testing :attr:`pynfft.NFFT.x`.

    :param shape: Number of entries per axis.
    :type shape: int or sequence of int

    :param dtype: Real floating-point data type.
    :type dtype: data-type

    :returns: A new array filled with random noise in ``[-0.5, 0.5)``.
    :rtype: numpy.ndarray
    """
    dtype, dtype_in = np.dtype(dtype), dtype
    if dtype.kind != "f":
        raise ValueError(
            "`dtype` must be a real floating-point data type, got {!r}"
            "".format(dtype_in)
        )
    arr = empty_aligned(shape, dtype)
    arr[:] = np.random.uniform(-0.5, 0.5, size=shape)
    return arr
