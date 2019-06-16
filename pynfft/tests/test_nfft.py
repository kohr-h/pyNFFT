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
from numpy import pi

from pynfft import NFFT
from pynfft.util import random_unit_complex, random_unit_shifted, empty_aligned

# --- Test fixtures --- #


@pytest.fixture(
    scope="module", params=[6, 12], ids=[" m={} ".format(p) for p in [6, 12]]
)
def m(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[True, False],
    ids=[" use_dft={} ".format(p) for p in [True, False]],
)
def use_dft(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=["single", "double", "long double"],
    ids=[" prec={!r} ".format(p) for p in ["single", "double", "long double"]],
)
def prec(request):
    return request.param


N_params = [8, 9, (16,), [20, 1], (4, 5, 6)]
N_ids = [" N={!r} ".format(p) for p in N_params]


@pytest.fixture(scope="module", params=N_params, ids=N_ids)
def N(request):
    return request.param


M_params = [10, 16, 64]
M_ids = [" M={!r} ".format(p) for p in M_params]


@pytest.fixture(scope="module", params=M_params, ids=M_ids)
def M(request):
    return request.param


N_M_params = (
    (8, 8),
    (16, 16),
    (24, 24),
    (32, 32),
    (64, 64),
    ((8, 8), 8 * 8),
    ((16, 16), 16 * 16),
    ((24, 24), 24 * 24),
    ((32, 32), 32 * 32),
    ((64, 64), 64 * 64),
    ((8, 8, 8), 8 * 8 * 8),
    ((16, 16, 8), 8 * 8 * 8),
    ((16, 16, 16), 16 * 16 * 16),
)


@pytest.fixture(
    scope="module",
    params=N_M_params,
    ids=[" (N, M)={} ".format(arg) for arg in N_M_params],
)
def plan(request, m, prec):
    N, M = request.param

    if prec == "single" and m > 6:
        # Unstable
        pytest.skip("likely to produce NaN")

    pl = NFFT(N, M, prec=prec, m=m)
    pl.x = random_unit_shifted(pl.x.shape, pl.x.dtype)
    pl.precompute()
    return pl


# --- Helpers --- #


def fdft(x, f_hat):
    N = f_hat.shape
    d = x.shape[-1]
    k = np.mgrid[[slice(-Nt / 2, Nt / 2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = np.exp(-2j * pi * np.dot(x, k))
    f_dft = np.dot(F, f_hat.ravel())
    return f_dft


def rdft(x, f, N):
    d = x.shape[-1]
    k = np.mgrid[[slice(-Nt / 2, Nt / 2) for Nt in N]]
    k = k.reshape([d, -1])
    x = x.reshape([-1, d])
    F = np.exp(-2j * pi * np.dot(x, k))
    f_hat_dft = np.dot(np.conjugate(F).T, f)
    f_hat = f_hat_dft.reshape(N)
    return f_hat


# --- Tests --- #


def test_plan_arrays(N, M, prec):
    """Check whether the plan member arrays work as expected."""
    plan = NFFT(N, M, prec=prec)
    ndim = len(plan.N)
    dtype_c = plan.dtype
    dtype_r = np.dtype(dtype_c.char.lower())

    # Basic checks
    assert plan.f_hat.shape == plan.N
    assert plan.f_hat.dtype == dtype_c
    assert plan.f.shape == (plan.M,)
    assert plan.f.dtype == dtype_c
    assert plan.x.shape == (plan.M, ndim)
    assert plan.x.dtype == dtype_r

    # Assignment
    x1 = empty_aligned(plan.x.shape, plan.x.dtype)
    x1[:] = 0.1
    x2 = empty_aligned(plan.x.shape, plan.x.dtype)
    x2[:] = 0.2

    plan.x = x1
    assert plan.x is x1

    plan.x = x2
    assert plan.x is x2
    # Next 2 checks should be obvious from above, but anyhow
    x1[:] = 0
    assert np.all(plan.x == x2)
    plan.x[:] = -0.5
    assert np.all(x1 == 0)

    f = empty_aligned(plan.f.shape, plan.f.dtype)
    plan.f = f
    assert plan.f is f

    f_hat = empty_aligned(plan.f_hat.shape, plan.f_hat.dtype)
    plan.f_hat = f_hat
    assert plan.f_hat is f_hat


def test_forward(plan, use_dft):
    """Check forward transform against hand-written version."""
    # return  # TODO: re-enable
    rtol = 1e-3 if plan.dtype == "complex64" else 1e-7
    plan.f_hat = random_unit_complex(plan.f_hat.shape, plan.f_hat.dtype)
    plan.trafo(use_dft=use_dft)
    true_fdft = fdft(plan.x, plan.f_hat)
    assert np.allclose(plan.f, true_fdft, rtol=rtol)


def test_adjoint(plan, use_dft):
    """Check adjoint transform against hand-written version."""
    return  # TODO: re-enable
    rtol = 1e-3 if plan.dtype == "complex64" else 1e-7
    plan.f = random_unit_complex(plan.f.shape, plan.f.dtype)
    plan.adjoint(use_dft=use_dft)
    true_rdft = rdft(plan.x, plan.f, plan.N)
    assert np.allclose(plan.f_hat, true_rdft, rtol=rtol)
