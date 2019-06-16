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
import scipy.sparse
import scipy.signal

from pynfft import NFFT
from pynfft.linop import as_linop
from pynfft.util import random_unit_complex, random_unit_shifted, empty_aligned

# --- Test fixtures ---

@pytest.fixture(
    scope="module",
    params=["single", "double", "long double"],
    ids=[" prec={!r} ".format(p) for p in ["single", "double", "long double"]],
)
def prec(request):
    return request.param


# --- Tests ---


def test_as_linop(prec):
    """Check construction of ``LinearOperator`` from an NFFT plan."""
    plan = NFFT((64, 64), 64 ** 2, m=6, prec=prec)
    plan.precompute()
    plan.x = random_unit_shifted(plan.x.shape, plan.x.dtype)
    F = as_linop(plan)

    # Basics
    assert F.shape == (64 ** 2, 64 * 64)

    # Forward: takes 1D vector of length `N_total`, returns 1D vector of length `M`
    plan.f_hat[:] = random_unit_complex(plan.f_hat.shape, plan.f_hat.dtype)
    plan.trafo()
    f = plan.f.copy()

    v = plan.f_hat.ravel(order="C").copy()
    u = F.matvec(v)
    assert u.shape == (plan.M,)
    assert np.allclose(u, f)

    # Adjoint: takes 1D vector of length `M`, returns 1D vector of length `N_total`
    plan.f[:] = random_unit_complex(plan.f.shape, plan.f.dtype)
    plan.adjoint()
    f_hat = plan.f_hat.copy()

    u = plan.f.copy()
    v = F.rmatvec(u)
    assert v.shape == (plan.N_total,)
    assert np.allclose(v, f_hat.ravel())


def test_solver():
    """Test inverting an NFFT with a SciPy solver."""
    # Create plan; with a slightly thinned regular grid the result should be okay
    N = (64, 64)
    npts = 84
    M = npts ** 2
    x = empty_aligned((M, 2), dtype=float)
    x_1d = np.linspace(-0.5, 0.5, npts, endpoint=False)
    x[:, 0] = np.repeat(x_1d, npts)
    x[:, 1] = np.tile(x_1d, npts)
    plan = NFFT(N, M)
    plan.precompute()
    plan.x = x

    # Make operator, generate true solution & data
    A = as_linop(plan)
    sol = np.random.normal(size=N)
    sol = scipy.signal.convolve(sol, np.ones((3, 3)), mode="same")  # dampen high frequencies
    sol = sol.ravel(order="C")
    b = A.matvec(sol)
    print("NaNs in b:", np.any(np.isnan(b)))

    # Very coarse norm estimation
    damp = 0.01 * np.linalg.norm(A.rmatvec(b)) / np.linalg.norm(sol)
    print(damp)
    res = scipy.sparse.linalg.lsqr(A, b, damp=damp, iter_lim=100)
    print(res)
    print(np.linalg.norm(res[0] - sol) / np.linalg.norm(sol))
    assert False
