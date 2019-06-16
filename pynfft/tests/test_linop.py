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
from scipy.sparse.linalg import lsqr

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
    print("prec:", prec)
    plan = NFFT((64, 64), 1024, prec=prec)
    print("plan init")
    plan.precompute()
    print("plan precompute")
    plan.x = random_unit_shifted(plan.x.shape, plan.x.dtype)
    print("plan.x set")
    F = as_linop(plan)
    print("conv to linop")

    # Basics
    assert F.shape == (1024, 128 * 128)

    # Forward: takes 1D vector of length `N_total`, returns 1D vector of length `M`
    plan.f_hat[:] = random_unit_complex(plan.f_hat.shape, plan.f_hat.dtype)
    print("f_hat mutated")
    plan.trafo()
    print("trafo() called")
    assert False
    f = plan.f.copy()
    print("f computed")
    assert False

    v = plan.f_hat.ravel(order="C").copy()
    u = F * v
    assert Fv.shape == (M,)
    assert np.all(Fv == f)


