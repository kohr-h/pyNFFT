// Copyright 2013-2019 PyNFFT developers and contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

//
// _nfft_impl.hpp -- implementation of the `_nfft` Python extension
//

#ifndef _NFFT_IMPL_HPP
#define _NFFT_IMPL_HPP

#define _unused(x) ((void)(x))  // to avoid warnings for assert-only variables

#include <cassert>
#include <iostream>  // debugging
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C" {
#include "cpu.h"
#include "fftw3.h"
#include "nfft3.h"
}

#include "_nfft_adaptor.hpp"
#include "_util.hpp"

namespace py = pybind11;

//
// class _NFFT
//
// A wrapper class around the C NFFT plan struct, exposing analogous properties.
//

template <typename FLOAT_T> class _NFFT {
public:
  _NFFT(py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags) {

    // Convert Py collections and perform sanity checks
    int d = py::len(N);
    std::vector<int> N_vec = shape_vec<int>(N);
    std::vector<int> n_vec = shape_vec<int>(n);
    assert(N_vec.size() == n_vec.size());
    size_t N_total = total_size<size_t>(N);
    // Remove `MALLOC_*` flags from `nfft_flags` to prevent NFFT from allocating
    // that memory automatically. We need to do that ourselves to be able to
    // feed the pointer into the refcounting machinery.
    // TODO: Make this user-controllable?
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));

    // Initialize C plan
    nfft_plan_init<FLOAT_T>(
        &plan_, d, N_vec.data(), M, n_vec.data(), m, nfft_flags, fftw_flags);

    // Allocate arrays with FFTW's aligned allocator and wrap them as Numpy
    // arrays.
    // NOTE: The general procedure is to first allocate memory using FFTW's
    // SIMD-aligning allocator, then wrap the pointer into a Python capsule for
    // lifetime management, and finally use that capsule as a base in the array
    // creation. That way, deallocation is done automatically at the correct
    // point in time.
    // The pointer is also assigned to the C NFFT plan struct.
    // The capsule is constructed using an array base pointer and a callback
    // function for deallocation of its memory. See
    // https://github.com/pybind/pybind11/issues/1042 for some discussion on the
    // usage of capsules.

    // `f_hat` has shape `N` and complex dtype
    plan_.f_hat = alloc_complex<FLOAT_T>(N_total);
    assert(plan_.f_hat != nullptr);
    py::capsule f_hat_caps(static_cast<void *>(plan_.f_hat), dealloc<FLOAT_T>);
    std::vector<ssize_t> f_hat_shape(N_vec.begin(), N_vec.end());
    f_hat_ = py::array_t<std::complex<FLOAT_T>>(
        f_hat_shape,
        reinterpret_cast<std::complex<FLOAT_T> *>(plan_.f_hat),
        f_hat_caps);

    // `f` has shape `(M,)` and complex dtype
    plan_.f = alloc_complex<FLOAT_T>(static_cast<size_t>(M));
    assert(plan_.raw_f != nullptr);
    py::capsule f_caps(static_cast<void *>(plan_.f), dealloc<FLOAT_T>);
    std::vector<ssize_t> f_shape{M};
    f_ = py::array_t<std::complex<FLOAT_T>>(
        f_shape, reinterpret_cast<std::complex<FLOAT_T> *>(plan_.f), f_caps);

    // `x` has shape `(M, d)` and real dtype
    plan_.x = alloc_real<FLOAT_T>(static_cast<size_t>(M * d));
    assert(plan_.x != nullptr);
    py::capsule x_caps(static_cast<void *>(plan_.x), dealloc<FLOAT_T>);
    std::vector<ssize_t> x_shape{M, d};
    x_ = py::array_t<FLOAT_T>(x_shape, plan_.x, x_caps);
  }

  ~_NFFT() { nfft_plan_finalize<FLOAT_T>(&plan_); }

  // Python interface relevant functions

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const { return f_hat_; }
  void f_hat_setter(py::array_t<std::complex<FLOAT_T>> &f_hat_new) {
    f_hat_ = f_hat_new;
  }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const { return f_; }
  void f_setter(py::array_t<std::complex<FLOAT_T>> &f_new) { f_ = f_new; }

  // Property x
  py::array_t<FLOAT_T> x() const { return x_; }
  void x_setter(py::array_t<FLOAT_T> &x_new) { x_ = x_new; }

  // Precomputation
  void precompute() { nfft_precompute_one_psi_impl<FLOAT_T>(&plan_); }

  // Forward trafo
  void trafo(bool use_dft) {
    if (use_dft) {
      nfft_trafo_direct_impl<FLOAT_T>(&plan_);
    } else {
      nfft_trafo_impl<FLOAT_T>(&plan_);
    }
  }

  // Adjoint trafo
  void adjoint(bool use_dft) {
    if (use_dft) {
      nfft_adjoint_direct_impl<FLOAT_T>(&plan_);
    } else {
      nfft_adjoint_impl<FLOAT_T>(&plan_);
    }
  }

  // Members
  nfft_plan_t<FLOAT_T> plan_;
  py::array_t<std::complex<FLOAT_T>> f_hat_;
  py::array_t<std::complex<FLOAT_T>> f_;
  py::array_t<FLOAT_T> x_;
};

//
// Array creation with SIMD-aligned memory
//

template <typename FLOAT_T>
py::array_t<FLOAT_T> _empty_aligned_real(py::tuple &shape) {
  // Convert shape & calculate number of elements
  std::vector<ssize_t> shp_vec = shape_vec<ssize_t>(shape);
  size_t num_el = total_size<size_t>(shape);

  // Allocate SIMD-aligned memory
  FLOAT_T *raw_arr = alloc_real<FLOAT_T>(num_el);
  assert(raw_arr != nullptr);

  // Wrap into Python capsule and construct wrapping Numpy array
  py::capsule arr_caps(static_cast<void *>(raw_arr), dealloc<FLOAT_T>);
  py::array_t<FLOAT_T> arr = py::array_t<FLOAT_T>(shp_vec, raw_arr, arr_caps);

  return arr;
}

template <typename FLOAT_T>
py::array_t<std::complex<FLOAT_T>> _empty_aligned_complex(py::tuple &shape) {
  // Convert shape & calculate number of elements
  std::vector<ssize_t> shp_vec = shape_vec<ssize_t>(shape);
  size_t num_el = total_size<size_t>(shape);

  // Allocate SIMD-aligned memory
  fftw_complex_t<FLOAT_T> *raw_arr = alloc_complex<FLOAT_T>(num_el);
  assert(raw_arr != nullptr);

  // Wrap into Python capsule and construct wrapping Numpy array
  py::capsule arr_caps(static_cast<void *>(raw_arr), dealloc<FLOAT_T>);
  py::array_t<std::complex<FLOAT_T>> arr = py::array_t<std::complex<FLOAT_T>>(
      shp_vec, reinterpret_cast<std::complex<FLOAT_T> *>(raw_arr), arr_caps);

  return arr;
}

//
// Module-level startup and teardown functions
//

template <typename FLOAT_T> void _nfft_atentry();
template <> void _nfft_atentry<float>() { fftwf_init_threads(); }
template <> void _nfft_atentry<double>() { fftw_init_threads(); }
template <> void _nfft_atentry<long double>() { fftwl_init_threads(); }

template <typename FLOAT_T> void _nfft_atexit();

template <> void _nfft_atexit<float>() {
  fftwf_cleanup();
  fftwf_cleanup_threads();
}

template <> void _nfft_atexit<double>() {
  fftw_cleanup();
  fftw_cleanup_threads();
}

template <> void _nfft_atexit<long double>() {
  fftwl_cleanup();
  fftwl_cleanup_threads();
}

#endif  // _NFFT_IMPL_HPP
