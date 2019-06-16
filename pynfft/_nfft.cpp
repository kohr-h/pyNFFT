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
// _nfft.cpp -- definition of the `_nfft` Python extension module
//

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include "_nfft_impl.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// NB: The startup functions can just be run as part of the module creation code
// (PYBIND11_MODULE), whereas the teardown functions need to be registered as
// callbacks somewhere. We use the approach with the Python `atexit` module, see
// https://pybind11.readthedocs.io/en/master/advanced/misc.html#module-destructors
// for details.

PYBIND11_MODULE(_nfft, m) {
  // Module docstring
  m.doc() = "Wrapper module for C NFFT plans and associated functions.";

  // Import `atexit` to register exit hooks for cleanup
  auto atexit = py::module::import("atexit");

  // _NFFT class definitions + entry and exit code for each float type
  py::class_<_NFFT<float>>(m, "_NFFTFloat")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property("f_hat", &_NFFT<float>::f_hat, &_NFFT<float>::f_hat_setter)
      .def_property("f", &_NFFT<float>::f, &_NFFT<float>::f_setter)
      .def_property("x", &_NFFT<float>::x, &_NFFT<float>::x_setter)
      .def("precompute", &_NFFT<float>::precompute)
      .def("trafo", &_NFFT<float>::trafo)
      .def("adjoint", &_NFFT<float>::adjoint);
  _nfft_atentry<float>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<float>));

  py::class_<_NFFT<double>>(m, "_NFFTDouble")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property(
          "f_hat", &_NFFT<double>::f_hat, &_NFFT<double>::f_hat_setter)
      .def_property("f", &_NFFT<double>::f, &_NFFT<double>::f_setter)
      .def_property("x", &_NFFT<double>::x, &_NFFT<double>::x_setter)
      .def("precompute", &_NFFT<double>::precompute)
      .def("trafo", &_NFFT<double>::trafo)
      .def("adjoint", &_NFFT<double>::adjoint);
  _nfft_atentry<double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<double>));

  py::class_<_NFFT<long double>>(m, "_NFFTLongDouble")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property("f_hat",
                    &_NFFT<long double>::f_hat,
                    &_NFFT<long double>::f_hat_setter)
      .def_property("f", &_NFFT<long double>::f, &_NFFT<long double>::f_setter)
      .def_property("x", &_NFFT<long double>::x, &_NFFT<long double>::x_setter)
      .def("precompute", &_NFFT<long double>::precompute)
      .def("trafo", &_NFFT<long double>::trafo)
      .def("adjoint", &_NFFT<long double>::adjoint);
  _nfft_atentry<long double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<long double>));

  // Module-level functions; to keep the module namespace somewhat more
  // tidy, we stick the dtype-specific functions into dictionaries for the
  // Python module to fetch
  m.attr("_empty_aligned_impl") = py::dict(
      "float32"_a = py::cpp_function(_empty_aligned_real<float>),
      "float64"_a = py::cpp_function(_empty_aligned_real<double>),
      "float128"_a = py::cpp_function(_empty_aligned_real<long double>),
      "complex64"_a = py::cpp_function(_empty_aligned_complex<float>),
      "complex128"_a = py::cpp_function(_empty_aligned_complex<double>),
      "complex256"_a = py::cpp_function(_empty_aligned_complex<long double>));
  m.attr("_simd_alignment") = py::cpp_function(simd_alignment);
}
