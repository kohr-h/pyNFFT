#include "_nfft_impl.hpp"
#include "_util.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_nfft, m) {
  m.doc() = "Wrapper module for C NFFT plans and associated functions.";
  auto atexit = py::module::import("atexit");

  py::class_<_NFFT<float>>(m, "_NFFTFloat")
      .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<float>::f_hat)
      .def_property_readonly("f", &_NFFT<float>::f)
      .def_property_readonly("x", &_NFFT<float>::x);
  _nfft_atentry<float>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<float>));

  py::class_<_NFFT<double>>(m, "_NFFTDouble")
      .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<double>::f_hat)
      .def_property_readonly("f", &_NFFT<double>::f)
      .def_property_readonly("x", &_NFFT<double>::x);
  _nfft_atentry<double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<double>));

  py::class_<_NFFT<long double>>(m, "_NFFTLongDouble")
      .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<long double>::f_hat)
      .def_property_readonly("f", &_NFFT<long double>::f)
      .def_property_readonly("x", &_NFFT<long double>::x);
  _nfft_atentry<long double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<long double>));
}
