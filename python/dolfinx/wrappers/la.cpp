// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>

namespace py = pybind11;

namespace
{
// Declare objects that have multiple scalar types
template <typename T>
void declare_objects(py::module& m, const std::string& type)
{
  // dolfinx::la::Vector
  std::string pyclass_vector_name = std::string("Vector_") + type;
  py::class_<dolfinx::la::Vector<T>, std::shared_ptr<dolfinx::la::Vector<T>>>(
      m, pyclass_vector_name.c_str())
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::common::IndexMap>& map,
             int bs) { return dolfinx::la::Vector<T>(map, bs); }))
      .def("set", &dolfinx::la::Vector<T>::set)
      .def("norm", &dolfinx::la::Vector<T>::norm,
           py::arg("type") = dolfinx::la::Norm::l2)
      .def_property_readonly("map", &dolfinx::la::Vector<T>::map)
      .def_property_readonly("bs", &dolfinx::la::Vector<T>::bs)
      .def_property_readonly("array",
                             [](dolfinx::la::Vector<T>& self)
                             {
                               xtl::span<T> array = self.mutable_array();
                               return py::array_t<T>(array.size(), array.data(),
                                                     py::cast(self));
                             })
      .def("scatter_forward", &dolfinx::la::Vector<T>::scatter_fwd)
      .def("scatter_reverse", &dolfinx::la::Vector<T>::scatter_rev);
}

void petsc_module(py::module& m)
{
  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::petsc::create_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_vector_wrap",
      [](dolfinx::la::Vector<PetscScalar, std::allocator<PetscScalar>>& x)
      { return dolfinx::la::petsc::create_vector_wrap(x); },
      py::return_value_policy::take_ownership,
      "Create a ghosted PETSc Vec that wraps a DOLFINx Vector");
  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p, const std::string& type)
      { return dolfinx::la::petsc::create_matrix(comm.get(), p, type); },
      py::return_value_policy::take_ownership, py::arg("comm"), py::arg("p"),
      py::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");

  // TODO: check reference counting for index sets
  m.def("create_index_sets", &dolfinx::la::petsc::create_index_sets,
        py::return_value_policy::take_ownership);

  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x_b,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<xtl::span<const PetscScalar>> _x_b;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, maps);
      },
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");
  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::petsc::get_local_vectors(x, maps);
        std::vector<py::array> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(dolfinx_wrappers::as_pyarray(std::move(v)));
        return ret;
      },
      "Gather an (ordered) list of sub vectors from a block vector.");
}

} // namespace

namespace dolfinx_wrappers
{
void la(py::module& m)
{
  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific linear algebra");
  petsc_module(petsc_mod);

  // dolfinx::la::SparsityPattern
  py::class_<dolfinx::la::SparsityPattern,
             std::shared_ptr<dolfinx::la::SparsityPattern>>(m,
                                                            "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs)
          { return dolfinx::la::SparsityPattern(comm.get(), maps, bs); }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             const std::array<std::vector<int>, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), patterns, maps, bs);
          }))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map)
      .def("assemble", &dolfinx::la::SparsityPattern::assemble)
      .def("num_nonzeros", &dolfinx::la::SparsityPattern::num_nonzeros)
      .def("insert",
           [](dolfinx::la::SparsityPattern& self,
              const py::array_t<std::int32_t, py::array::c_style>& rows,
              const py::array_t<std::int32_t, py::array::c_style>& cols)
           {
             self.insert(xtl::span(rows.data(), rows.size()),
                         xtl::span(cols.data(), cols.size()));
           })
      .def("insert_diagonal", &dolfinx::la::SparsityPattern::insert_diagonal)
      .def_property_readonly("diagonal_pattern",
                             &dolfinx::la::SparsityPattern::diagonal_pattern,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "off_diagonal_pattern",
          &dolfinx::la::SparsityPattern::off_diagonal_pattern,
          py::return_value_policy::reference_internal);

  py::enum_<dolfinx::la::Norm>(m, "Norm")
      .value("l1", dolfinx::la::Norm::l1)
      .value("l2", dolfinx::la::Norm::l2)
      .value("linf", dolfinx::la::Norm::linf)
      .value("frobenius", dolfinx::la::Norm::frobenius);

  // Declare objects that are templated over type
  declare_objects<double>(m, "float64");
  declare_objects<float>(m, "float32");
  declare_objects<std::complex<double>>(m, "complex128");
  declare_objects<std::complex<float>>(m, "complex64");
}
} // namespace dolfinx_wrappers
