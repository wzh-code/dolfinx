// Copyright (C) 2020-2021 Garth N. Wells, Igor A. Baratta
// and Jørgen S.Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include <functional>
#include <numeric>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::fem
{
template <typename T>
class Function;
template <typename T>
class Expression;

namespace impl
{
/// Apply interpolation operator Pi to data to evaluate the dos
/// coefficients
/// @param[in] Pi The interpolation matrix (shape = (num dofs,
/// num_points * value_size))
/// @param[in] data Function evaluations, by point, e.g. (f0(x0),
/// f1(x0), f0(x1), f1(x1), ...)
/// @param[out] coeffs The degrees of freedom to compute
/// @param[in] bs The block size
template <typename U, typename V, typename T>
void interpolation_apply(const U& Pi, const V& data, std::vector<T>& coeffs,
                         int bs)
{
  // Compute coefficients = Pi * x (matrix-vector multiply)
  if (bs == 1)
  {
    assert(data.shape(0) * data.shape(1) == Pi.shape(1));
    for (std::size_t i = 0; i < Pi.shape(0); ++i)
    {
      coeffs[i] = 0.0;
      for (std::size_t k = 0; k < data.shape(1); ++k)
        for (std::size_t j = 0; j < data.shape(0); ++j)
          coeffs[i] += Pi(i, k * data.shape(0) + j) * data(j, k);
    }
  }
  else
  {
    const std::size_t cols = Pi.shape(1);
    assert(data.shape(0) == Pi.shape(1));
    assert(data.shape(1) == bs);
    for (int k = 0; k < bs; ++k)
    {
      for (std::size_t i = 0; i < Pi.shape(0); ++i)
      {
        T acc = 0;
        for (std::size_t j = 0; j < cols; ++j)
          acc += Pi(i, j) * data(j, k);
        coeffs[bs * i + k] = acc;
      }
    }
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh. The function is for cases where the finite element basis
/// functions are mapped in the same way, e.g. both use the same Piola
/// map.
/// @param[out] u1 The function to interpolate to
/// @param[in] u0 The function to interpolate from
/// @pre The functions `u1` and `u0` must share the same mesh and the
/// elements must share the same basis function map. Neither is checked
/// by the function.
template <typename T>
void interpolate_same_map(Function<T>& u1, const Function<T>& u0)
{
  assert(u0.function_space());
  auto mesh = u0.function_space()->mesh();
  assert(mesh);

  std::shared_ptr<const FiniteElement> element1
      = u1.function_space()->element();
  assert(element1);
  std::shared_ptr<const FiniteElement> element0
      = u0.function_space()->element();
  assert(element0);

  const int tdim = mesh->topology().dim();
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  xtl::span<T> u1_array = u1.x()->mutable_array();
  xtl::span<const T> u0_array = u0.x()->array();

  xtl::span<const std::uint32_t> cell_info;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap1 = u1.function_space()->dofmap();
  auto dofmap0 = u0.function_space()->dofmap();

  // Create interpolation operator
  const xt::xtensor<double, 2> i_m
      = element1->create_interpolation_operator(*element0);

  // Get block sizes and dof transformation operators
  const int bs1 = element1->block_size();
  const int bs0 = element0->block_size();
  const auto apply_dof_transformation
      = element0->get_dof_transformation_function<T>(false, true, false);
  const auto apply_inverse_dof_transform
      = element1->get_dof_transformation_function<T>(true, true, false);

  // Creat working array
  std::vector<T> local0(element0->space_dimension());
  std::vector<T> local1(element1->space_dimension());

  // Iterate over mesh and interpolate on each cell
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    xtl::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
    for (std::size_t i = 0; i < dofs0.size(); i++)
      for (int k = 0; k < bs0; ++k)
        local0[bs0 * i + k] = u0_array[bs0 * dofs0[i] + k];

    apply_dof_transformation(local0, cell_info, c, 1);

    // FIXME: Get compile-time ranges from Basix
    // Apply interpolation operator
    std::fill(local1.begin(), local1.end(), 0);
    for (std::size_t i = 0; i < i_m.shape(0); ++i)
      for (std::size_t j = 0; j < i_m.shape(1); ++j)
        local1[i] += i_m(i, j) * local0[j];

    apply_inverse_dof_transform(local1, cell_info, c, 1);

    xtl::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
    for (std::size_t i = 0; i < dofs1.size(); ++i)
      for (int k = 0; k < bs1; ++k)
        u1_array[bs1 * dofs1[i] + k] = local1[bs1 * i + k];
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh. The function is for cases where the finite element basis
/// functions for the two elements are mapped differently, e.g. one may
/// be Piola mapped and the other with a standard isoparametric map.
/// @param[out] u1 The function to interpolate to
/// @param[in] u0 The function to interpolate from
/// @pre The functions `u1` and `u0` must share the same mesh. This is
/// not checked by the function.
template <typename T>
void interpolate_nonmatching_maps(Function<T>& u1, const Function<T>& u0)
{
  // Get mesh
  assert(u0.function_space());
  auto mesh = u0.function_space()->mesh();
  assert(mesh);

  // Mesh dims
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();

  // Get elements
  std::shared_ptr<const FiniteElement> element0
      = u0.function_space()->element();
  assert(element0);
  std::shared_ptr<const FiniteElement> element1
      = u1.function_space()->element();
  assert(element1);

  xtl::span<const std::uint32_t> cell_info;
  if (element1->needs_dof_transformations()
      or element0->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Get dofmaps
  auto dofmap0 = u0.function_space()->dofmap();
  auto dofmap1 = u1.function_space()->dofmap();

  const xt::xtensor<double, 2> X = element1->interpolation_points();

  // Get block sizes and dof transformation operators
  const int bs0 = element0->block_size();
  const int bs1 = element1->block_size();
  const auto apply_dof_transformation0
      = element0->get_dof_transformation_function<double>(false, false, false);
  const auto apply_inverse_dof_transform1
      = element1->get_dof_transformation_function<T>(true, true, false);

  // Get sizes of elements
  const std::size_t dim0 = element0->space_dimension() / bs0;
  const std::size_t value_size_ref0 = element0->reference_value_size() / bs0;
  const std::size_t value_size0 = element0->value_size() / bs0;

  // Get geometry data
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  xtl::span<const double> x_g = mesh->geometry().x();

  // Evaluate coordinate map basis at reference interpolation points
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, X.shape(0)));
  xt::xtensor<double, 2> dphi;
  cmap.tabulate(1, X, phi);
  dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);

  // Evaluate v basis functions at reference interpolation points
  xt::xtensor<double, 4> basis_derivatives_reference0(
      {1, X.shape(0), dim0, value_size_ref0});
  element0->tabulate(basis_derivatives_reference0, X, 0);

  // Create working arrays
  std::vector<T> local1(element1->space_dimension());
  std::vector<T> coeffs0(element0->space_dimension());
  xt::xtensor<double, 3> basis0({X.shape(0), dim0, value_size0});
  xt::xtensor<double, 3> basis_reference0({X.shape(0), dim0, value_size_ref0});
  xt::xtensor<T, 3> values0({X.shape(0), 1, element1->value_size()});
  xt::xtensor<T, 3> mapped_values0({X.shape(0), 1, element1->value_size()});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});
  xt::xtensor<double, 3> J({X.shape(0), gdim, tdim});
  xt::xtensor<double, 3> K({X.shape(0), tdim, gdim});
  std::vector<double> detJ(X.shape(0));

  // Get interpolation operator
  const xt::xtensor<double, 2>& Pi_1 = element1->interpolation_operator();

  using u_t = xt::xview<decltype(basis_reference0)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(basis_reference0)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto push_forward_fn0 = element0->map_fn<u_t, U_t, J_t, K_t>();

  using u1_t = xt::xview<decltype(values0)&, std::size_t, xt::xall<std::size_t>,
                         xt::xall<std::size_t>>;
  using U1_t = xt::xview<decltype(mapped_values0)&, std::size_t,
                         xt::xall<std::size_t>, xt::xall<std::size_t>>;
  auto pull_back_fn1 = element1->map_fn<U1_t, u1_t, K_t, J_t>();

  // Iterate over mesh and interpolate on each cell
  xtl::span<const T> array0 = u0.x()->array();
  xtl::span<T> array1 = u1.x()->mutable_array();
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  const int num_cells = cell_map->size_local() + cell_map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    // Compute Jacobians and reference points for current cell
    J.fill(0);
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      auto _J = xt::view(J, p, xt::all(), xt::all());
      cmap.compute_jacobian(dphi, coordinate_dofs, _J);
      cmap.compute_jacobian_inverse(_J, xt::view(K, p, xt::all(), xt::all()));
      detJ[p] = cmap.compute_jacobian_determinant(_J);
    }

    // Get evaluated basis on reference, apply DOF transformations, and
    // push forward to physical element
    basis_reference0 = xt::view(basis_derivatives_reference0, 0, xt::all(),
                                xt::all(), xt::all());
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      apply_dof_transformation0(
          xtl::span(basis_reference0.data() + p * dim0 * value_size_ref0,
                    dim0 * value_size_ref0),
          cell_info, c, value_size_ref0);
    }

    for (std::size_t i = 0; i < basis0.shape(0); ++i)
    {
      auto _K = xt::view(K, i, xt::all(), xt::all());
      auto _J = xt::view(J, i, xt::all(), xt::all());
      auto _u = xt::view(basis0, i, xt::all(), xt::all());
      auto _U = xt::view(basis_reference0, i, xt::all(), xt::all());
      push_forward_fn0(_u, _U, _J, detJ[i], _K);
    }

    // Copy expansion coefficients for v into local array
    xtl::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
    for (std::size_t i = 0; i < dofs0.size(); ++i)
      for (int k = 0; k < bs0; ++k)
        coeffs0[bs0 * i + k] = array0[bs0 * dofs0[i] + k];

    // Evaluate v at the interpolation points (physical space values)
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      for (int k = 0; k < bs0; ++k)
      {
        for (std::size_t j = 0; j < value_size0; ++j)
        {
          T acc = 0;
          for (std::size_t i = 0; i < dim0; ++i)
            acc += coeffs0[bs0 * i + k] * basis0(p, i, j);
          values0(p, 0, j * bs0 + k) = acc;
        }
      }
    }

    // Pull back the physical values to the u reference
    for (std::size_t i = 0; i < values0.shape(0); ++i)
    {
      auto _K = xt::view(K, i, xt::all(), xt::all());
      auto _J = xt::view(J, i, xt::all(), xt::all());
      auto _u = xt::view(values0, i, xt::all(), xt::all());
      auto _U = xt::view(mapped_values0, i, xt::all(), xt::all());
      pull_back_fn1(_U, _u, _K, 1.0 / detJ[i], _J);
    }

    auto _mapped_values0 = xt::view(mapped_values0, xt::all(), 0, xt::all());
    interpolation_apply(Pi_1, _mapped_values0, local1, bs1);
    apply_inverse_dof_transform1(local1, cell_info, c, 1);

    // Copy local coefficients to the correct position in u dof array
    xtl::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
    for (std::size_t i = 0; i < dofs1.size(); ++i)
      for (int k = 0; k < bs1; ++k)
        array1[bs1 * dofs1[i] + k] = local1[bs1 * i + k];
  }
}
} // namespace impl

/// Compute the evaluation points in the physical space at which an
/// expression should be computed to interpolate it in a finite elemenet
/// space.
///
/// @param[in] element The element to be interpolated into
/// @param[in] mesh The domain
/// @param[in] cells Indices of the cells in the mesh to compute
/// interpolation coordinates for
/// @return The coordinates in the physical space at which to evaluate
/// an expression. The shape is (3, num_points).
xt::xtensor<double, 2>
interpolation_coords(const fem::FiniteElement& element, const mesh::Mesh& mesh,
                     const xtl::span<const std::int32_t>& cells);

/// Interpolate an expression f(x) in a finite element space
///
/// @param[out] u The function to interpolate into
/// @param[in] f Evaluation of the function `f(x)` at the physical
/// points `x` given by fem::interpolation_coords. The element used in
/// fem::interpolation_coords should be the same element as associated
/// with `u`. The shape of `f` should be (value_size, num_points), or if
/// value_size=1 the shape can be (num_points,).
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate(Function<T>& u, xt::xarray<T>& f,
                 const xtl::span<const std::int32_t>& cells)
{
  const std::shared_ptr<const FiniteElement> element
      = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();
  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  if (f.dimension() == 1)
  {
    if (element->value_size() != 1)
      throw std::runtime_error("Interpolation data has the wrong shape.");
    f.reshape({std::size_t(element->value_size()),
               std::size_t(f.shape(0) / element->value_size())});
  }

  if (f.shape(0) != element->value_size())
    throw std::runtime_error("Interpolation data has the wrong shape.");

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = element->value_size() / element_bs;

  xtl::span<T> coeffs = u.x()->mutable_array();
  std::vector<T> _coeffs(num_scalar_dofs);

  // This assumes that any element with an identity interpolation matrix
  // is a point evaluation
  if (element->interpolation_ident())
  {
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>
        apply_inv_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true, true);
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];
      xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k)
      {
        for (int i = 0; i < num_scalar_dofs; ++i)
          _coeffs[i] = f(k, c * num_scalar_dofs + i);
        apply_inv_transpose_dof_transformation(_coeffs, cell_info, cell, 1);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
  else
  {
    // Get the interpolation points on the reference cells
    const xt::xtensor<double, 2>& X = element->interpolation_points();
    if (X.shape(0) == 0)
    {
      throw std::runtime_error(
          "Interpolation into this space is not yet supported.");
    }

    if (f.shape(1) != cells.size() * X.shape(0))
      throw std::runtime_error("Interpolation data has the wrong shape.");

    // Get coordinate map
    const fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Get geometry data
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Create data structures for Jacobian info
    xt::xtensor<double, 3> J = xt::empty<double>({int(X.shape(0)), gdim, tdim});
    xt::xtensor<double, 3> K = xt::empty<double>({int(X.shape(0)), tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::empty<double>({X.shape(0)});

    xt::xtensor<double, 2> coordinate_dofs
        = xt::empty<double>({num_dofs_g, gdim});

    xt::xtensor<T, 3> reference_data({X.shape(0), 1, value_size});
    xt::xtensor<T, 3> _vals({X.shape(0), 1, value_size});

    // Tabulate 1st order derivatives of shape functions at interpolation coords
    xt::xtensor<double, 3> dphi = xt::view(
        cmap.tabulate(1, X), xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>
        apply_inverse_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true);

    // Get interpolation operator
    const xt::xtensor<double, 2>& Pi = element->interpolation_operator();

    using U_t = xt::xview<decltype(reference_data)&, std::size_t,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    auto pull_back_fn = element->map_fn<U_t, U_t, J_t, J_t>();

    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];
      auto x_dofs = x_dofmap.links(cell);
      for (int i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (int j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[pos + j];
      }

      // Compute J, detJ and K
      J.fill(0);
      for (std::size_t p = 0; p < X.shape(0); ++p)
      {
        cmap.compute_jacobian(xt::view(dphi, xt::all(), p, xt::all()),
                              coordinate_dofs,
                              xt::view(J, p, xt::all(), xt::all()));
        cmap.compute_jacobian_inverse(xt::view(J, p, xt::all(), xt::all()),
                                      xt::view(K, p, xt::all(), xt::all()));
        detJ[p] = cmap.compute_jacobian_determinant(
            xt::view(J, p, xt::all(), xt::all()));
      }

      xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
      for (int k = 0; k < element_bs; ++k)
      {
        // Extract computed expression values for element block k
        for (int m = 0; m < value_size; ++m)
        {
          std::copy_n(&f(k * value_size + m, c * X.shape(0)), X.shape(0),
                      xt::view(_vals, xt::all(), 0, m).begin());
        }

        // Get element degrees of freedom for block
        for (std::size_t i = 0; i < X.shape(0); ++i)
        {
          auto _K = xt::view(K, i, xt::all(), xt::all());
          auto _J = xt::view(J, i, xt::all(), xt::all());
          auto _u = xt::view(_vals, i, xt::all(), xt::all());
          auto _U = xt::view(reference_data, i, xt::all(), xt::all());
          pull_back_fn(_U, _u, _K, 1.0 / detJ[i], _J);
        }

        auto ref_data = xt::view(reference_data, xt::all(), 0, xt::all());
        impl::interpolation_apply(Pi, ref_data, _coeffs, element_bs);
        apply_inverse_transpose_dof_transformation(_coeffs, cell_info, cell, 1);

        // Copy interpolation dofs into coefficient vector
        assert(_coeffs.size() == num_scalar_dofs);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
}

/// Interpolate from one finite element Function to another on the same
/// mesh
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v)
{
  assert(u.function_space());
  assert(v.function_space());
  if (u.function_space() == v.function_space())
  {
    // Same function spaces
    xtl::span<T> u1_array = u.x()->mutable_array();
    xtl::span<const T> u0_array = v.x()->array();
    std::copy(u0_array.begin(), u0_array.end(), u1_array.begin());
  }
  else
  {
    // Get mesh and check that functions share the same mesh
    const auto mesh = u.function_space()->mesh();
    assert(mesh);
    if (mesh != v.function_space()->mesh())
    {
      throw std::runtime_error(
          "Interpolation on different meshes not supported (yet).");
    }

    // Get elements
    auto element0 = v.function_space()->element();
    assert(element0);
    auto element1 = u.function_space()->element();
    assert(element1);
    if (element1->hash() == element0->hash())
    {
      // Same element, different dofmaps

      const int tdim = mesh->topology().dim();
      auto cell_map = mesh->topology().index_map(tdim);
      assert(cell_map);

      // Get dofmaps
      std::shared_ptr<const fem::DofMap> dofmap0 = v.function_space()->dofmap();
      assert(dofmap0);
      std::shared_ptr<const fem::DofMap> dofmap1 = u.function_space()->dofmap();
      assert(dofmap1);

      xtl::span<T> u1_array = u.x()->mutable_array();
      xtl::span<const T> u0_array = v.x()->array();

      // Iterate over mesh and interpolate on each cell
      const int num_cells = cell_map->size_local() + cell_map->num_ghosts();
      const int bs = dofmap0->bs();
      assert(bs == dofmap1->bs());
      for (int c = 0; c < num_cells; ++c)
      {
        xtl::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
        xtl::span<const std::int32_t> dofs1 = dofmap1->cell_dofs(c);
        assert(dofs0.size() == dofs1.size());
        for (std::size_t i = 0; i < dofs0.size(); ++i)
          for (int k = 0; k < bs; ++k)
            u1_array[bs * dofs1[i] + k] = u0_array[bs * dofs0[i] + k];
      }
    }
    else if (element1->map_type() == element0->map_type())
    {
      // Different elements, same basis function map type
      impl::interpolate_same_map(u, v);
    }
    else
    {
      //  Different elements with different maps for basis functions
      impl::interpolate_nonmatching_maps(u, v);
    }
  }
}

/// Interpolate from an Expression into a compatible Function on the
/// same mesh
/// @param[out] u The function to interpolate into
/// @param[in] expr The Expression to be interpolated. The Expression
/// must have been created using the reference coordinates
/// `FiniteElement::interpolation_points()` for the element associated
/// with `u`.
/// @param[in] cells List of cell indices to interpolate on
template <typename T>
void interpolate(Function<T>& u, const Expression<T>& expr,
                 const xtl::span<const std::int32_t>& cells)
{
  // Check that spaces are compatible
  std::size_t value_size = expr.value_size();
  assert(u.function_space());
  assert(u.function_space()->element());
  assert(value_size == u.function_space()->element()->value_size());
  assert(expr.x().shape()
         == u.function_space()->element()->interpolation_points().shape());

  // Array to hold evaluted Expression
  std::size_t num_cells = cells.size();
  std::size_t num_points = expr.x().shape(0);
  xt::xtensor<T, 3> f({num_cells, num_points, value_size});

  // Evaluate Expression at points
  auto f_view = xt::reshape_view(f, {num_cells, num_points * value_size});
  expr.eval(cells, f_view);

  // Reshape evaluated data to fit interpolate
  // Expression returns matrix of shape (num_cells, num_points *
  // value_size), i.e. xyzxyz ordering of dof values per cell per point.
  // The interpolation uses xxyyzz input, ordered for all points of each
  // cell, i.e. (value_size, num_cells*num_points)
  xt::xarray<T> _f = xt::reshape_view(xt::transpose(f, {2, 0, 1}),
                                      {value_size, num_cells * num_points});

  // Interpolate values into appropriate space
  interpolate(u, _f, cells);
}
} // namespace dolfinx::fem
