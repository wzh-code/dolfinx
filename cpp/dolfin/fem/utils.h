// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateMapping.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/cell_types.h>
#include <memory>
#include <vector>

struct fenics_dofmap;
struct fenics_form;
struct fenics_coordinate_mapping;
struct fenics_function_space;

namespace dolfin
{
namespace common
{
class IndexMap;
}

namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la
namespace function
{
class Constant;
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Geometry;
class Mesh;
} // namespace mesh

namespace fem
{
class Form;

/// Compute IndexMaps for stacked index maps
std::vector<std::vector<std::shared_ptr<const common::IndexMap>>>
blocked_index_sets(const std::vector<std::vector<const fem::Form*>> a);

/// Create matrix. Matrix is not zeroed.
la::PETScMatrix create_matrix(const Form& a);

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
la::PETScMatrix
create_matrix_block(std::vector<std::vector<const fem::Form*>> a);

/// Create nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix
create_matrix_nest(std::vector<std::vector<const fem::Form*>> a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector create_vector_block(std::vector<const fem::Form*> L);

/// Initialise nested (VecNest) vector. Vector is not zeroed.
la::PETScVector create_vector_nest(std::vector<const fem::Form*> L);

/// Get new global index in 'spliced' indices
std::size_t get_global_index(const std::vector<const common::IndexMap*> maps,
                             const int field, const int n);

/// Create an ElementDofLayout from a fenics_dofmap
ElementDofLayout create_element_dof_layout(const fenics_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create dof map on mesh from a fenics_dofmap
///
/// @param[in] dofmap The fenics_dofmap.
/// @param[in] mesh The mesh.
DofMap create_dofmap(const fenics_dofmap& dofmap, const mesh::Mesh& mesh);

/// Create form (shared data)
///
/// @param[in] fenics_form The FEniCS form.
/// @param[in] spaces Vector of function spaces.
Form create_form(
    const fenics_form& fenics_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces);

/// Extract coefficients from FEniCS form
std::vector<std::tuple<int, std::string, std::shared_ptr<function::Function>>>
get_coeffs_from_fenics_form(const fenics_form& fenics_form);

/// Extract coefficients from FEniCS form
std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
get_constants_from_fenics_form(const fenics_form& fenics_form);

/// Get dolfin::fem::CoordinateMapping from FEniCS
std::shared_ptr<const fem::CoordinateMapping>
get_cmap_from_fenics_cmap(const fenics_coordinate_mapping& fenics_cmap);

/// Create FunctionSpace from FEniCS function space
/// @param fptr Function Pointer to a fenics_function_space_create function
/// @param mesh Mesh
/// @return The created FunctionSpace
std::shared_ptr<function::FunctionSpace>
create_functionspace(fenics_function_space* (*fptr)(void), std::shared_ptr<mesh::Mesh> mesh);

} // namespace fem
} // namespace dolfin
