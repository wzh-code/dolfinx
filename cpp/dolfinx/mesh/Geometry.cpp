// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Topology.h"
#include <boost/functional/hash.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/partition.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
int Geometry::dim() const { return _dim; }
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& Geometry::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Geometry::index_map() const
{
  return _index_map;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>& Geometry::x() { return _x; }
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& Geometry::x() const { return _x; }
//-----------------------------------------------------------------------------
const fem::CoordinateElement& Geometry::cmap() const { return _cmap; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::input_global_indices() const
{
  return _input_global_indices;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry mesh::create_geometry(
    MPI_Comm comm, const Topology& topology,
    const std::vector<std::uint8_t>& cell_layout_type,
    const std::vector<fem::CoordinateElement>& coordinate_element,
    const graph::AdjacencyList<std::int64_t>& cell_nodes,
    const xt::xtensor<double, 2>& x,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::build_dofmap_data

  LOG(INFO) << "Create geometry";

  std::vector<fem::ElementDofLayout> element_dof_layouts;

  for (std::size_t i = 0; i < coordinate_element.size(); ++i)
    element_dof_layouts.push_back(coordinate_element[i].dof_layout());

  LOG(INFO) << "Create dofmap " << element_dof_layouts.size();

  //  Build 'geometry' dofmap on the topology
  auto [dof_index_map, bs, dofmap] = fem::build_dofmap_data(
      comm, topology, cell_layout_type, element_dof_layouts, reorder_fn);

  LOG(INFO) << "Got geometry dofmap";
  LOG(INFO) << dofmap.str();

  // If the mesh has higher order geometry, permute the dofmap
  if (coordinate_element[0].needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      coordinate_element[0].unpermute_dofs(dofmap.links(cell), cell_info[cell]);
  }

  // Build list of unique (global) node indices from adjacency list
  // (geometry nodes)
  std::vector<std::int64_t> indices = cell_nodes.array();
  dolfinx::radix_sort(xtl::span(indices));
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

  //  Fetch node coordinates by global index from other ranks. Order of
  //  coords matches order of the indices in 'indices'
  xt::xtensor<double, 2> coords
      = graph::build::distribute_data<double>(comm, indices, x);

  LOG(INFO) << "x shape = " << x.shape(0) << "x" << x.shape(1);
  LOG(INFO) << "Coords shape = " << coords.shape(0) << "x" << coords.shape(1);
  LOG(INFO) << "Cell-nodes = " << cell_nodes.str() << "\n";

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cell_nodes
  std::vector l2g
      = graph::build::compute_local_to_global_links(cell_nodes, dofmap);

  std::stringstream s;
  for (auto q : l2g)
    s << q << "-";
  LOG(INFO) << s.str();

  // Compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  std::vector l2l = graph::build::compute_local_to_local(l2g, indices);

  // Build coordinate dof array,  copying coordinates to correct
  // position
  xt::xtensor<double, 2> xg({coords.shape(0), coords.shape(1)});
  for (std::size_t i = 0; i < coords.shape(0); ++i)
  {
    auto row = xt::view(coords, l2l[i]);
    std::copy(row.cbegin(), row.cend(), xt::row(xg, i).begin());
  }

  // Allocate space for input global indices and copy data
  std::vector<std::int64_t> igi(indices.size());
  std::transform(l2l.cbegin(), l2l.cend(), igi.begin(),
                 [&indices](auto index) { return indices[index]; });

  return Geometry(dof_index_map, std::move(dofmap), coordinate_element[0],
                  std::move(xg), std::move(igi));
}
//-----------------------------------------------------------------------------
