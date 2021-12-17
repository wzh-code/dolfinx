// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/mesh/Mesh.h>
#include <stdexcept>
#include <unordered_set>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::extract_topology(const CellType& cell_type,
                       const fem::ElementDofLayout& layout,
                       const graph::AdjacencyList<std::int64_t>& cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int> local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology(cells.num_nodes() * num_vertices_per_cell);
  for (int c = 0; c < cells.num_nodes(); ++c)
  {
    auto p = cells.links(c);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      topology[num_vertices_per_cell * c + j] = p[local_vertices[j]];
  }

  return graph::build_adjacency_list<std::int64_t>(std::move(topology),
                                                   num_vertices_per_cell);
}
//-----------------------------------------------------------------------------
std::vector<double> mesh::h(const Mesh& mesh,
                            const xtl::span<const std::int32_t>& entities,
                            int dim)
{
  if (dim != mesh.topology().dim())
    throw std::runtime_error("Cell size when dim ne tdim  requires updating.");

  if (mesh.topology().cell_type() == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  // Get number of cell vertices
  const CellType type = cell_entity_type(mesh.topology().cell_type(), dim, 0);
  const int num_vertices = num_cell_vertices(type);

  // Get geometry dofmap and dofs
  const Geometry& geometry = mesh.geometry();
  const graph::AdjacencyList<std::int32_t>& x_dofs = geometry.dofmap();
  const xt::xtensor<double, 2>& geom_dofs = geometry.x();
  std::vector<double> h_cells(entities.size(), 0);
  assert(num_vertices <= 8);
  xt::xtensor_fixed<double, xt::xshape<8, 3>> points;
  for (std::size_t e = 0; e < entities.size(); ++e)
  {
    // Get the coordinates  of the vertices
    auto dofs = x_dofs.links(entities[e]);

    // The below should work, but misbehaves with the Intel icpx compiler
    // xt::view(points, xt::range(0, num_vertices), xt::all())
    //     = xt::view(geom_dofs, xt::keep(dofs), xt::all());
    auto points_view = xt::view(points, xt::range(0, num_vertices), xt::all());
    points_view.assign(xt::view(geom_dofs, xt::keep(dofs), xt::all()));

    // Get maximum edge length
    for (int i = 0; i < num_vertices; ++i)
    {
      for (int j = i + 1; j < num_vertices; ++j)
      {
        auto p0 = xt::row(points, i);
        auto p1 = xt::row(points, j);
        h_cells[e] = std::max(h_cells[e], xt::norm_l2(p0 - p1)());
      }
    }
  }

  return h_cells;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
mesh::cell_normals(const mesh::Mesh& mesh, int dim,
                   const xtl::span<const std::int32_t>& entities)
{
  if (mesh.topology().cell_type() == mesh::CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  const int gdim = mesh.geometry().dim();
  const CellType type = cell_entity_type(mesh.topology().cell_type(), dim, 0);

  // Find geometry nodes for topology entities
  const xt::xtensor<double, 2>& xg = mesh.geometry().x();

  // Orient cells if they are tetrahedron
  bool orient = false;
  if (mesh.topology().cell_type() == mesh::CellType::tetrahedron)
    orient = true;
  xt::xtensor<std::int32_t, 2> geometry_entities
      = entities_to_geometry(mesh, dim, entities, orient);

  const std::size_t num_entities = entities.size();
  xt::xtensor<double, 2> n({num_entities, 3});
  switch (type)
  {
  case CellType::interval:
  {
    if (gdim > 2)
      throw std::invalid_argument("Interval cell normal undefined in 3D");
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get the two vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);

      // Define normal by rotating tangent counter-clockwise
      auto t = p1 - p0;
      auto ni = xt::row(n, i);
      ni[0] = -t[1];
      ni[1] = t[0];
      ni[2] = 0.0;
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  case CellType::triangle:
  {
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get the three vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);
      auto p2 = xt::row(xg, vertices[2]);

      // Define cell normal via cross product of first two edges
      auto ni = xt::row(n, i);
      ni = math::cross((p1 - p0), (p2 - p0));
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  case CellType::quadrilateral:
  {
    // TODO: check
    for (std::size_t i = 0; i < num_entities; ++i)
    {
      // Get three vertices as points
      auto vertices = xt::row(geometry_entities, i);
      auto p0 = xt::row(xg, vertices[0]);
      auto p1 = xt::row(xg, vertices[1]);
      auto p2 = xt::row(xg, vertices[2]);

      // Defined cell normal via cross product of first two edges:
      auto ni = xt::row(n, i);
      ni = math::cross((p1 - p0), (p2 - p0));
      ni /= xt::norm_l2(ni);
    }
    return n;
  }
  default:
    throw std::invalid_argument(
        "cell_normal not supported for this cell type.");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
mesh::compute_midpoints(const Mesh& mesh, int dim,
                        const xtl::span<const std::int32_t>& entities)
{
  const xt::xtensor<double, 2>& x = mesh.geometry().x();

  // Build map from entity -> geometry dof
  // FIXME: This assumes a linear geometry.
  xt::xtensor<std::int32_t, 2> entity_to_geometry
      = entities_to_geometry(mesh, dim, entities, false);

  xt::xtensor<double, 2> x_mid({entities.size(), 3});
  for (std::size_t e = 0; e < entity_to_geometry.shape(0); ++e)
  {
    auto rows = xt::row(entity_to_geometry, e);
    // The below should work, but misbehaves with the Intel icpx compiler
    // xt::row(x_mid, e) = xt::mean(xt::view(x, xt::keep(rows)), 0);
    auto _x = xt::row(x_mid, e);
    _x.assign(xt::mean(xt::view(x, xt::keep(rows)), 0));
  }

  return x_mid;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities(
    const Mesh& mesh, int dim,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker)
{
  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim, 0);
  if (dim < tdim)
    mesh.topology_mutable().create_connectivity(dim, 0);

  // Get all vertex 'node' indices
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const std::int32_t num_vertices = topology.index_map(0)->size_local()
                                    + topology.index_map(0)->num_ghosts();
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  std::vector<std::int32_t> vertex_to_node(num_vertices);
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    auto vertices = c_to_v->links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_node[vertices[i]] = x_dofs[i];
  }

  // Pack coordinates of vertices
  const xt::xtensor<double, 2>& x_nodes = mesh.geometry().x();
  xt::xtensor<double, 2> x_vertices({3, vertex_to_node.size()});
  for (std::size_t i = 0; i < vertex_to_node.size(); ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x_vertices(j, i) = x_nodes(vertex_to_node[i], j);

  // Run marker function on vertex coordinates
  const xt::xtensor<bool, 1> marked = marker(x_vertices);
  if (marked.shape(0) != x_vertices.shape(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Iterate over entities to build vector of marked entities
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (int e = 0; e < e_to_v->num_nodes(); ++e)
  {
    // Iterate over entity vertices
    bool all_vertices_marked = true;
    for (std::int32_t v : e_to_v->links(e))
    {
      if (!marked[v])
      {
        all_vertices_marked = false;
        break;
      }
    }

    if (all_vertices_marked)
      entities.push_back(e);
  }

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::locate_entities_boundary(
    const Mesh& mesh, int dim,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker)
{
  const Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  if (dim == tdim)
  {
    throw std::runtime_error(
        "Cannot use mesh::locate_entities_boundary (boundary) for cells.");
  }

  // Compute marker for boundary facets
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  const std::vector boundary_facet = compute_boundary_facets(topology);

  // Create entities and connectivities
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, dim);
  mesh.topology_mutable().create_connectivity(tdim - 1, 0);
  mesh.topology_mutable().create_connectivity(0, tdim);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  // Build set of vertices on boundary and set of boundary entities
  auto f_to_v = topology.connectivity(tdim - 1, 0);
  assert(f_to_v);
  auto f_to_e = topology.connectivity(tdim - 1, dim);
  assert(f_to_e);
  std::unordered_set<std::int32_t> boundary_vertices;
  std::unordered_set<std::int32_t> facet_entities;
  for (std::size_t f = 0; f < boundary_facet.size(); ++f)
  {
    if (boundary_facet[f])
    {
      for (auto e : f_to_e->links(f))
        facet_entities.insert(e);

      for (auto v : f_to_v->links(f))
        boundary_vertices.insert(v);
    }
  }

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const xt::xtensor<double, 2>& x_nodes = mesh.geometry().x();

  // Build vector of boundary vertices
  const std::vector<std::int32_t> vertices(boundary_vertices.begin(),
                                           boundary_vertices.end());

  // Get all vertex 'node' indices
  auto v_to_c = topology.connectivity(0, tdim);
  assert(v_to_c);
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  xt::xtensor<double, 2> x_vertices({3, vertices.size()});
  std::vector<std::int32_t> vertex_to_pos(v_to_c->num_nodes(), -1);
  for (std::size_t i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t v = vertices[i];

    // Get first cell and find position
    const int c = v_to_c->links(v)[0];
    auto vertices = c_to_v->links(c);
    auto it = std::find(vertices.begin(), vertices.end(), v);
    assert(it != vertices.end());
    const int local_pos = std::distance(vertices.begin(), it);

    auto dofs = x_dofmap.links(c);
    for (int j = 0; j < 3; ++j)
      x_vertices(j, i) = x_nodes(dofs[local_pos], j);

    vertex_to_pos[v] = i;
  }

  // Run marker function on the vertex coordinates
  const xt::xtensor<bool, 1> marked = marker(x_vertices);
  if (marked.shape(0) != x_vertices.shape(1))
    throw std::runtime_error("Length of array of markers is wrong.");

  // Loop over entities and check vertex markers
  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  std::vector<std::int32_t> entities;
  for (auto e : facet_entities)
  {
    // Assume all vertices on this entity are marked
    bool all_vertices_marked = true;

    // Iterate over entity vertices
    for (auto v : e_to_v->links(e))
    {
      const std::int32_t pos = vertex_to_pos[v];
      if (!marked[pos])
      {
        all_vertices_marked = false;
        break;
      }
    }

    // Mark facet with all vertices marked
    if (all_vertices_marked)
      entities.push_back(e);
  }

  return entities;
}
//-----------------------------------------------------------------------------
xt::xtensor<std::int32_t, 2>
mesh::entities_to_geometry(const Mesh& mesh, int dim,
                           const xtl::span<const std::int32_t>& entity_list,
                           bool orient)
{
  CellType cell_type = mesh.topology().cell_type();
  if (cell_type == CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cells");

  const std::size_t num_entity_vertices
      = num_cell_vertices(cell_entity_type(cell_type, dim, 0));
  xt::xtensor<std::int32_t, 2> entity_geometry(
      {entity_list.size(), num_entity_vertices});

  if (orient
      and (cell_type != dolfinx::mesh::CellType::tetrahedron or dim != 2))
  {
    throw std::runtime_error("Can only orient facets of a tetrahedral mesh");
  }

  const Geometry& geometry = mesh.geometry();
  const xt::xtensor<double, 2>& geom_dofs = geometry.x();
  const Topology& topology = mesh.topology();

  const int tdim = topology.dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(dim, 0);
  mesh.topology_mutable().create_connectivity(tdim, 0);

  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
  const auto e_to_c = topology.connectivity(dim, tdim);
  assert(e_to_c);
  const auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);
  const auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
  {
    const std::int32_t idx = entity_list[i];
    const std::int32_t cell = e_to_c->links(idx)[0];
    auto ev = e_to_v->links(idx);
    assert(ev.size() == num_entity_vertices);
    const auto cv = c_to_v->links(cell);
    const auto xc = xdofs.links(cell);
    for (std::size_t j = 0; j < num_entity_vertices; ++j)
    {
      int k = std::distance(cv.begin(), std::find(cv.begin(), cv.end(), ev[j]));
      assert(k < (int)cv.size());
      entity_geometry(i, j) = xc[k];
    }

    if (orient)
    {
      // Compute cell midpoint
      xt::xtensor_fixed<double, xt::xshape<3>> midpoint = {0, 0, 0};
      for (std::int32_t j : xc)
        for (int k = 0; k < 3; ++k)
          midpoint[k] += geom_dofs(j, k);
      midpoint /= xc.size();

      // Compute vector triple product of two edges and vector to midpoint
      auto p0 = xt::row(geom_dofs, entity_geometry(i, 0));
      auto p1 = xt::row(geom_dofs, entity_geometry(i, 1));
      auto p2 = xt::row(geom_dofs, entity_geometry(i, 2));

      xt::xtensor_fixed<double, xt::xshape<3, 3>> a;
      xt::row(a, 0) = midpoint - p0;
      xt::row(a, 1) = p1 - p0;
      xt::row(a, 2) = p2 - p0;

      // Midpoint direction should be opposite to normal, hence this
      // should be negative. Switch points if not.
      if (math::det(a) > 0.0)
        std::swap(entity_geometry(i, 1), entity_geometry(i, 2));
    }
  }

  return entity_geometry;
}
//------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Mesh& mesh)
{
  // Note: Possible duplication of mesh::Topology::compute_boundary_facets

  const Topology& topology = mesh.topology();
  std::vector<std::int32_t> surface_facets;

  // Get number of facets owned by this process
  const int tdim = topology.dim();
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  assert(topology.index_map(tdim - 1));

  // Only need to consider shared facets when there are no ghost cells
  std::set<std::int32_t> fwd_shared_facets;
  if (topology.index_map(tdim)->num_ghosts() == 0)
  {
    fwd_shared_facets.insert(
        topology.index_map(tdim - 1)->scatter_fwd_indices().array().begin(),
        topology.index_map(tdim - 1)->scatter_fwd_indices().array().end());
  }

  // Find all owned facets (not ghost) with only one attached cell, which are
  // also not shared forward (ghost on another process)
  const int num_facets = topology.index_map(tdim - 1)->size_local();
  for (int f = 0; f < num_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1
        and fwd_shared_facets.find(f) == fwd_shared_facets.end())
    {
      surface_facets.push_back(f);
    }
  }

  return surface_facets;
}
//------------------------------------------------------------------------------
mesh::CellPartitionFunction
mesh::create_cell_partitioner(const graph::partition_fn& partfn)
{
  return
      [partfn](
          MPI_Comm comm, int nparts, int tdim,
          const graph::AdjacencyList<std::int64_t>& cells,
          GhostMode ghost_mode) -> dolfinx::graph::AdjacencyList<std::int32_t>
  {
    LOG(INFO) << "Compute partition of cells across ranks";

    // Compute distributed dual graph (for the cells on this process)
    const auto [dual_graph, num_ghost_edges]
        = build_dual_graph(comm, cells, tdim);

    // Just flag any kind of ghosting for now
    bool ghosting = (ghost_mode != mesh::GhostMode::none);

    // Compute partition
    return partfn(comm, nparts, dual_graph, num_ghost_edges, ghosting);
  };
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::compute_incident_entities(const Mesh& mesh,
                                const xtl::span<const std::int32_t>& entities,
                                int d0, int d1)
{
  auto map0 = mesh.topology().index_map(d0);
  if (!map0)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d0)
                             + " have not been created.");
  }

  auto map1 = mesh.topology().index_map(d1);
  if (!map1)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(d1)
                             + " have not been created.");
  }

  auto e0_to_e1 = mesh.topology().connectivity(d0, d1);
  if (!e0_to_e1)
  {
    throw std::runtime_error("Connectivity missing: (" + std::to_string(d0)
                             + ", " + std::to_string(d1) + ")");
  }

  std::vector<std::int32_t> entities1;
  for (std::int32_t entity : entities)
  {
    auto e = e0_to_e1->links(entity);
    entities1.insert(entities1.end(), e.begin(), e.end());
  }

  std::sort(entities1.begin(), entities1.end());
  entities1.erase(std::unique(entities1.begin(), entities1.end()),
                  entities1.end());
  return entities1;
}
//-----------------------------------------------------------------------------
mesh::Mesh mesh::update_ghosts(const mesh::Mesh& mesh,
                               graph::AdjacencyList<std::int32_t>& dest)
{
  // Get topology information
  const mesh::Topology& topology = mesh.topology();
  int tdim = topology.dim();

  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(tdim);
  std::shared_ptr<const common::IndexMap> vert_map = topology.index_map(0);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cv
      = topology.connectivity(tdim, 0);

  std::int32_t num_local_cells = cell_map->size_local();
  std::int32_t num_ghosts = cell_map->num_ghosts();

  // Get geometry information
  const mesh::Geometry& geometry = mesh.geometry();
  int gdim = geometry.dim();
  const xt::xtensor<double, 2>& coord = geometry.x();

  std::vector<std::int32_t> vertex_to_coord(vert_map->size_local()
                                            + vert_map->num_ghosts());
  for (std::int32_t c = 0; c < num_local_cells + num_ghosts; ++c)
  {
    auto vertices = cv->links(c);
    auto dofs = geometry.dofmap().links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_coord[vertices[i]] = dofs[i];
  }

  std::vector<std::int64_t> topology_array;
  std::vector<std::int32_t> counter(num_local_cells);
  std::vector<int64_t> global_inds(cv->num_links(0));

  // Compute topology information
  for (std::int32_t i = 0; i < num_local_cells; i++)
  {
    vert_map->local_to_global(cv->links(i), global_inds);
    topology_array.insert(topology_array.end(), global_inds.begin(),
                          global_inds.end());
    counter[i] += global_inds.size();
  }

  std::vector<std::int32_t> offsets(counter.size() + 1, 0);
  std::partial_sum(counter.begin(), counter.end(), offsets.begin() + 1);
  graph::AdjacencyList<std::int64_t> cell_vertices(topology_array, offsets);

  // Copy over existing mesh vertices
  const std::int32_t num_local_vertices = vert_map->size_local();
  xt::xtensor<double, 2> x = xt::empty<double>({num_local_vertices, gdim});
  for (int v = 0; v < num_local_vertices; ++v)
    for (int j = 0; j < gdim; ++j)
      x(v, j) = coord(vertex_to_coord[v], j);

  auto partitioner = [&dest](...) { return dest; };

  return mesh::create_mesh(mesh.comm(), cell_vertices, geometry.cmap(), x,
                           mesh::GhostMode::shared_facet, partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh add_ghost_layer(const mesh::Mesh& mesh, int dim)
{
  // Get topology information
  const mesh::Topology& topology = mesh.topology();
  int tdim = topology.dim();
  int facet_dim = tdim - 1;

  if (dim >= tdim || dim < 0)
    throw std::runtime_error("Entity dimension should an integer between 0 and "
                             "the facet dimension ("
                             + std::to_string(facet_dim) + ")");

  // Compute facets on the interface between this process and adjacent
  // neighbors. The facets can be either owned and shared or ghosts is this
  // process.
  std::vector<bool> facets_bool
      = dolfinx::mesh::compute_interface_facets(topology);
  int num_interface_facets = std::reduce(facets_bool.begin(), facets_bool.end(),
                                         int(0), std::plus<int>());

  if (num_interface_facets == 0)
    return mesh;

  // Convert vector of bools (flags) to vector of facet indices
  std::vector<std::int32_t> facets(num_interface_facets);
  std::int32_t pos = 0;
  for (auto it = facets_bool.begin(); it != facets_bool.end(); it++)
  {
    if (*it)
    {
      std::ptrdiff_t facet = std::distance(facets_bool.begin(), it);
      facets[pos++] = static_cast<std::int32_t>(facet);
    }
  }

  // Compute interface entities of dimension "dim", entities that are incident
  // to the interface facets.
  std::vector<std::int32_t> entities;
  if (dim == facet_dim)
    std::swap(entities, facets);
  else
  {
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int32_t>>
        facets_entities = topology.connectivity(facet_dim, dim);

    entities.reserve(num_interface_facets * facets_entities->num_links(0));
    for (const std::int32_t& f : facets)
    {
      for (const std::int32_t& e : facets_entities->links(f))
        entities.push_back(e);
    }
  }

  // Sort entities and remove duplicates
  dolfinx::radix_sort<std::int32_t>(entities);
  entities.erase(std::unique(entities.begin(), entities.end()), entities.end());

  // Get owned and ghost entities range
  std::shared_ptr<const common::IndexMap> entity_map = topology.index_map(dim);
  std::int32_t num_entities_dim = entity_map->size_local();
  std::int32_t num_owned_entities = std::distance(
      entities.begin(),
      std::lower_bound(entities.begin(), entities.end(), num_entities_dim));
  std::int32_t num_ghost_entities = entities.size() - num_owned_entities;

  // The computation of ghost layer is performed in 5 neighbor-wise
  // communication steps:
  // 1 - Request [ghost->owner] : Process request entity-cell connectivity
  // information.
  // 2 - Distribute Requests [onwer->ghost] : Owner distributes the request to
  // all sharing processes.
  // 3 - Reply[ghost->owner] : Sharing process replies (entity-cell connectivity
  // information).
  // 4 - Distribute Replies [onwer->ghost] : Owner process distribute replies
  // entity-cell connectivity information to the requesting processes.
  // 5 - Finalize [ghost->owner]: Process computes new ghosts and informs the
  // owning processes.

  // -----------------------
  // First communication step: [ghost->owner]
  // Each process requests entity-cell connectivity to entity owner.
  // Output: Adjacency list [neighbor - entities for wich connectivity
  // information is required]
  graph::AdjacencyList<std::int64_t> remote_entities(0);
  {
    // Ghost to owner communicator
    MPI_Comm comm = entity_map->comm(common::IndexMap::Direction::reverse);
    const auto dest_ranks = dolfinx::MPI::neighbors(comm)[1];

    // Global-to-neigbourhood map for destination ranks
    std::unordered_map<int, std::int32_t> dest_proc_to_neighbor;
    for (std::size_t i = 0; i < dest_ranks.size(); ++i)
      dest_proc_to_neighbor.insert({dest_ranks[i], i});

    // Compute size of data to send to each process
    std::vector<std::int32_t> counter(dest_ranks.size(), 0);
    std::vector<std::int64_t> send_data(num_ghost_entities);
    const std::vector<std::int64_t>& ghosts = entity_map->ghosts();
    std::vector<int> ghost_owner_rank = entity_map->ghost_owner_rank();
    for (std::int32_t i = 0; i < num_ghost_entities; ++i)
    {
      // entity local index in ghost range
      int entity = entities[num_owned_entities + i] - num_entities_dim;
      // convert to global and add to send buffer
      send_data[i] = ghosts[entity];
      const auto it = dest_proc_to_neighbor.find(ghost_owner_rank[entity]);
      assert(it != dest_proc_to_neighbor.end());
      // count number of entities per owner
      counter[it->second]++;
    }

    std::vector<int> send_disp(dest_ranks.size() + 1, 0);
    std::partial_sum(counter.begin(), counter.end(),
                     std::next(send_disp.begin(), 1));

    // Send ghost entities to the owners and receive ghost entities which
    // require cell connectivity information
    const graph::AdjacencyList<std::int64_t> ghost_entities(
        std::move(send_data), std::move(send_disp));
    remote_entities = dolfinx::MPI::neighbor_all_to_all(comm, ghost_entities);
  }

  //-------------------------
  // Second communication step: [owner->ghost]
  // Entity owner requests information of entity-cell connectivity to all
  // processes that share a given entity marked as interface.
  // Output: list of ghost entities (local entity number) which requires
  // entity-cell connectivity data
  std::vector<std::int32_t> ghost_entities;
  {
    // Get list of all onwed entities that need cell connectivity information
    // that includes received remote entities and local entities marked as
    // interface
    std::vector<std::int32_t> owned_entities;
    {
      std::vector<std::int64_t> global_entities = remote_entities.array();
      std::vector<std::int32_t> local_entities(global_entities.size());
      entity_map->global_to_local(global_entities, local_entities);
      dolfinx::radix_sort<std::int32_t>(local_entities);

      owned_entities.reserve(local_entities.size() + num_owned_entities);
      std::merge(local_entities.begin(), local_entities.end(), entities.begin(),
                 std::next(entities.begin(), num_owned_entities),
                 std::back_inserter(owned_entities));
      owned_entities.erase(
          std::unique(owned_entities.begin(), owned_entities.end()),
          owned_entities.end());
    }

    std::vector<std::int8_t> marked_entities(entity_map->num_ghosts());
    std::vector<std::int8_t> entity_needs_info(entity_map->size_local(), 0);

    // Mark entities that require cell connectivity info
    for (const std::int32_t& e : owned_entities)
      entity_needs_info[e] = 1;

    // get remove markerd entities
    entity_map->scatter_fwd<std::int8_t>(entity_needs_info, marked_entities, 1);

    std::int32_t num_ghost_entities
        = std::reduce(marked_entities.begin(), marked_entities.end(),
                      std::int32_t(0), std::plus<std::int32_t>());
    ghost_entities.reserve(num_ghost_entities);
    for (auto it = marked_entities.begin(); it != marked_entities.end(); it++)
    {
      if (*it)
      {
        std::ptrdiff_t facet = std::distance(marked_entities.begin(), it);
        ghost_entities.push_back(static_cast<std::int32_t>(facet));
      }
    }
  }

  //-------------------------
  // Third communication step: [ghost->owner]
  // Reply request with entity-cell connectivity data to entity owner
  // [entity num_cells cell0 ...]
  // Output: List of owned entities cell connectivity
  {
    // Get entity cell connectivity
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int32_t>> entity_cell
        = topology.connectivity(dim, tdim);

    // Ghost to owner communicator
    MPI_Comm comm = entity_map->comm(common::IndexMap::Direction::reverse);
    const auto dest_ranks = dolfinx::MPI::neighbors(comm)[1];

    // Global-to-neigbourhood map for destination ranks
    std::unordered_map<int, std::int32_t> dest_proc_to_neighbor;
    for (std::size_t i = 0; i < dest_ranks.size(); ++i)
      dest_proc_to_neighbor.insert({dest_ranks[i], i});

    // Compute size of data to send to each process
    std::vector<std::int32_t> counter(dest_ranks.size(), 0);
    const std::vector<std::int64_t>& ghosts = entity_map->ghosts();
    std::vector<int> ghost_owner_rank = entity_map->ghost_owner_rank();

    for (std::size_t i = 0; i < ghost_entities.size(); ++i)
    {
      // entity local index
      int entity = ghost_entities[i];
      // entity local index in ghost range
      int entity_pos = ghost_entities[i] - entity_map->size_local();

      const auto it = dest_proc_to_neighbor.find(ghost_owner_rank[entity_pos]);
      assert(it != dest_proc_to_neighbor.end());

      auto cells = entity_cell->links(entity);
      // count number of entities per owner
      counter[it->second] += cells.size() + 2;
    }
  }

  //-------------------------
  // Fourth communication step: [owner->ghost]
  // Entity owner send entity-cell connectivity data plus cell ownership to
  // requesting process
  // [entity num_cells owner_0 cell0 ...]

  //-------------------------
  // Fifth communication step: [ghost->owner]
  // Compute new ghosts cells and inform owners

  // // Ghost to owner communicator
  // MPI_Comm comm = map->comm(common::IndexMap::Direction::reverse);
  // const auto dest_ranks = dolfinx::MPI::neighbors(comm)[1];

  // // Global-to-neigbourhood map for destination ranks
  // std::map<int, std::int32_t> dest_proc_to_neighbor;
  // for (std::size_t i = 0; i < dest_ranks.size(); ++i)
  //   dest_proc_to_neighbor.insert({dest_ranks[i], i});

  // // Get the owner of interface entities
  // std::shared_ptr<const common::IndexMap> map = topology.index_map(dim);
  // std::int32_t local_size = map->size_local();
  // std::vector<int> ghost_owner_rank = map->ghost_owner_rank();

  // // Get first ghost entity, ignore owned entities for now
  // auto ghost_begin
  //     = std::lower_bound(entities.begin(), entities.end(), local_size);
  // std::int32_t num_owned = std::distance(entities.begin(), ghost_begin);
  // std::int32_t num_ghost_entities = entities.size() - num_owned;

  // // Ghost to owner communicator
  // MPI_Comm comm = map->comm(common::IndexMap::Direction::reverse);
  // const auto dest_ranks = dolfinx::MPI::neighbors(comm)[1];

  // // Global-to-neigbourhood map for destination ranks
  // std::map<int, std::int32_t> dest_proc_to_neighbor;
  // for (std::size_t i = 0; i < dest_ranks.size(); ++i)
  //   dest_proc_to_neighbor.insert({dest_ranks[i], i});

  // // Compute size of data to send to each process
  // std::vector<std::int32_t> counter(dest_ranks.size(), 0);
  // std::vector<int> ghost_to_neighbour_rank(num_ghost_entities, -1);
  // const std::vector<std::int64_t>& ghosts = map->ghosts();
  // std::vector<std::int64_t> ghost_data(num_ghost_entities);

  // for (std::int32_t i = 0; i < num_ghost_entities; ++i)
  // {
  //   int entity = entities[num_owned + i];
  //   ghost_data[i] = ghosts[entity];
  //   const auto it = dest_proc_to_neighbor.find(ghost_owner_rank[entity]);
  //   assert(it != dest_proc_to_neighbor.end());
  //   ghost_to_neighbour_rank[i] = it->second;
  //   counter[ghost_to_neighbour_rank[i]]++;
  // }

  // std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  // std::partial_sum(counter.begin(), counter.end(),
  //                  std::next(send_disp.begin(), 1));

  // // Create and communicate adjacency list to neighborhood
  // const graph::AdjacencyList<std::int64_t>
  // ghost_data_out(std::move(ghost_data),
  //                                                         std::move(send_disp));
  // const graph::AdjacencyList<std::int64_t> ghost_data_in
  //     = dolfinx::MPI::neighbor_all_to_all(comm, ghost_data_out);

  // // Get list of all onwed entities that need cell connectivity information
  // std::vector<std::int32_t> owned_entities;
  // {
  //   std::vector<std::int64_t> global_entities = ghost_data_in.array();
  //   std::vector<std::int32_t> local_entities(global_entities.size());
  //   map->global_to_local(global_entities, local_entities);
  //   dolfinx::radix_sort<std::int32_t>(local_entities);
  //   owned_entities.reserve(local_entities.size() + num_owned);
  //   std::merge(local_entities.begin(), local_entities.end(),
  //   entities.begin(),
  //              std::next(entities.begin(), num_owned),
  //              std::back_inserter(owned_entities));
  //   owned_entities.erase(
  //       std::unique(owned_entities.begin(), owned_entities.end()),
  //       owned_entities.end());
  // }

  // std::vector<short int> remote_entities_bool(ghosts.size());
  // std::vector<short int> entity_needs_info(local_size);
  // map->scatter_fwd<short int>(entity_needs_info, remote_entities_bool, 1);

  // std::int32_t num_received_entities
  //     = std::reduce(remote_entities_bool.begin(), remote_entities_bool.end(),
  //                   std::int32_t(0), std::plus<std::int32_t>());

  // std::vector<std::int32_t> remote_entities(num_received_entities);
  // {
  //   std::int32_t pos = 0;
  //   for (auto it = remote_entities_bool.begin();
  //        it != remote_entities_bool.end(); it++)
  //   {
  //     if (*it)
  //     {
  //       std::ptrdiff_t entity = std::distance(remote_entities_bool.begin(),
  //       it); remote_entities[pos++] = static_cast<std::int32_t>(entity);
  //     }
  //   }
  // }

  // // Get all cells incident to remote entities
  // std::shared_ptr<const dolfinx::graph::AdjacencyList<int32_t>> entity_cell
  //     = topology.connectivity(dim, tdim);

  // std::vector<std::int32_t> num_incident_cells(remote_entities.size(), 0);
  // for (std::size_t i = 0; i < remote_entities.size(); i++)
  // {
  //   auto&& cells = entity_cell->links(remote_entities[i]);
  //   num_incident_cells[i] = cells.size();
  // }

  // std::vector<int> cells_offsets(remote_entities.size() + 1, 0);
  // std::partial_sum(num_incident_cells.begin(), num_incident_cells.end(),
  //                  std::next(cells_offsets.begin(), 1));

  return mesh;
}

//-----------------------------------------------------------------------------
