// Copyright (C) 2021 Joseph Dean, Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "MeshView.h"
#include <dolfinx/common/IndexMap.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
using namespace dolfinx;
using namespace dolfinx::mesh;

MeshView::MeshView(std::shared_ptr<const MeshTags<std::int32_t>> meshtag)
    : _mesh(meshtag->mesh()), _entities(meshtag->indices()),
      _dim(meshtag->dim()), _topology()
{

  // Get vertex
  _mesh->topology_mutable().create_connectivity(_dim, 0);
  auto e_to_v = _mesh->topology().connectivity(_dim, 0);
  assert(e_to_v);
  auto v_to_v = _mesh->topology().connectivity(0, 0);
  assert(v_to_v);
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> v_to_v_nonconst
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(*v_to_v);

  const std::int32_t num_mesh_entitites = e_to_v->num_nodes();
  std::vector<bool> is_entity(num_mesh_entitites);
  std::fill(is_entity.begin(), is_entity.end(), false);
  for (auto e : _entities)
    is_entity[e] = true;

  std::vector<std::int32_t> data;
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(_entities.size());
  for (int i = 0; i < num_mesh_entitites; ++i)
  {
    if (is_entity[i])
    {
      auto vertices = e_to_v->links(i);
      data.insert(data.end(), vertices.begin(), vertices.end());
    }
    offsets.push_back(data.size());
  }

  // Create Adjacency lists
  graph::AdjacencyList<std::int32_t> connectivity(data, offsets);

  // Create topology
  CellType entity_type = cell_entity_type(_mesh->topology().cell_type(), _dim);
  _topology = std::make_shared<Topology>(_mesh->mpi_comm(), entity_type);

  // Set index maps (vertex map and cell map)
  _topology->set_index_map(0, _mesh->topology().index_map(0));
  _topology->set_index_map(_dim, _mesh->topology().index_map(_dim));

  _topology->set_connectivity(v_to_v_nonconst, 0, 0);
  _topology->set_connectivity(
      std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity), _dim,
      0);
}

std::shared_ptr<mesh::Topology> MeshView::topology() { return _topology; }
std::shared_ptr<const Mesh> MeshView::mesh() { return _mesh; }
std::int32_t MeshView::dim() { return _dim; }
const std::vector<std::int32_t>& MeshView::entities() { return _entities; }