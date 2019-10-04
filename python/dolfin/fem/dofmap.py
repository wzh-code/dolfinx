# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


def make_fenics_finite_element(fenics_finite_element):
    """Returns finite element from a pointer"""
    return cpp.fem.make_fenics_finite_element(fenics_finite_element)


def make_fenics_dofmap(fenics_dofmap):
    """Returns dofmap from a pointer"""
    return cpp.fem.make_fenics_dofmap(fenics_dofmap)


def make_fenics_form(fenics_form):
    """Returns form from a pointer"""
    return cpp.fem.make_fenics_form(fenics_form)


def make_coordinate_mapping(fenics_coordinate_mapping):
    """Returns CoordinateMapping from a pointer to a fenics_coordinate_mapping"""
    return cpp.fem.make_coordinate_mapping(fenics_coordinate_mapping)


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a fenics_dofmap on a specific mesh.
    """

    def __init__(self, dofmap: cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        return self._cpp_object.cell_dofs(cell_index)

    def dofs(self, mesh, entity_dim: int):
        return self._cpp_object.dofs(mesh, entity_dim)

    def set(self, x, value):
        self._cpp_object.set(x, value)

    @property
    def dof_layout(self):
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        return self._cpp_object.index_map

    @property
    def dof_array(self):
        return self._cpp_object.dof_array()
