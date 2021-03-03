# Copyright (C) 2021 Joe Dean & Matthew W. Scroggs
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

from sympy import Curve, line_integrate
from sympy.abc import x, y, t

import dolfinx
import numpy as np
from dolfinx import Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from mpi4py import MPI
from ufl import dS, ds, inner
from dolfinx.fem.assemble import assemble_scalar


@pytest.mark.parametrize("k", [1, 2, 3])
def test_facet_space(k):
    mesh_skeleton = []
    # Top
    mesh_skeleton.append(Curve([t, 1], (t, 0, 1)))
    # Bottom
    mesh_skeleton.append(Curve([t, 0], (t, 0, 1)))
    # Left
    mesh_skeleton.append(Curve([0, t], (t, 0, 1)))
    # Right
    mesh_skeleton.append(Curve([1, t], (t, 0, 1)))
    # Diagonal
    mesh_skeleton.append(Curve([t, t], (t, 0, 1)))

    integral = 0
    for facet in mesh_skeleton:
        integral += line_integrate(x**k * x**k, facet, [x, y])
    integral = float(integral.evalf())
    print(integral)

    # FEniCS
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [1, 1],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
        diagonal="right")

    V = FunctionSpace(mesh, ("DGT", k))
    f = Function(V)

    f.interpolate(lambda x: x[0]**k)

    print(f.vector[:])

    # FIXME inner(f("-"), f("-")) * ds seems problematic and isn't needed.
    # The notation also doesn't make much sense in this context
    # TODO Check if this causes problems in DG spaces
    integral_h = mesh.mpi_comm().allreduce(assemble_scalar(
        inner(f("-"), f("-")) * (dS + ds)), op=MPI.SUM)
    print(integral)
    print(integral_h)

    assert np.isclose(integral, integral_h)


# TODO: remove this test once interpolation is working
def test_facet_space_with_manual_interpolation():
    k = 1

    mesh_skeleton = []
    # Top
    mesh_skeleton.append(Curve([t, 1], (t, 0, 1)))
    # Bottom
    mesh_skeleton.append(Curve([t, 0], (t, 0, 1)))
    # Left
    mesh_skeleton.append(Curve([0, t], (t, 0, 1)))
    # Right
    mesh_skeleton.append(Curve([1, t], (t, 0, 1)))
    # Diagonal
    mesh_skeleton.append(Curve([t, t], (t, 0, 1)))

    integral = 0
    for facet in mesh_skeleton:
        integral += line_integrate(x**k * x**k, facet, [x, y])
    integral = float(integral.evalf())
    print(integral)

    # FEniCS
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [1, 1],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
        diagonal="right")

    V = FunctionSpace(mesh, ("DGT", k))
    f = Function(V)

    mesh.topology.create_connectivity(1, 0)
    c = mesh.topology.connectivity(1, 0)

    data = np.zeros(f.vector[:].shape)
    for edge in range(c.num_nodes):
        vs = c.links(edge)
        for v, dof in zip(vs, V.dofmap.cell_dofs(edge)):
            data[dof] = mesh.geometry.x[v][0]
    f.vector[:] = data

    print(f.vector[:])

    # FIXME f("-") etc. notation doesn't really make sense here.
    integral_h = mesh.mpi_comm().allreduce(assemble_scalar(
        inner(f, f) * ds + inner(f("-"), f("-")) * dS), op=MPI.SUM)
    print(integral)
    print(integral_h)

    assert np.isclose(integral, integral_h)
