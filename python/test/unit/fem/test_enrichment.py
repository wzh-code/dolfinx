# Copyright (C) 2021 Matthew Scroggs & Jorgen Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from mpi4py import MPI
import ufl


def test_bubble_enrichment():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    lagrange = ufl.FiniteElement("P", ufl.triangle, 1)
    bubble = ufl.FiniteElement("B", ufl.triangle, 3)

    dolfinx.FunctionSpace(mesh, lagrange + bubble)


def test_bubble_vector_enrichment():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    lagrange = ufl.VectorElement("P", ufl.triangle, 1)
    bubble = ufl.VectorElement("B", ufl.triangle, 3)

    dolfinx.FunctionSpace(mesh, lagrange + bubble)
