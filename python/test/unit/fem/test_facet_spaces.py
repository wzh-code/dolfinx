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
from ufl import dS, ds, inner, TrialFunction, TestFunction
from dolfinx.fem.assemble import assemble_scalar
import ffcx
import cffi
import numba
from petsc4py import PETSc


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
    # FIXME Skipping interpolation and setting e.g. data[0] = 1 and rest to 0
    # above seems seems to indicate a bug in the assembly
    integral_h = mesh.mpi_comm().allreduce(assemble_scalar(
        inner(f, f) * ds + inner(f("-"), f("-")) * dS), op=MPI.SUM)
    print(integral)
    print(integral_h)

    assert np.isclose(integral, integral_h)


def sympy_element_tensor(facet, phi):
    n = len(phi)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = \
                float(line_integrate(phi[i] * phi[j], facet, [x, y]).evalf())
    return A


def test_facet_space_custom_kernel():
    # Compute tensor for each facet using sympy
    As_sympy = [sympy_element_tensor(Curve([t, 1 - t], (t, 0, 1)), [y, x]),
                sympy_element_tensor(Curve([t, 0], (t, 0, 1)), [1 - x, x]),
                sympy_element_tensor(Curve([0, t], (t, 0, 1)), [1 - y, y])]

    # Compute with FEniCS
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [1, 1],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
        diagonal="right")
    V = FunctionSpace(mesh, ("DGT", 1))
    u = TrialFunction(V)
    v = TestFunction(V)
    ele_space_dim = V.dolfin_element().space_dimension()
    nfacets = mesh.ufl_cell().num_facets()

    a = u * v * ds

    forms = [a]
    c_type, np_type = "double", np.float64
    compiled_forms, module = ffcx.codegeneration.jit.compile_forms(
        forms, parameters={"scalar_type": c_type})  # , cache_dir=".")

    ffi = cffi.FFI()
    integral_a_facet = \
        compiled_forms[0][0].create_exterior_facet_integral(-1).tabulate_tensor

    A = np.zeros((ele_space_dim, ele_space_dim), dtype=np_type)
    w = np.array([], dtype=np_type)
    c = np.array([], dtype=np_type)
    coords = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    facet = np.zeros((1), dtype=np.int32)
    # TODO Check this
    quad_perm = np.array([0], dtype=np.uint8)

    for i in range(nfacets):
        A.fill(0)
        facet[0] = i
        integral_a_facet(
            ffi.cast('{type} *'.format(type=c_type), A.ctypes.data),
            ffi.cast('{type} *'.format(type=c_type), w.ctypes.data),
            ffi.cast('{type} *'.format(type=c_type), c.ctypes.data),
            ffi.cast('double *', coords.ctypes.data),
            ffi.cast('int *', facet.ctypes.data),
            ffi.cast('uint8_t * ', quad_perm.ctypes.data),
            0)
        assert np.allclose(A, As_sympy[i])


def test_facet_space_custom_kernel_assemble():
    # TODO Try just assembling over facets
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [1, 1],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
        diagonal="right")
    V = FunctionSpace(mesh, ("DGT", 1))
    u = TrialFunction(V)
    v = TestFunction(V)
    ele_space_dim = V.dolfin_element().space_dimension()
    nfacets = mesh.ufl_cell().num_facets()

    # NOTE Assembling this as below (cellwise over boundaries) will
    # give a contribution from each element on the shared diagonal.
    a = u * v * ds

    forms = [a]
    c_type = "double"
    compiled_forms, module = ffcx.codegeneration.jit.compile_forms(
        forms, parameters={"scalar_type": c_type})

    ffi = cffi.FFI()
    integral_a_facet = \
        compiled_forms[0][0].create_exterior_facet_integral(-1).tabulate_tensor

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.int32))

    @numba.cfunc(c_signature, nopython=True)
    def tabulate_tensor_A(A_, w_, c_, coords_, entity_local_index,
                          cell_orientation):
        A = numba.carray(A_, (ele_space_dim, ele_space_dim),
                         dtype=PETSc.ScalarType)
        facet = np.zeros((1), dtype=np.int32)
        quad_perm = np.array([0], dtype=np.uint8)
        for i in range(nfacets):
            facet[0] = i
            # TODO Check what quad perm should be. Looking at auto-generated
            # code, the last parameter doesn't seem to be used
            integral_a_facet(ffi.from_buffer(A), w_, c_, coords_,
                             ffi.from_buffer(facet),
                             ffi.from_buffer(quad_perm), 0)

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor_A.address)], None)}
    a_form = dolfinx.cpp.fem.Form([V._cpp_object, V._cpp_object],
                                  integrals, [], [], False)
    A = dolfinx.fem.assemble_matrix(a_form)
    A.assemble()

    mesh.topology.create_connectivity_all()
    c_cell_facet = mesh.topology.connectivity(2, 1)
    c_facet_point = mesh.topology.connectivity(1, 0)

    # FIXME Horrible hacky temporary method
    # FIXME Don't specify size manually
    n = V.dofmap.index_map.size_global  # TODO Is this correct?
    A_sympy_assemble = np.zeros((n, n))
    for element in range(c_cell_facet.num_nodes):
        for facet in c_cell_facet.links(element):
            points = c_facet_point.links(facet)
            h_facet = np.linalg.norm(
                mesh.geometry.x[points[0]] - mesh.geometry.x[points[1]])
            if np.isclose(h_facet, 1):
                A_e_sym = sympy_element_tensor(
                    Curve([t, 0], (t, 0, 1)), [1 - x, x])
            else:
                A_e_sym = sympy_element_tensor(
                    Curve([t, 1 - t], (t, 0, 1)), [y, x])
            for i in range(2):
                for j in range(2):
                    dofs = V.dofmap.cell_dofs(facet)
                    A_sympy_assemble[dofs[i], dofs[j]] += A_e_sym[i, j]
    print("FEniCS:")
    print(A[:, :])
    print("Sympy:")
    print(A_sympy_assemble)
    assert(np.allclose(A[:, :], A_sympy_assemble))
