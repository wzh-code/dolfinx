# Copyright (C) 2011-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class"""

import importlib

import cffi
import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.geometry import (BoundingBoxTree, compute_colliding_cells,
                              compute_collisions)
from dolfinx.mesh import create_mesh, create_unit_cube
from dolfinx_utils.test.skips import skip_if_complex, skip_in_parallel

from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def W(mesh):
    return VectorFunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def Q(mesh):
    return TensorFunctionSpace(mesh, ('Lagrange', 1))


def test_name_argument(W):
    u = Function(W)
    v = Function(W, name="v")
    assert u.name == "f_{}".format(u.count())
    assert v.name == "v"
    assert str(v) == "v"


def test_compute_point_values(V, W, mesh):
    u = Function(V)
    u.x.array[:] = 1.0
    v = Function(W)
    v.x.array[:] = 1.0
    u_values = u.compute_point_values()
    v_values = v.compute_point_values()

    u_ones = np.ones_like(u_values, dtype=np.float64)
    assert np.all(np.isclose(u_values, u_ones))
    v_ones = np.ones_like(v_values, dtype=np.float64)
    assert np.all(np.isclose(v_values, v_ones))
    u_values2 = u.compute_point_values()
    assert all(u_values == u_values2)


def test_assign(V, W):
    for V_ in [V, W]:
        u = Function(V_)
        u0 = Function(V_)
        u1 = Function(V_)
        u2 = Function(V_)
        u.x.array[:] = 1
        u0.x.array[:] = 2
        u1.x.array[:] = 3
        u2.x.array[:] = 4

        # Test assign + scale
        uu = Function(V_)
        u.vector.copy(result=uu.vector)
        uu.vector.scale(2)
        assert uu.vector.array.sum() == u0.vector.array.sum()

        # Test complex assignment
        expr = 3 * u.vector - 4 * u1.vector - 0.1 * 4 * u.vector * 4 + u2.vector + 3 * u0.vector / 3. / 0.5
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr_scalar = 3 - 4 * 3 - 0.1 * 4 * 4 + 4. + 3 * 2. / 3. / 0.5

        expr.copy(result=uu.vector)
        uu.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert np.isclose(uu.vector.array.sum() - expr_scalar * uu.vector.local_size, 0)

        # Test self assignment
        expr = 3 * u.vector - 5.0 * u2.vector + u1.vector - 5 * u.vector
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr_scalar = 3 - 5 * 4. + 3. - 5
        expr.copy(result=u.vector)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert np.isclose(u.vector.array.sum() - expr_scalar * u.vector.local_size, 0)

        # Test zero assignment
        expr = -u2.vector / 2 + 2 * u1.vector - u1.vector / 0.5 + u2.vector * 0.5
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr.copy(result=u.vector)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert round(u.vector.array.sum() - 0.0, 7) == 0


def test_eval(V, W, Q, mesh):
    u1 = Function(V)
    u2 = Function(W)
    u3 = Function(Q)

    def e2(x):
        values = np.empty((3, x.shape[1]))
        values[0] = x[0] + x[1] + x[2]
        values[1] = x[0] - x[1] - x[2]
        values[2] = x[0] + x[1] + x[2]
        return values

    def e3(x):
        values = np.empty((9, x.shape[1]))
        values[0] = x[0] + x[1] + x[2]
        values[1] = x[0] - x[1] - x[2]
        values[2] = x[0] + x[1] + x[2]
        values[3] = x[0]
        values[4] = x[1]
        values[5] = x[2]
        values[6] = -x[0]
        values[7] = -x[1]
        values[8] = -x[2]
        return values

    u1.interpolate(lambda x: x[0] + x[1] + x[2])
    u2.interpolate(e2)
    u3.interpolate(e3)

    x0 = (mesh.geometry.x[0] + mesh.geometry.x[1]) / 2.0
    tree = BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions(tree, x0)
    cell = compute_colliding_cells(mesh, cell_candidates, x0)
    first_cell = cell[0]
    assert np.allclose(u3.eval(x0, first_cell)[:3], u2.eval(x0, first_cell), rtol=1e-15, atol=1e-15)


@skip_in_parallel
def test_eval_manifold():
    # Simple two-triangle surface in 3d
    vertices = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0,
                                                                    0.0)]
    cells = [(0, 1, 2), (0, 1, 3)]
    cell = ufl.Cell("triangle", geometric_dimension=3)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(Q)
    u.interpolate(lambda x: x[0] + x[1])
    assert np.isclose(u.eval([0.75, 0.25, 0.5], 0)[0], 1.0)


def test_interpolation_mismatch_rank0(W):
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones(x.shape[1]))


def test_interpolation_mismatch_rank1(W):
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones((2, x.shape[1])))


def test_mixed_element_interpolation():
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, ufl.MixedElement([el, el]))
    u = Function(V)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones(2, x.shape[1]))


def test_interpolation_rank0(V):
    class MyExpression:
        def __init__(self):
            self.t = 0.0

        def eval(self, x):
            return np.full(x.shape[1], self.t)

    f = MyExpression()
    f.t = 1.0
    w = Function(V)
    w.interpolate(f.eval)
    assert (w.x.array[:] == 1.0).all()

    num_vertices = V.mesh.topology.index_map(0).size_global
    assert np.isclose(w.vector.norm(PETSc.NormType.N1) - num_vertices, 0)

    f.t = 2.0
    w.interpolate(f.eval)
    assert (w.x.array[:] == 2.0).all()


def test_interpolation_rank1(W):
    def f(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 1.0
        values[1] = 1.0
        values[2] = 1.0
        return values

    w = Function(W)
    w.interpolate(f)
    x = w.vector
    assert x.max()[1] == 1.0
    assert x.min()[1] == 1.0

    num_vertices = W.mesh.topology.index_map(0).size_global
    assert round(w.vector.norm(PETSc.NormType.N1) - 3 * num_vertices, 7) == 0


@skip_if_complex
def test_cffi_expression(V):
    code_h = """
    void eval(double* values, int num_points, int value_size, const double* x);
    """

    code_c = """
    void eval(double* values, int num_points, int value_size, const double* x)
    {
      /* x0 + x1 */
      for (int i = 0; i < num_points; ++i)
        values[i  + 0] = x[i] + x[i + num_points];
    }
    """
    module = "_expr_eval" + str(MPI.COMM_WORLD.rank)

    # Build the kernel
    ffi = cffi.FFI()
    ffi.set_source(module, code_c)
    ffi.cdef(code_h)
    ffi.compile()

    # Import the compiled kernel
    kernel_mod = importlib.import_module(module)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Get pointer to the compiled function
    eval_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "eval"))

    # Handle C func address by hand
    f1 = Function(V)
    f1.interpolate(int(eval_ptr))

    f2 = Function(V)
    f2.interpolate(lambda x: x[0] + x[1])
    assert (f1.vector - f2.vector).norm() < 1.0e-12


def test_interpolation_function(mesh):
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.vector.set(1)
    Vh = FunctionSpace(mesh, ("Lagrange", 1))
    uh = Function(Vh)
    uh.interpolate(u)
    assert np.allclose(uh.vector.array, 1)


@skip_in_parallel
@pytest.mark.parametrize("dim", [2, 3])
def test_compute_point_values_manifold(dim):
    degree = 1
    cell = ufl.Cell("triangle", dim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree, dim=dim))
    if dim == 2:
        x = [[0., 0.], [0., 1.], [1., 1.]]
    else:
        x = [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.]]

    cells = [[0, 1, 2]]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    FE = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    FS = FunctionSpace(mesh, FE)
    fx = Function(FS)

    fx.interpolate(lambda x: x[0])
    ux = fx.compute_point_values().reshape(-1)
    dof_to_vertex = np.zeros(3, dtype=np.int32)
    for i in range(3):
        dof_to_vertex[i] = FS.dofmap.dof_layout.entity_dofs(0, i)[0]
    assert np.allclose(fx.x.array, ux[dof_to_vertex])
