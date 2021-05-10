# Similar test as in the Stokes Taylor-Hood demo
import dolfinx
from mpi4py import MPI
import ufl
import dolfinx.io
import numpy as np


def noslip_boundary(x):
    # Function to mark x = 0, x = 1 and y = 0
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))


def lid(x):
    # Function to mark the lid (y = 1)
    return np.isclose(x[1], 1.0)


def lid_velocity_expression(x):
    # Lid velocity
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
B = ufl.VectorElement("Bubble", mesh.ufl_cell(), mesh.topology.dim + 1)
VP1 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
# V = ufl.VectorElement(ufl.NodalEnrichedElement(P1, B))
V = VP1 + B
Q = P1
W = dolfinx.FunctionSpace(mesh, ufl.MixedElement([V, Q]))

W0 = W.sub(0).collapse()

# No slip boundary condition
noslip = dolfinx.Function(V)
facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, noslip_boundary)
dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, facets)
bc0 = dolfinx.DirichletBC(noslip, dofs, W.sub(0))


# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = dolfinx.Function(W0)
lid_velocity.interpolate(lid_velocity_expression)
facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lid)
dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), 1, facets)
bc1 = dolfinx.DirichletBC(lid_velocity, dofs, W.sub(0))


# Since for this problem the pressure is only determined up to a
# constant, we pin the pressure at the point (0, 0)
zero = dolfinx.Function(Q)
with zero.vector.localForm() as zero_local:
    zero_local.set(0.0)
dofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Q),
                                           lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
bc2 = dolfinx.DirichletBC(zero, dofs, W.sub(1))


# Collect Dirichlet boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = dolfinx.Function(W0)
zero = dolfinx.Constant(mesh, 0.0)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(p, ufl.div(v)) + ufl.inner(ufl.div(u), q)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

problem = dolfinx.fem.LinearProblem(a, L, bcs)
w = problem.solve()

# Split the mixed solution and collapse
u = w.sub(0).collapse()
p = w.sub(1).collapse()

# Compute norms
norm_u = u.vector.norm()
norm_p = p.vector.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(D) Norm of velocity coefficient vector (monolithic, direct): {}".format(norm_u))
    print("(D) Norm of pressure coefficient vector (monolithic, direct): {}".format(norm_p))

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u)
