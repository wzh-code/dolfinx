import dolfinx
from mpi4py import MPI
import numpy as np
from IPython import embed


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
indices = np.arange(num_cells, dtype=np.int32)
bools = np.zeros(num_cells, dtype=bool)
bools[1] = True
bools[3] = True
values = indices[bools]
ct = dolfinx.MeshTags(mesh, tdim, values, values)
# mv = dolfinx.cpp.mesh.MeshView(ct)
# top = mv.topology

# V = dolfinx.FunctionSpace(mv, ("CG", 1))

class MeshViewPy():
    def __init__(self, meshtag: dolfinx.MeshTags):
        mesh = meshtag.mesh
        mesh.topology.create_connectivity(meshtag.dim, 0)
        e_to_v = mesh.topology.connectivity(meshtag.dim, 0)

        data= []
        offset = [0]

        num_entities = e_to_v.num_nodes
        is_entity = np.zeros(num_entities, dtype=bool)

        is_entity[meshtag.indices] = True

        for i in range(num_entities):
            if is_entity[i]:
                vertices = e_to_v.links(i)
                for vertex in vertices:
                    data.append(vertex)
            offset.append(len(data))

        v_to_v = mesh.topology.connectivity(0, 0)
        c_map = mesh.topology.index_map(meshtag.dim)
        v_map = mesh.topology.index_map(0)
        new_connectivity = dolfinx.cpp.graph.AdjacencyList_int32(
            np.array(data, dtype=np.int32), np.array(offset, dtype=np.int32))

        # Create topology of MeshView
        new_top = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, mesh.topology.cell_type)

        # Create vertex to vertex map (is identity)
        new_top.set_connectivity(v_to_v, 0, 0)
        # Set Index map for vertices
        new_top.set_index_map(0, v_map)

        # Set mesh connectivity for cells and index map for cells
        new_top.set_connectivity(new_connectivity, meshtag.dim, 0)
        new_top.set_index_map(meshtag.dim, c_map)
        self.topology = new_top
        self.mesh = mesh
        self.cell_map = meshtag.indices
        self.dim = meshtag.dim


mv = MeshViewPy(ct)

print("HI")
# mv = MeshViewPy(ct)
# num_cells_local = mv.topology.index_map(mv.dim).size_local
# for cell in range(num_cells_local):
#     print(cell, mv.cell_map[cell], mv.mesh.topology.connectivity(2, 0).links(mv.cell_map[cell]))
# # embed()
# mesh.topology.create_connectivity(ct.dim, 0)
# c_to_v = mesh.topology.connectivity(ct.dim, 0)
# print(c_to_v)

# new_top.create_connectivity(tdim - 1, 0)
# new_f_to_v = new_top.connectivity(tdim - 1, 0)
# print(new_f_to_v)
