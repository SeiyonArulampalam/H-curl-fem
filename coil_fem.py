import numpy as np
import parser
import multiregion_mesh
import curl_utils as Hcurl

np.set_printoptions(precision=2, linewidth=200)

mesh = multiregion_mesh.Mesh(lc=0.5, lc1=0.4)
mesh.create()

# Parse the mesh
p = parser.InpParser()
p.parse_inp("multiregion_mesh.inp")

# Get the node locations
X = p.get_nodes()

# Get element connectivity
conn_surface1 = p.get_conn("SURFACE1", "CPS3")  # Airgap
conn_surface2 = p.get_conn("SURFACE2", "CPS3")  # Magnet/Coil

# Get the edge connectivity for each surface
edge_node_conn_surf1, elem_edge_conn_surf1 = p.get_edge_conn(
    "SURFACE1",
    "CPS3",
    edge_counter=0,
)
edge_node_conn_surf2, elem_edge_conn_surf2 = p.get_edge_conn(
    "SURFACE2",
    "CPS3",
    edge_counter=len(edge_node_conn_surf1),
)

# Combine connectivities
elem_conn = np.concatenate((conn_surface1, conn_surface2)).tolist()
edge_node_conn = np.concatenate((edge_node_conn_surf1, edge_node_conn_surf2)).tolist()
elem_edge_conn = np.concatenate((elem_edge_conn_surf1, elem_edge_conn_surf2)).tolist()

# Plot the mesh and the connectivity
# Hcurl.plot_mesh(X, elem_conn, edge_node_conn, elem_edge_conn, "multiregion_mesh")

# Get nodes along the interface edges
edge5_node_tags = p.get_conn("LINE5", "T3D2").tolist()
edge6_node_tags = p.get_conn("LINE6", "T3D2").tolist()
edge7_node_tags = p.get_conn("LINE7", "T3D2").tolist()
edge8_node_tags = p.get_conn("LINE8", "T3D2").tolist()

# Get the edges
edge5_tags = [edge_node_conn.index(e) for e in edge5_node_tags]
edge6_tags = [edge_node_conn.index(e) for e in edge6_node_tags]
edge7_tags = [edge_node_conn.index(e) for e in edge7_node_tags]
edge8_tags = [edge_node_conn.index(e) for e in edge8_node_tags]

# Assemble the global stiffness matrix
nelems = len(elem_conn)
nedges = len(edge_node_conn)
K = np.zeros((nelems * 3, nelems * 3))
print(K.shape)
# Loop through each element
for e in range(len(elem_conn)):
    # Compute the local stiffness matrix (3x3)
    print(elem_edge_conn[e])
    for i in range(3):
        row = elem_edge_conn[e][i]
        for j in range(3):
            col = elem_edge_conn[e][j]
            print(row, col)
            K[row, col] += Hcurl.TriangleGuassianQuadrature(
                e, edge_node_conn, elem_edge_conn, X, i, j
            )

# Define shared edge tags
shared_edges = Hcurl.get_shared_edge_tags(edge_node_conn)
num_shared_edges = len(shared_edges)

constraint_edges = []
for key, val in shared_edges.items():
    constraint_edges.append(val)

G = np.zeros((num_shared_edges, 3 * nelems))
print(G.shape)
for i, pair in enumerate(constraint_edges):
    G[i, pair[0]] = 1.0
    G[i, pair[1]] = 1.0

# Finite Element System with Lagrange Multipliers.
# The coupled linear system is:
#     | K   G.T | | u | = | f |
#     | G   0   | | λ |   | 0 |
E = np.zeros((3 * nelems + num_shared_edges, 3 * nelems + num_shared_edges))
E[0 : 3 * nelems, 0 : 3 * nelems] = K
E[0 : 3 * nelems, -num_shared_edges:] = G.T
E[-num_shared_edges:, 0 : 3 * nelems] = G

# Loop through the elements in the magnet for the forcing component
rhs = np.zeros(3 * nelems + num_shared_edges)

# Define the current in a coil
I = 10.0

# Use Ampere's Law to apply a flux in the correct orientation
bcs = [
    [edge5_tags, I],
    [edge6_tags, I],
    [edge7_tags, I],
    [edge8_tags, I],
]

for bc in bcs:
    for t in bc[0]:
        t = bc[0]  # tag
        v = bc[1]  # value
        E[t, :] = 0.0
        E[t, t] = 1.0
        rhs[t] = v

# # Apply forcing functions to the rhs vector
# for e in range(len(conn_surface2)):
#     for i in range(3):
#         row = elem_edge_conn[e][i]
#         rhs[row] += Hcurl.TriangleGuassianQuadratureCoils(
#             Jz, e, edge_node_conn, elem_edge_conn, X, i
#         )
# for i in rhs:
#     print(i)
# Solve the problem
u = np.linalg.solve(E, rhs)

# Plot the results
Hcurl.plot_vector_field(elem_conn, edge_node_conn, elem_edge_conn, X, u, "coil_fem")
