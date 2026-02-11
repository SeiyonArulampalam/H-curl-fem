import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from collections import defaultdict
import parser
import plate_mesh
import curl_utils as Hcurl

np.set_printoptions(precision=2, linewidth=200)

# Create the plate mesh in gmsh
plate_mesh.create(lc=0.3)

# Parse the mesh
p = parser.InpParser()
p.parse_inp("plate.inp")

# Get the node locations
X = p.get_nodes()

# Get element connectivity
conn_surface1 = p.get_conn("SURFACE1", "CPS3")

# Get the edge connectivity for each surface
edge_node_conn_surf1, elem_edge_conn_surf1 = p.get_edge_conn("SURFACE1", "CPS3")

elem_conn = conn_surface1
edge_node_conn = edge_node_conn_surf1
elem_edge_conn = elem_edge_conn_surf1


# Get nodes on an edge
edge0_node_tags = p.get_conn("LINE1", "T3D2").tolist()
edge1_node_tags = p.get_conn("LINE2", "T3D2").tolist()
edge2_node_tags = p.get_conn("LINE3", "T3D2").tolist()
edge3_node_tags = p.get_conn("LINE4", "T3D2").tolist()


# Get the edges
edge0_tags = [edge_node_conn.index(e) for e in edge0_node_tags]
edge1_tags = [edge_node_conn.index(e) for e in edge1_node_tags]
edge2_tags = [edge_node_conn.index(e) for e in edge2_node_tags]
edge3_tags = [edge_node_conn.index(e) for e in edge3_node_tags]


# Plot the mesh and the connectivity
# Hcurl.plot_mesh(X, elem_conn, edge_node_conn, elem_edge_conn, "plate_fem_mesh")

# Assemble stiffness matrix
nelems = len(elem_conn)
K = np.zeros((3 * nelems, 3 * nelems))
# Loop through each element
for e in range(len(elem_conn)):
    # Compute the local stiffness matrix (3x3)
    for i in range(3):
        row = elem_edge_conn[e][i]
        for j in range(3):
            col = elem_edge_conn[e][j]
            K[row, col] += Hcurl.TriangleGuassianQuadrature(
                e, edge_node_conn, elem_edge_conn, X, i, j
            )

# Define shared edge tags
shared_edges = Hcurl.get_shared_edge_tags(edge_node_conn)
num_shared_edges = len(shared_edges)

constraint_edges = []
for key, val in shared_edges.items():
    constraint_edges.append(val)


G = np.zeros((num_shared_edges, len(edge_node_conn)))
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
rhs = np.zeros(3 * nelems + num_shared_edges)

# Apply boundary condition to E
bcs = [
    [edge0_tags, -2.0],
    # [edge1_tags, 0.0],
    # [edge2_tags, 1.0],
    # [edge3_tags, 0.0],
]

for bc in bcs:
    for t in bc[0]:
        t = bc[0]  # tag
        v = bc[1]  # value
        E[t, :] = 0.0
        E[t, t] = 1.0
        rhs[t] = v


# Solve the problem
u = np.linalg.solve(E, rhs)

# Plot element by element
# fig1, ax1 = plt.subplots()
# for e in range(len(elem_conn)):
#     Hcurl.plot_element_solution(
#         e,
#         edge_node_conn,
#         elem_edge_conn,
#         X,
#         u[elem_edge_conn[e]],
#         fig1,
#         ax1,
#         npts=1,
#     )
# plt.savefig("plate_fem.jpg", dpi=800)

# Plot global solution preserving the vector lengths
Hcurl.plot_vector_field(elem_conn, edge_node_conn, elem_edge_conn, X, u, "plate_fem")
