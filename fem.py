import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import curl_utils as Hcurl

# coords
X = np.array(
    [
        [0.0, 0.0],  # node 0
        [1.0, 1.0],  # node 1
        [0.0, 2.5],  # node 2
        [1.0, 2.5],  # node 3
        [2.0, 2.5],  # node 4
        [2.0, 1.0],  # node 5
        [3.0, 2.5],  # node 6
        [3.0, 0.0],  # node 7
    ]
)

# elem -> node
elem_conn = [
    [1, 2, 0],  # e0
    [3, 2, 1],  # e1
    [1, 4, 3],  # e2
    [1, 5, 4],  # e3
    [5, 6, 4],  # e4
    [5, 7, 6],  # e5
]

# edge -> node
edge_conn = [
    [0, 1],  # E0
    [1, 2],  # E1
    [2, 0],  # E2
    [1, 3],  # E3
    [3, 2],  # E4
    [2, 1],  # E5
    [3, 1],  # E6
    [1, 4],  # E7
    [4, 3],  # E8
    [1, 5],  # E9
    [5, 4],  # E10
    [4, 1],  # E11
    [5, 6],  # E12
    [6, 4],  # E13
    [4, 5],  # E14
    [6, 5],  # E15
    [5, 7],  # E16
    [7, 6],  # E17
]

# elem -> edge
elem_edge_conn = [
    [2, 0, 1],  # e0
    [5, 3, 4],  # e1
    [7, 8, 6],  # e2
    [9, 10, 11],  # e3
    [12, 13, 14],  # e4
    [15, 16, 17],  # e5
]

# Plot the mesh and the connectivity
Hcurl.plot_mesh(X, elem_conn, edge_conn, elem_edge_conn, "simple_fem_mesh")

# Assemble stiffness matrix
nelems = len(elem_conn)
K = np.zeros((3 * nelems, 3 * nelems))
# Loop through each element
for e in range(len(elem_conn)):
    Hcurl.visualize_element_basis(
        e,
        edge_conn,
        elem_edge_conn,
        X,
        title=f"Element {e} H(curl)",
    )

    # Compute the local stiffness matrix (3x3)
    for i in range(3):
        row = elem_edge_conn[e][i]
        for j in range(3):
            col = elem_edge_conn[e][j]
            K[row, col] += Hcurl.TriangleGuassianQuadrature(
                e, edge_conn, elem_edge_conn, X, i, j
            )
print("\nStiffness Matrix (No BCs)")
print(K)

# Define G (Constraint jacobian)
shared_edges = [[1, 5], [3, 6], [7, 11], [14, 10], [12, 15]]
num_shared_edges = len(shared_edges)
G = np.zeros((num_shared_edges, len(edge_conn)))
for i, pair in enumerate(shared_edges):
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

print("\nMatrix E w/ Lagrange Multipliers")
print(E)

# Apply boundary condition to E
bcs = [
    [8, 1.0],
    [4, 1.0],
    [13, 1.0],
    [0, -1.0],
    [9, -1.0],
    [16, -1.0],
]

for bc in bcs:
    t = bc[0]  # tag
    v = bc[1]  # value
    print(t, v)
    E[t, :] = 0.0
    E[t, t] = 1.0
    rhs[t] = v

print("\nMatrix E w/ BCs")
print(E)

print("\n RHS Vector")
print(rhs)
print()
u = np.linalg.solve(E, rhs)

for i, soln_i in enumerate(u):
    print(f"u[{i}] = {soln_i:4f}")

# Initialize plot
print()
fig1, ax1 = plt.subplots()
for e in range(len(elem_conn)):
    Hcurl.plot_element_solution(
        e,
        edge_conn,
        elem_edge_conn,
        X,
        u[elem_edge_conn[e]],
        fig1,
        ax1,
        npts=30,
    )
plt.savefig("simple_fem.jpg", dpi=800)
# plt.show()
