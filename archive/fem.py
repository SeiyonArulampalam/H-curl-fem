import numpy as np
import matplotlib.pyplot as plt
import H1, Hcurl
import plot

"""Main Script for a 2 element mesh example"""


def GaussQuadrature(wts, pts, row, col, x, y):
    I = 0.0
    for k in range(len(wts)):
        w = wts[k]
        xi = pts[k][0]
        eta = pts[k][1]
        _, _, _, _, detJ = H1.compute_detJ(xi, eta, x, y)

        # Local basis evaluation of the shape functions
        N1, N2, N3 = Hcurl.eval_shape_funcs(xi, eta)
        N = [N1, N2, N3]

        curl_N1, curl_N2, curl_N3 = Hcurl.eval_curl(xi, eta)
        curl = [curl_N1, curl_N2, curl_N3]

        comp1 = np.dot(curl[row], curl[col])  # (∇xNi)·(∇xNj)
        comp2 = 1e-8 * np.dot(N[row], N[col])  # Ni·Nj

        I += w * (comp1 + comp2)
    return I


# Quadrature
weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
points = np.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])

# Define the simple 2 element mesh
nodeTags = [0, 1, 2, 3]

# coords
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

# elem -> node
elem_conn = [
    [0, 1, 2],
    [1, 2, 3],
]


edge_conn = [
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 1],
]


elem_edge_conn = [
    [0, 1, 2],
    [3, 4, 1],
]

elem_edge_orient = [
    [1, 1, -1],
    [1, 1, 1],
]

# Loop through each edge and compute local stiffness matrix
n_edges = len(edge_conn)
E = np.zeros((n_edges, n_edges))

# Loop through each element
for e in range(len(elem_conn)):
    # Extract the element edge connectivity for element e
    edges = elem_edge_conn[e]
    sign = elem_edge_orient[e]

    # Extract the x and y coordinates for the element
    x = X[:, 0][elem_conn[e]]
    y = X[:, 1][elem_conn[e]]
    for i, row in enumerate(edges):
        s_i = sign[i]
        for j, col in enumerate(edges):
            s_j = sign[j]
            integral = GaussQuadrature(weights, points, i, j, x, y)
            E[row, col] += s_i * s_j * integral

# Create a fake rhs
rhs = np.zeros(len(edge_conn))

# Apply boundary condition
bc = 2
E[bc, :] = 0.0
E[bc, bc] = 1.0
rhs[bc] = 1.0

# Solve
u = np.linalg.solve(E, rhs)
print(E)
print(rhs)
print(u)

# Plot
fig, ax = plt.subplots()
plot.mesh(elem_conn, X, fig, ax)

for e in range(len(elem_conn)):
    # Extract nodal data
    n1 = elem_conn[e][0]
    n2 = elem_conn[e][1]
    n3 = elem_conn[e][2]

    n1_coords = X[n1]
    n2_coords = X[n2]
    n3_coords = X[n3]

    # Extract the solution for each edge in the triangle
    u_elem = u[elem_edge_conn[e]]

    for k in range(3):
        u_elem[k] *= elem_edge_orient[e][k]

    for i in range(300):
        # Compute a point in the reference element
        xi, eta = plot.barycenter_sampling_ref_element()

        # Compute the shape derivatives
        x = [n1_coords[0], n2_coords[0], n3_coords[0]]
        y = [n1_coords[1], n2_coords[1], n3_coords[1]]
        N, N_xi, N_ea, Nx, Ny, invJ = H1.compute_shape_derivs(xi, eta, x, y)

        # Compute the solution vector
        vec = Hcurl.compute_vector_field(xi, eta, u_elem, x, y)

        # Transform xi and eta to x,y points
        xpt, ypt = H1.transform(xi, eta, n1_coords, n2_coords, n3_coords)

        plot.vector_field(xpt, ypt, vec, fig=fig, ax=ax)

plt.savefig("fem.png", dpi=1000)
# plt.show()
