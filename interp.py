import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=200)


def get_node_coords(element, edge_conn, elem_edge_conn, X):
    # Extract the edges
    E1 = elem_edge_conn[element][0]
    E2 = elem_edge_conn[element][1]
    E3 = elem_edge_conn[element][2]

    nodes_edge1 = edge_conn[E1]
    nodes_edge2 = edge_conn[E2]
    nodes_edge3 = edge_conn[E3]

    # Comine the edges based on the elem_edge_conn
    nodes = [
        nodes_edge1[0],
        nodes_edge1[1],
        nodes_edge2[0],
        nodes_edge2[1],
        nodes_edge3[0],
        nodes_edge3[1],
    ]

    # Extract the unique node tags in the order of the edge connectivity
    n1_tag = nodes_edge1[0]
    n2_tag = nodes_edge2[0]
    n3_tag = nodes_edge3[0]

    # Extract the coordinates
    n1_coord = X[n1_tag]
    n2_coord = X[n2_tag]
    n3_coord = X[n3_tag]
    return n1_coord, n2_coord, n3_coord


def TriangleGuassianQuadrature(element, edge_conn, elem_edge_conn, X, idx, idy):
    # Get the node coordinates for the element.
    n1_coord, n2_coord, n3_coord = get_node_coords(
        element, edge_conn, elem_edge_conn, X
    )

    pts = np.array([[1 / 3, 1 / 3, 1 / 3]])
    wts = np.array([1.0])
    I = 0.0
    for m in range(len(wts)):
        wt = wts[m]

        L = pts[m]
        L1 = L[0]
        L2 = L[1]
        L3 = L[2]

        # Extract x and y values for each node
        x1 = n1_coord[0]
        y1 = n1_coord[1]

        x2 = n2_coord[0]
        y2 = n2_coord[1]

        x3 = n3_coord[0]
        y3 = n3_coord[1]

        # Compute the point to evalute the shape functions at
        x = x1 * L1 + x2 * L2 + x3 * L3
        y = y1 * L1 + y2 * L2 + y3 * L3

        # Compute the shape stuff
        N1, N2, N3, curlN1, curlN2, curlN3, area = eval_shape_funcs(
            n1_coord, n2_coord, n3_coord, x, y
        )

        # Accumulate
        N = [N1, N2, N3]
        curlN = [curlN1, curlN2, curlN3]

        I += wt * (curlN[idx] * curlN[idy] + np.dot(N[idx], N[idy])) * area
    return I


def eval_shape_funcs(n1_coord, n2_coord, n3_coord, x, y):
    # Extract x and y values for each node
    x1 = n1_coord[0]
    y1 = n1_coord[1]

    x2 = n2_coord[0]
    y2 = n2_coord[1]

    x3 = n3_coord[0]
    y3 = n3_coord[1]

    # Element coefficiencts
    a1 = x2 * y3 - y2 * x3
    a2 = x3 * y1 - y3 * x1
    a3 = x1 * y2 - y1 * x2

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    # Area of an element
    area = 0.5 * (b1 * c2 - b2 * c1)
    assert area > 0

    # Compute the length of each edge
    length = lambda x1, y1, x2, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l1 = length(x1, y1, x2, y2)
    l2 = length(x2, y2, x3, y3)
    l3 = length(x3, y3, x1, y1)

    # Collect information and return them
    a = [a1, a2, a3]  # a term coeffs
    b = [b1, b2, b3]  # b term coeffs
    c = [c1, c2, c3]  # c term coeffs
    l = [l1, l2, l3]  # lengths

    # Area coordinate derivatives
    gradL1 = np.array([b[0], c[0]])  # ∇L_1
    gradL2 = np.array([b[1], c[1]])  # ∇L_2
    gradL3 = np.array([b[2], c[2]])  # ∇L_3

    # Area coordinate function eval at point (x,y)
    alpha = 1 / (2 * area)
    L1 = alpha * (a[0] + b[0] * x + c[0] * y)
    L2 = alpha * (a[1] + b[1] * x + c[1] * y)
    L3 = alpha * (a[2] + b[2] * x + c[2] * y)

    # Compute the shape function
    N1 = (L1 * gradL2 - L2 * gradL1) * l[0]
    N2 = (L2 * gradL3 - L3 * gradL2) * l[1]
    N3 = (L3 * gradL1 - L1 * gradL3) * l[2]

    # Compute the curl of the shape functions
    curlN1 = (l[0] / (2 * area)) * (
        (b[0] * c[1] - b[1] * c[0]) - (c[0] * b[1] - c[1] * b[0])
    )

    curlN2 = (l[1] / (2 * area)) * (
        (b[1] * c[2] - b[2] * c[1]) - (c[1] * b[2] - c[2] * b[1])
    )

    curlN3 = (l[2] / (2 * area)) * (
        (b[2] * c[0] - b[0] * c[2]) - (c[2] * b[0] - c[0] * b[2])
    )

    return N1, N2, N3, curlN1, curlN2, curlN3, area


# Barycenter sampling
def barycenter_sampling(n1_coord, n2_coord, n3_coord):
    """
    Reference element: new_point = (x, y)
    P1 = (x,y) for node 1
    P2 = (x,y) for node 2
    P3 = (x,y) for node 3
    """
    P1 = n1_coord
    P2 = n2_coord
    P3 = n3_coord

    r1 = np.random.rand()
    r2 = np.random.rand()
    # Ensure uniform distribution in triangle
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    new_point = (1 - r1 - r2) * P1 + r1 * P2 + r2 * P3
    return new_point[0], new_point[1]  # xi and eta frame


def visualize_element_basis(element, edge_conn, elem_edge_conn, X, title=None):
    # Get the node coordinates for the element.
    n1_coord, n2_coord, n3_coord = get_node_coords(
        element, edge_conn, elem_edge_conn, X
    )

    # Initialize plot
    fig, ax = plt.subplots(ncols=3)

    # Plot the mesh
    triang = tri.Triangulation(
        [n1_coord[0], n2_coord[0], n3_coord[0]],
        [n1_coord[1], n2_coord[1], n3_coord[1]],
        [[0, 1, 2]],
    )
    ax[0].triplot(triang, "bo--")
    ax[1].triplot(triang, "go--")
    ax[2].triplot(triang, "ro--")

    # Compute the vector field inside the element
    npts = 100  # total number of points in the vector field
    for i in range(npts):
        # Get the x, y point in the triangle to evaluate the shape function
        x, y = barycenter_sampling(n1_coord, n2_coord, n3_coord)

        # Evaluate the shape funcs
        N1, N2, N3, curlN1, curlN2, curlN3, area = eval_shape_funcs(
            n1_coord, n2_coord, n3_coord, x, y
        )

        # Update the plot
        ax[0].quiver(x, y, N1[0], N1[1], width=4e-3)
        ax[1].quiver(x, y, N2[0], N2[1], width=4e-3)
        ax[2].quiver(x, y, N3[0], N3[1], width=4e-3)

    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[2].set_aspect("equal")
    ax[0].set_title(r"$N_1$")
    ax[1].set_title(r"$N_2$")
    ax[2].set_title(r"$N_3$")
    fig.suptitle(title)
    plt.savefig(title + ".jpg", dpi=800)
    return


def plot_element_solution(
    element,
    edge_conn,
    elem_edge_conn,
    X,
    soln,
    fix,
    ax,
    signs=[1.0, 1.0, 1.0],
):
    # Get the node coordinates for the element.
    n1_coord, n2_coord, n3_coord = get_node_coords(
        element, edge_conn, elem_edge_conn, X
    )

    # Plot the element mesh
    triang = tri.Triangulation(
        [n1_coord[0], n2_coord[0], n3_coord[0]],
        [n1_coord[1], n2_coord[1], n3_coord[1]],
        [[0, 1, 2]],
    )
    ax.triplot(triang, color="grey", linestyle="-", linewidth=1.0)

    # Compute the vector field inside the element
    npts = 150  # total number of points in the vector field
    for i in range(npts):
        # Get the x, y point in the triangle to evaluate the shape function
        x, y = barycenter_sampling(n1_coord, n2_coord, n3_coord)

        # Evaluate the shape funcs
        N1, N2, N3, curlN1, curlN2, curlN3, area = eval_shape_funcs(
            n1_coord, n2_coord, n3_coord, x, y
        )

        E = signs[0] * N1 * soln[0] + signs[1] * N2 * soln[1] + signs[2] * N3 * soln[2]

        # Update the plot
        ax.quiver(x, y, E[0], E[1], width=4e-3)

    ax.set_aspect("equal")
    return


def plot_mesh(X, elem_conn, edge_conn, elem_edge_conn):
    fig, ax = plt.subplots()
    triang = tri.Triangulation(X[:, 0], X[:, 1], elem_conn)
    ax.triplot(triang, color="grey", linestyle="--", linewidth=1.0)

    # Plot the node tags for each element
    for e in range(len(elem_conn)):
        elem_node_tags = elem_conn[e]
        n1_tag = elem_node_tags[0]
        n2_tag = elem_node_tags[1]
        n3_tag = elem_node_tags[2]
        plt.text(X[n1_tag, 0], X[n1_tag, 1], f"{n1_tag}")
        plt.text(X[n2_tag, 0], X[n2_tag, 1], f"{n2_tag}")
        plt.text(X[n3_tag, 0], X[n3_tag, 1], f"{n3_tag}")

    # Plot the edges for each element
    for e in range(len(elem_conn)):
        edge_tags = elem_edge_conn[e]

        for edge_i in edge_tags:
            n1_tag = edge_conn[edge_i][0]  # Start node for edge
            n2_tag = edge_conn[edge_i][1]  # End node for edge
            x1 = X[n1_tag, 0]
            y1 = X[n1_tag, 1]
            x2 = X[n2_tag, 0]
            y2 = X[n2_tag, 1]
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            plt.text(xm, ym, f"\nE{edge_i}")

    plt.savefig("mesh.jpg", dpi=1000)

    return


if __name__ == "__main__":
    # coords
    X = np.array(
        [
            [0.0, 0.0],  # node 0
            [1.0, 0.0],  # node 1
            [0.0, 1.0],  # node 2
            [1.0, 1.0],  # node 3
            [2.0, 1.0],  # node 4
        ]
    )

    # elem -> node
    elem_conn = [
        [1, 2, 0],  # e0
        [3, 2, 1],  # e1
        [1, 4, 3],  # e2
    ]

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
    ]

    elem_edge_conn = [
        [2, 0, 1],
        [5, 3, 4],
        [7, 8, 6],
    ]

    # Plot the mesh and the connectivity
    plot_mesh(X, elem_conn, edge_conn, elem_edge_conn)

    # Assemble stiffness matrix
    K = np.zeros((9, 9))
    # Loop through each element
    for e in range(len(elem_conn)):
        visualize_element_basis(
            e,
            edge_conn,
            elem_edge_conn,
            X,
            title=f"Element {e} H(curl)",
        )

        for i in range(3):
            row = elem_edge_conn[e][i]
            for j in range(3):
                col = elem_edge_conn[e][j]
                K[row, col] += TriangleGuassianQuadrature(
                    e, edge_conn, elem_edge_conn, X, i, j
                )
    print("\nStiffness Matrix (No BCs)")
    print(K)

    # Define G
    G = np.array(
        [
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
        ]
    )

    # New system
    E = np.zeros((11, 11))
    E[0:9, 0:9] = K
    E[0:9, [-1, -2]] = G.T
    E[[-1, -2], 0:9] = G
    rhs = np.zeros(11)

    print("\nMatrix E w/ Lagrange Multipliers")
    print(E)

    # Apply boundary condition to E
    # bc = 0
    # E[bc, :] = 0.0
    # E[bc, bc] = 1.0
    # rhs[bc] = 0.0

    # E[0, :] = 0.0
    # E[0, 0] = 1.0
    # rhs[0] = 1.0

    E[2, :] = 0.0
    E[2, 2] = 1.0
    rhs[2] = 1.0

    E[4, :] = 0.0
    E[4, 4] = 1.0
    rhs[4] = 1.0

    # E[7, :] = 0.0
    # E[7, 7] = 1.0
    # rhs[7] = 0.0

    E[8, :] = 0.0
    E[8, 8] = 1.0
    rhs[8] = 1.0

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
    soln1 = u[elem_edge_conn[0]]
    soln2 = u[elem_edge_conn[1]]
    soln3 = u[elem_edge_conn[2]]
    plot_element_solution(
        0, edge_conn, elem_edge_conn, X, soln1, fig1, ax1, signs=[1, 1, 1]
    )
    plot_element_solution(
        1, edge_conn, elem_edge_conn, X, soln2, fig1, ax1, signs=[1, 1, 1]
    )
    plot_element_solution(
        2, edge_conn, elem_edge_conn, X, soln3, fig1, ax1, signs=[1, 1, 1]
    )
    plt.savefig("fem.jpg", dpi=800)
    # plt.show()
