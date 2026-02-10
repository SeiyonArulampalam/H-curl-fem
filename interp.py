import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def TriangleGuassianQuadrature(n1_coord, n2_coord, n3_coord, idx, idy):
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


def visualize_element_basis(n1_coord, n2_coord, n3_coord):
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
    npts = 250  # total number of points in the vector field
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
    return


def plot_element_solution(n1_coord, n2_coord, n3_coord, soln):
    # Initialize plot
    fig, ax = plt.subplots()

    # Plot the mesh
    triang = tri.Triangulation(
        [n1_coord[0], n2_coord[0], n3_coord[0]],
        [n1_coord[1], n2_coord[1], n3_coord[1]],
        [[0, 1, 2]],
    )
    ax.triplot(triang, "bo--")

    # Compute the vector field inside the element
    npts = 250  # total number of points in the vector field
    for i in range(npts):
        # Get the x, y point in the triangle to evaluate the shape function
        x, y = barycenter_sampling(n1_coord, n2_coord, n3_coord)

        # Evaluate the shape funcs
        N1, N2, N3, curlN1, curlN2, curlN3, area = eval_shape_funcs(
            n1_coord, n2_coord, n3_coord, x, y
        )

        E = N1 * soln[0] + N2 * soln[1] + N3 * soln[2]

        # Update the plot
        ax.quiver(x, y, E[0], E[1], width=4e-3)

    ax.set_aspect("equal")
    ax.set_title(r"$soln$")
    return


if __name__ == "__main__":
    # Define mesh
    n1_coord = np.array([0.0, 0.0])
    n2_coord = np.array([1.0, 0.0])
    n3_coord = np.array([0.0, 1.0])

    # Compute the stiffness matrix
    E = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            E[row, col] += TriangleGuassianQuadrature(
                n1_coord, n2_coord, n3_coord, row, col
            )

    # Create a rhs
    rhs = np.zeros(3)

    # Apply dirichlet bc
    bc = 2
    E[bc, :] = 0.0
    E[bc, bc] = 1.0
    rhs[bc] = 1.0

    # Solve
    print(E)
    print(rhs)
    u = np.linalg.solve(E, rhs)
    print(u)

    plot_element_solution(n1_coord, n2_coord, n3_coord, u)

    # Visualize the basis functions
    visualize_element_basis(n1_coord, n2_coord, n3_coord)
    plt.show()
