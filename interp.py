import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def TriangleGuassianQuadrature():
    return


def element_data(n1_coord, n2_coord, n3_coord):
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

    return a, b, c, l, area


def eval_shape_funcs(a, b, c, l, area, x, y):
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

    return N1, N2, N3, curlN1, curlN2, curlN3


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


def visualize_element_basis():
    # Initialize plot
    fig, ax = plt.subplots(nrows=3)

    # Define element
    n1_coord = np.array([0.0, 0.0])
    n2_coord = np.array([1.0, 1.0])
    n3_coord = np.array([0.0, 1.0])

    # Plot the mesh
    triang = tri.Triangulation(
        [n1_coord[0], n2_coord[0], n3_coord[0]],
        [n1_coord[1], n2_coord[1], n3_coord[1]],
        [[0, 1, 2]],
    )
    ax[0].triplot(triang, "bo--")
    ax[1].triplot(triang, "go--")
    ax[2].triplot(triang, "ro--")

    # Extract element data
    a, b, c, l, area = element_data(n1_coord, n2_coord, n3_coord)

    # Compute the vector field inside the element
    npts = 100  # total number of points in the vector field
    for i in range(npts):
        # Get the x, y point in the triangle to evaluate the shape function
        x, y = barycenter_sampling(n1_coord, n2_coord, n3_coord)

        # Evaluate the shape funcs
        N1, N2, N3, curlN1, curlN2, curlN3 = eval_shape_funcs(a, b, c, l, area, x, y)
        print(curlN1, curlN2, curlN3)

        # Update the plot
        ax[0].quiver(x, y, N1[0], N1[1], width=4e-3)
        ax[1].quiver(x, y, N2[0], N2[1], width=4e-3)
        ax[2].quiver(x, y, N3[0], N3[1], width=4e-3)

    plt.show()
    return


if __name__ == "__main__":
    # Visualize the basis functions
    visualize_element_basis()
