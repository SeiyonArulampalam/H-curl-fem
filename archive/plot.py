import numpy as np
import matplotlib.tri as tri


# Barycenter sampling
def barycenter_sampling_ref_element():
    """
    Reference element: new_point = (xi, eta)
    P1 = (x,y) for node 1
    P2 = (x,y) for node 2
    P3 = (x,y) for node 3
    """
    P1 = np.array([0.0, 0.0])
    P2 = np.array([0.0, 1.0])
    P3 = np.array([1.0, 0.0])

    r1 = np.random.rand()
    r2 = np.random.rand()
    # Ensure uniform distribution in triangle
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    new_point = (1 - r1 - r2) * P1 + r1 * P2 + r2 * P3
    return new_point[0], new_point[1]  # xi and eta frame


# Plot the mesh
def mesh(elem_conn, X, fig, ax):
    triang = tri.Triangulation(X[:, 0], X[:, 1], elem_conn)
    ax.triplot(triang, "bo--")
    return


# Plot vector field inside an element
def vector_field(x, y, vec, fig=None, ax=None):
    ax.quiver(x, y, vec[0], vec[1], width=4e-3)
    return
