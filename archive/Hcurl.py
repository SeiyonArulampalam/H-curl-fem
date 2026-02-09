import numpy as np
import H1

"""Linear Triangle Curl Element (Hcurl)"""


def eval_curl(xi, eta):
    curl_N1 = np.array([0, 0, 2])
    curl_N2 = np.array([0, 0, 2])
    curl_N3 = np.array([0, 0, -2])
    return curl_N1, curl_N2, curl_N3


def eval_shape_funcs(xi, eta):
    N1 = np.array([1 - eta, xi])
    N2 = np.array([-eta, xi])
    N3 = np.array([-eta, -1 + xi])
    return N1, N2, N3


def compute_vector_field(xi, eta, u, X, Y):
    N1, N2, N3 = eval_shape_funcs(xi, eta)
    return u[0] * N1 + u[1] * N2 + u[2] * N3
