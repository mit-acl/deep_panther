import scipy.io
import numpy as np
from pypoman import compute_polytope_vertices


mat = scipy.io.loadmat('Ab.mat')

A=np.array(mat['Ax_nz'])
b=np.array(mat['b_nz'])
b=b.ravel()
np.set_printoptions(edgeitems=30, linewidth=100000)
# print(A)
# print(b)

# vertices = compute_polytope_vertices(A, b.ravel())


# A = np.array([
#     [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#     [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#     [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
#     [1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0],
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],
#     [1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0],
#     [0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0],
#     [0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1]])
# b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 3])


print(A)

print(A.shape)
print(b.shape)



vertices = compute_polytope_vertices(A.T, b)
