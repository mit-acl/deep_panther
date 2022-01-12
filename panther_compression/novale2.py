from scipy.optimize import linear_sum_assignment
import numpy as np
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
row_ind, col_ind = linear_sum_assignment(cost)

print(row_ind, col_ind)