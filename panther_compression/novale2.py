from scipy.optimize import linear_sum_assignment
import numpy as np
cost = np.array([[4, 1, 3, 0], [2, 0, 5, 7], [3, 2, 2, 4], [1, 2, 2, 4]])
row_ind, col_ind = linear_sum_assignment(cost)

print(row_ind, col_ind)

#Note that the row indexes are sorted, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
for m in row_ind:
	print(m)
	print((row_ind[m], col_ind[m]))