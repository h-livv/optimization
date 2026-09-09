import numpy as np

'''
max z = 5x1 + 4x2 + 3x3

2x1 + x2 + x3 + s1 = 8
x1 + 2x2 + x3 + s2 = 8
x1 + x2 + 2x3 + s3 = 8
'''

#Do you want to maximize or minimize?
process = "max"

#Initialization

#Column vectors of the coefficient matrix
x1 = c0 = np.array([2, 1, 1])
x2 = c1 = np.array([1, 2, 1])
x3 = c2 = np.array([1, 1, 2])
s1 = c3 = np.array([1, 0, 0])
s2 = c4 = np.array([0, 1, 0])
s3 = c5 = np.array([0, 0, 1])

#Coefficient matrix
A = np.column_stack((x1, x2, x3, s1, s2, s3))

#Solution matrix
b = np.array([8, 8, 8])

#Coefficient matrix of the objective function
c = np.array([5, 4, 3, 0, 0, 0])

#Index of the basis matrix columns
basic_index = [3, 4, 5]

n = 1

while True:
    B = A[:, basic_index]
    B_inv = np.linalg.inv(B)

    print(f"\n========== ITERATION {n} ==========")
    print(f"Basic variables:\n{B}")

    x_b = B_inv @ b
    print(f"Basic variable values: {x_b}")

    c_B = c[basic_index]

    # Non-basic variables
    all_indices = range(len(c))
    non_basic_index = [i for i in all_indices if i not in basic_index]

    N = A[:, non_basic_index]

    print(f"Non-basic variables:\n{N}")

    c_N = c[non_basic_index]

    # Reduced costs
    reduced_costs = c_N - c_B.T @ B_inv @ N
    print(f"Reduced costs: {reduced_costs}")

    # Stopping conditions
    if process == "max":
        if not np.any(reduced_costs > 0):
            print("STATUS: OPTIMAL")
            break

    elif process == "min":
        if not np.any(reduced_costs < 0):
            print("STATUS: OPTIMAL")
            break

    # Find the entering variable
    if process == "max":
        entering_position = np.argmax(reduced_costs)
    elif process == "min":
        entering_position = np.argmin(reduced_costs)

    movement_index = non_basic_index[entering_position]
    a1 = A[:, movement_index]

    print(f"Entering variable:\n{a1}")

    # Ratio test
    d = B_inv @ a1

    valid_rows = d > 0
    ratios = x_b[valid_rows] / d[valid_rows]

    leaving_position = np.where(valid_rows)[0][np.argmin(ratios)]

    print(f"Ratios: {ratios}")
    print(f"Selected ratio: {ratios.min()}")

    leaving_variable_index = basic_index[leaving_position]

    print(f"Leaving variable:\n{A[:, leaving_variable_index]}")

    # Update basis
    basic_index[leaving_position] = movement_index

    n += 1

print(f"\nFinal basic variables:\n{A[:, basic_index]}")
print(f"Final solution: {x_b}")
print(f"Optimal value: {c[basic_index].T @ x_b}")