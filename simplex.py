import numpy as np

'''
z = 3x + 2y

x + y + s1 = 4
2x + y + s2 = 5
x + s3 = 2
'''

#Helper functions
def basis(arr1, arr2, arr3):
    return np.column_stack((arr1, arr2, arr3))

def non_basis(arr1, arr2):
    return np.column_stack((arr1, arr2))

#Initialization

#Column vectors of the coefficient matrix
x1 = c0 = np.array([1, 2, 1])
x2 = c1 = np.array([1, 1, 0])
s1 = c2 = np.array([1, 0, 0])
s2 = c3 = np.array([0, 1, 0])
s3 = c4 = np.array([0, 0, 1])

#Coefficient matrix
A = np.column_stack((x1, x2, s1, s2, s3))

#Solution matrix
b = np.array([4, 5, 2])

#Coefficient matrix of the objective function
c = np.array([3, 2, 0, 0, 0])

basic_index = [2, 3, 4]

n =1

non_basic_coeff = np.array([])

while True:
    #Extract the initial basis matrix. We assume the decision variables are non-basic and set them to zero (x1 = x2 = 0)
    #B is the identity matrix
    B = basis(*(A[:, basic_index].T))

    print(B)

    #Calculate the current BFS.
    x_b = np.linalg.inv(B)@b

    print(x_b)

    all_indices = range(len(c))
    non_basic_index = [i for i in all_indices if i not in basic_index]

    print(non_basic_index)

    N = non_basis(*(A[:, non_basic_index].T))

    print(N)

    non_basic_coeff = np.array([])

    for index in non_basic_index:
        non_basic_coeff = np.append(non_basic_coeff, c[index])

    print(non_basic_coeff)


    if not np.any(non_basic_coeff > 0):
        break

    movement_index = np.where(c == non_basic_coeff.max())[0][0]

    print(movement_index)

    #Column of the corresponding entering variable
    a1 = A[:, movement_index]

    print(a1)

    #Calcluate the ratios to find the leaving variable
    d = np.linalg.inv(B)@a1

    print(d)

    valid_rows = (d > 0).flatten()

    # Filter both matrices using the mask
    x_b_filtered = x_b[valid_rows]
    d_filtered = d[valid_rows]

    # Calculate the ratios
    ratios = x_b_filtered / d_filtered

    t = ratios[ratios > 0].min()

    print(ratios)

    print(t)

    leaving_variable_index = basic_index[np.where(x_b == t)[0][0]]

    print(leaving_variable_index)

    basic_index[int(np.where(x_b == t)[0][0])] = int(movement_index)

    print(basic_index)

    print(f"--------{n} ITERATION(S) COMPLETED---------")

    n = n+1

print(x_b)