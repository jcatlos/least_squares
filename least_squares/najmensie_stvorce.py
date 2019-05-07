import csv
import matplotlib.pyplot as plt
import numpy as np

# function definition

def transpose(matrix):
    new_matrix = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        new_matrix.append(row)
    return new_matrix

def REF(matrix):
    for i in range(len(matrix)):
        alpha = 1
        pivot_position = -1
        for j in range(len(matrix[i])):
            if matrix [i][j] != 0:
                alpha = matrix[i][j]
                pivot_position = j
                break
        for j in range(len(matrix[i])):
            matrix[i][j] /= alpha

        for remainder in range(i+1, len(matrix)):
            beta = matrix[remainder][pivot_position]
            for j in range(len(matrix[remainder])):
                matrix[remainder][j] -= (matrix[i][j] * beta)
    
    # Ordering rows
    for i in range(len(matrix)):
        pivot_position = len(matrix[i])+1
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                pivot_position = j
                break
        for remainder in range(i+1, len(matrix)):
            #print("pozicia: ", pivot_position, " i: ", i)
            for j in range(min(pivot_position, len(matrix[remainder]))):
                #print("j: ", j)
                if matrix[remainder][j] != 0:
                    pivot_position = j
                    for tmpj in range(len(matrix[i])):
                        tmp = matrix[remainder][tmpj]
                        matrix[remainder][tmpj] = matrix[i][tmpj]
                        matrix[i][tmpj] = tmp
                    break
    return matrix


def RREF(matrix):
    matrix = REF(matrix)

    for i in range(len(matrix)-1, 0, -1):
        pivot_position = -1
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                pivot_position = j
                break
        if pivot_position >= 0:
            for remainder in range(i-1, -1, -1):
                gamma = matrix[remainder][pivot_position]
                for j in range(len(matrix[remainder])):
                    matrix[remainder][j] -= (matrix[i][j]*gamma)
    return matrix

def matrix_multiply(A,B):
    result = []
    for i in range(len(A)):
        result_row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += A[i][k] * B[k][j]
            result_row.append(sum)
        result.append(result_row)
    return result

def inverse(matrix):
    width = len(matrix[0])
    new_matrix = matrix
    for i in range(len(matrix)):
        for j in range(width):
            if i == j:
                matrix[i].append(1)
            else:
                matrix[i].append(0)
    
    new_matrix = RREF(new_matrix)
    for i in range(len(new_matrix)):
        new_matrix[i] = new_matrix[i][width:]
    return new_matrix


# Start of actual program

indep_values = []
dep_values = []


with open('data.csv') as data:
    reader = csv.reader(data, delimiter = ',')
    for row in reader:
        # Saving only rows containing non-negative integers (for sure)
        indep_val = None
        dep_val = None
        try:
            indep_val = int(row[0])
        except ValueError:
            pass

        try:
            dep_val = int(row[1])
        except ValueError:
            pass

        if indep_val != None and dep_val != None:
            indep_values.append(indep_val)
            dep_values.append(dep_val)

# AtAx = Atb
# x = (AtA)-1 AtB

A = []
for i in range(len(indep_values)):
    row = [indep_values[i]**2, indep_values[i], 1]
    A.append(row)

At = transpose(A)

B = []
for i in range(len(dep_values)):
    row = [dep_values[i]]
    B.append(row)

invAtA = inverse(matrix_multiply(At, A))
x = matrix_multiply(invAtA, matrix_multiply(At, B))

#print(x) 

t = np.arange(min(indep_values)-5,max(indep_values)+15, 10)

plt.plot(indep_values, dep_values, 'ro')
plt.plot(t, x[0]*t**2 + x[1]*t + x[2])

plt.show()