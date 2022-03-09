from munkres import Munkres

m = Munkres()

from munkres import Munkres, print_matrix

matrix = [[11-5,11- 9],
          [11-10, 11-3],
          [11-8, 11-7]]
m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total cost: {total}')