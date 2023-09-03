import numc as nc

mat1 = nc.Matrix(3,3,1.0)
mat2 = nc.Matrix(3,3,2)
mat3 = nc.Matrix(3,3)
mat3 = mat1+mat2
print(mat3)