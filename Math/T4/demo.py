import numpy as np
import sympy as sy


def determinant_laplace(A):
    if A.size == 1:
        return A[0, 0]
    s = 0
    for i in range(A.shape[0]):
        s += A[i, 0]*(-1)**(i+0) * \
            determinant_laplace(A[[x for x in range(A.shape[0]) if x != i],1:])
    return s

def determinant_triangle(A):
    # chay theo duong cheo chinh

    A = np.array(A, dtype=np.float64)
    for i in range(A.shape[0]):
        # neu phan tu tren duong cheo chinh bang 0
        if A[i, i] == 0:
            # tim các hàng con lai
            for j in range(i+1, A.shape[0]):
                if A[j, i] != 0:
                    A[[i, j]] = A[[j, i]]
                    break
            else:
                return 0
        for j in range(i+1, A.shape[0]):
            A[j] += A[i]*-A[j, i]/A[i, i]
    return np.prod(np.diagonal(A))


def inverse(A):
    A = np.array(A, dtype=np.float64)
    det_A = determinant_triangle(A)
    if det_A == 0:
        raise ValueError

    result = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            result[i, j] = (-1)**(i+j)*\
                determinant_triangle(A[[x for x in range(A.shape[0])if x != j]]\
                    [:, [x for x in range(A.shape[0]) if x != i]])/det_A

    return result


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
print(A[[1,2]][[1,1]])
print(inverse(A))
