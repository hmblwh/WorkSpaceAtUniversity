
import numpy as np


def cholesky_decomposition(A):
    A = np.array(A, dtype=np.float64)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Ma trận không hợp lệ")
    if (A != np.transpose(A)).any():
        raise ValueError("Không phải ma trận đối xứng")
    if (np.linalg.eigvals(A) <= 0).any():
        raise ValueError("Không phải ma trận xác định dương")
    L = np.zeros_like(A)
    for j in range(A.shape[0]):
        L[j, j] = np.sqrt(A[j, j]-np.sum(L[j, range(j)]**2))
        for i in range(j+1, A.shape[0]):
            L[i, j] = 1/L[j, j] * \
                (A[i, j] - np.sum(L[i, range(j)]*L[j, range(j)]))
    return L


if __name__ == '__main__':
    A = [[4, 12, -16],
         [12, 37, -43],
         [-16, -43, 98]]
    print(cholesky_decomposition(A))
