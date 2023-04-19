import numpy as np


def solve(g, W, c, J, lambda_, mu):
    
    if J is not None and lambda_ is not None:
        L = np.diag(lambda_[:,0])
        C_diag = np.diag(c[:,0])

        A = W
        B = -np.transpose(J)
        C = np.matmul(L, J)
        D = C_diag

        G = np.block([[A, B], [C, D]])
        b = -np.block([[g - np.matmul(np.transpose(J), lambda_)], [np.matmul(C_diag, lambda_) - mu*np.ones((np.size(c), 1))]])

        return np.linalg.solve(G, b)

    else:
        G = W
        b = -g

        return np.linalg.solve(G, b)

    # import pdb; pdb.set_trace()

