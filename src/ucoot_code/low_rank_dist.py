import torch
from torch.distributions.categorical import Categorical

def low_rank_squared_l2(X, Y):
    """
    Write square Euclidean distance matrix as exact product of two low rank matrices.
    """
    device, dtype = X.device, X.dtype
    nx, ny = X.shape[0], Y.shape[0]

    Vx = (X ** 2).sum(1, keepdim=True)  # shape nx x 1
    Vy = (Y ** 2).sum(1, keepdim=True)  # shape ny x 1
    ones_x = torch.ones(nx, 1).to(device).to(dtype)  # shape nx x 1
    ones_y = torch.ones(ny, 1).to(device).to(dtype)  # shape ny x 1

    D1 = torch.cat([Vx, ones_x, -(2 ** 0.5) * X], dim=1)  # shape nx x (d+2)
    D2 = torch.cat([ones_y, Vy, 2 ** 0.5 * Y], dim=1)  # shape ny x (d+2)

    return (D1, D2)

def low_rank_l2(X, Y, r, gamma):
    """
    gamma: tolerance of the factorisation. See section 3.5 in the low rank paper
    r: rank of the factorisation.
    
    Return M, N such that || X - Y ||_2 \approx M @ N.T

    During the approximation, this algo requires to stock a matrix of size (n, t)
    where t = r / gamma. So, need to be careful with the choice of r and gamma when using GPU.
    It may be better not to use GPU, but CPU only.
    """

    t = int(r / gamma)
    n, m = X.shape[0], Y.shape[0]
    device = X.device

    i_star, j_star = torch.randint(0, n, (1,)), torch.randint(0, m, (1,))
    p = torch.cdist(X, Y[j_star].reshape(1,-1)).pow(2) + \
        ((X[i_star] - Y[j_star])**2).sum() + \
        torch.cdist(X[i_star].reshape(1,-1), Y).pow(2).mean() # shape (n,)
    p = p.squeeze() / p.sum() # shape (n,)

    proba = Categorical(p)
    it = [proba.sample() for _ in range(t)] # shape (t,)
    Pt = (t * p[it])**0.5 # shape (t,)
    S = torch.cdist(X[it], Y) / Pt.reshape(-1,1) # shape (t,m) => COMPUTATIONALLY EXPENSIVE
    q = (S**2).sum(0) # shape (m,)
    q = q / q.sum() # shape (m,)

    proba = Categorical(q)
    jt = [proba.sample() for _ in range(t)] # shape (t,)
    W = S[:,jt] / (t * q[jt])**0.5 # shape (t,t) => NEED TO BE STOCKABLE IN MEMORY

    W_cpu = W.detach().cpu() # shape (t,t)
    U1, _, _ = torch.linalg.svd(W_cpu) # D1 will be ordered in descending order.
    U1 = U1.to(device) # shape (t,t)
    N = S.T @ U1[:,:r] / torch.linalg.norm(W.T @ U1[:,:r], ord="fro") # shape m x r

    N_cpu = N.detach().cpu()
    # U2, D2, _ = torch.linalg.svd(N_cpu.T @ N_cpu) # U2 = (r,r)
    D2, U2 = torch.linalg.eigh(N_cpu.T @ N_cpu) # U2 = (r,r), faster than SVD
    U2 = U2 / D2
    U2 = U2.to(device)

    jt = torch.randint(0, m, (t,)) # shape (t,)
    Dt = torch.cdist(X, Y[jt]) / (t**0.5) # shape (n,t) => COMPUTATIONALLY EXPENSIVE

    B = U2.T @ N.T[:,jt] / (t**0.5) # shape (r,t)
    B_cpu = B.detach().cpu()
    A = torch.linalg.inv(B_cpu @ B_cpu.T) # shape (r,r)
    A = A.to(device)
    M = (A @ B @ Dt.T).T @ U2.T # shape (n,r)

    return M, N

# # Implementation of Meyer Scetbon
# import numpy as np

# def Learning_linear_subspace(X, Y, cost, U, tol=1e-3):
#     rank, m = np.shape(U)
#     U_sym = np.dot(U, U.T)  # k x k
#     d, v = np.linalg.eigh(U_sym)
#     v = v / np.sqrt(d)  # k x k

#     t = int(rank / tol)
#     ind_column = np.random.choice(m, size=t)
#     U_trans = U[:, ind_column]  # k x k/tol

#     A_trans = cost(X, Y[ind_column, :])

#     A_trans = A_trans / np.sqrt(t)
#     B = np.dot(v.T, U_trans) / np.sqrt(t)# k x k/tol
#     Mat = np.linalg.inv(np.dot(B, B.T))
#     Mat = np.dot(Mat, B)  # k x k/tol
#     alpha = np.dot(Mat, A_trans.T)  # k x n

#     V_f = np.dot(alpha.T, v.T)

#     return V_f

# def factorized_distance_cost(X, Y, rank, cost, tol=1e-3, seed=49):
#     np.random.seed(seed)
#     n, m = np.shape(X)[0], np.shape(Y)[0]

#     i_ = np.random.randint(n, size=1)
#     j_ = np.random.randint(m, size=1)

#     X_trans = X[i_, :].reshape(1, -1)
#     cost_trans_i = cost(X_trans, Y)
#     mean = np.mean(cost_trans_i ** 2)

#     Y_trans = Y[j_, :].reshape(1, -1)
#     cost_trans_j = cost(X, Y_trans)

#     p_row = cost_trans_j ** 2 + cost_trans_i[0, j_] ** 2 + mean
#     p_row = p_row / np.sum(p_row)  # vector of size n

#     # Compute S
#     ind_row = np.random.choice(n, size=int(rank / tol), p=p_row.reshape(-1))
#     S = cost(X[ind_row, :], Y)  # k/tol x m

#     p_row_sub = p_row[ind_row]
#     S = S / np.sqrt(int(rank / tol) * p_row_sub)

#     norm_square_S = np.sum(S ** 2)
#     p_column = np.zeros(m)
#     for j in range(m):
#         p_column[j] = np.sum(S[:, j] ** 2) / norm_square_S
#     # p_column = p_column / np.sum(p_column) # vector of size m

#     # Compute W
#     ind_column = np.random.choice(m, size=int(rank / tol), p=p_column.reshape(-1))
#     W = S[:, ind_column]  # k/tol x k/tol
#     p_column_sub = p_column[ind_column]
#     W = (W.T / np.sqrt(int(rank / tol) * p_column_sub)).T

#     # Compute U
#     u, d, v = np.linalg.svd(W)
#     U = u[:, :rank]  # k/tol x k
#     U_trans = np.dot(W.T, U)  # k/tol x k

#     norm_U = np.sum(U_trans ** 2, axis=0)
#     norm_U = np.sqrt(norm_U)

#     U = np.dot(S.T, U)  # m x k
#     U = U / norm_U

#     # Compute V
#     V = Learning_linear_subspace(X, Y, cost, U.T, tol=tol)

#     return V, U.T