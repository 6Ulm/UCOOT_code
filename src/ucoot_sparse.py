import torch
from tqdm import tqdm

def sparsify(sparse_tensor):
    val = sparse_tensor._values()
    idx = sparse_tensor._indices()
    dim = sparse_tensor.size()

    idx_nnz = torch.where(val != 0)[0]
    val = val[idx_nnz]
    idx = idx[:,idx_nnz]
    sparse_tensor = torch.sparse_coo_tensor(idx, val, dim)

    return sparse_tensor

def get_sparsity(sparse_tensor):
    nnz, dim = sparse_tensor._nnz(), sparse_tensor.size()
    sparsity = 1 - nnz / (dim[0] * dim[1])
    return sparsity

def uot_l2_sparse(a, b, cost, reg, niter, tol):
    thres = torch.clamp(a[:,None] + b[None,:] - cost / reg, min=0)

    thres = thres.to_sparse()
    thres_val = thres._values()
    idx = thres._indices()
    dim = thres.size()

    pi = a[idx[0]] * b[idx[1]]
    m1, m2 = a, b
    
    for _ in tqdm(range(niter)):

        m1_old, m2_old = m1, m2

        pi_sparse = torch.sparse_coo_tensor(idx, pi, dim)

        m1 = torch.sparse.sum(pi_sparse, 1).to_dense()
        m2 = torch.sparse.sum(pi_sparse, 0).to_dense()

        pi = pi * thres_val / (m1[idx[0]] + m2[idx[1]])

        if max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break
    
    pi_sparse = torch.sparse_coo_tensor(idx, pi, dim)
    pi_sparse = sparsify(pi_sparse)

    return pi_sparse

def uot_l2(a, b, cost, reg, niter, tol):
    thres = torch.clamp(a[:,None] + b[None,:] - cost / reg, min=0)
    
    pi = a[:,None] * b[None,:]
    m1, m2 = a, b

    for _ in tqdm(range(niter)):
        m1_old, m2_old = m1, m2
        denom = m1[:,None] + m2[None,:]
        pi = pi * thres / denom
        m1, m2 = pi.sum(1), pi.sum(0)
        if max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break
    
    return pi

def uot_dc(a, b, cost, tuple_lamb, niter, tol):
    l1, l2, l_reg = tuple_lamb
    l_all = l1 + l2 + l_reg
    rho1, rho2, rho_r = l1 / l_all, l2 / l_all, l_reg / l_all

    log_a, log_b = a.log(), b.log()
    log_K = (rho1 + rho_r) * log_a[:,None] + (rho2 + rho_r) * log_b[None,:] - cost / l_all
    log_pi = log_a[:,None] + log_b[None,:]
    log_m1, log_m2 = log_a, log_b

    for _ in tqdm(range(niter)):
        log_m1_old, log_m2_old = log_m1, log_m2
        log_pi = (rho1 + rho2) * log_pi - (rho1 * log_m1[:,None] + rho2 * log_m2[None,:]) + log_K
        pi = log_pi.exp()
        log_m1, log_m2 = pi.sum(1).log(), pi.sum(0).log()
        if max((log_m1 - log_m1_old).abs().max(), (log_m2 - log_m2_old).abs().max()) < tol:
            break
    
    return pi

