import torch
from tqdm import tqdm
from ot.lp import emd

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

def emd_solver(cost, p1, p2, device, dtype):
    p1, p2 = p1.detach().cpu().numpy(), p2.detach().cpu().numpy()
    p1_norm, p2_norm = p1 / p1.sum(), p2 / p2.sum()

    cost_np = cost.detach().cpu().numpy()

    pi_np, log = emd(p1_norm, p2_norm, cost_np, log=True)
    f1 = torch.from_numpy(log["u"]).to(device).to(dtype)
    f2 = torch.from_numpy(log["v"]).to(device).to(dtype)
    pi = torch.from_numpy(pi_np).to(device).to(dtype) # pi has unit mass

    return (f1, f2), pi

def normalise(p1, p2):
    p1, p2 = p1 * (p2.sum() / p1.sum())**0.5, p2 * (p1.sum() / p2.sum())**0.5
    return (p1, p2)

def uot(a, b, cost, rho1, rho2, niter, tol):
    device, dtype = a.device, a.dtype
    m1, m2 = normalise(a, b)

    for _ in tqdm(range(niter)):
        m1_old, m2_old = m1, m2

        (f, g), pi = emd_solver(cost, m1, m2, device, dtype)
        m1 = a / (f / rho1).exp()
        m2 = b / (g / rho2).exp()
        m1, m2 = normalise(m1, m2)

        if max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break
    pi = pi * m1.sum()

    return (f, g), pi