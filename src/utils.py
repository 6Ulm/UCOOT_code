import torch
# require torch >= 1.9
from functools import partial
from tqdm import tqdm

def batch_calculation(mat1, mat2, vec):
    m, n = mat1.shape[0], mat2.shape[0]
    bs = min(int(1e8 / n), m)
    rg = range(0, m, bs)
    res = torch.cat([(mat1[i:i+bs:, :] @ mat2.T)**2 @ vec for i in rg], dim=0)
    return res

def approx_kl(p, q):
    """
    Calculate p * log (p/q). By convention: 0 log 0 = 0
    """

    return torch.nan_to_num(p * p.log(), nan=0.0, posinf=0.0, neginf=0.0).sum() - (p * q.log()).sum()

def kl(p, q):
    """
    Calculate KL divergence in the most general case:
    KL = p * log (p/q) - mass(p) + mass(q)
    """

    return approx_kl(p, q) - p.sum() + q.sum()

def quad_kl(mu, nu, alpha, beta):
    """
    Calculate the KL divergence between two product measures: 
    KL(mu \otimes nu, alpha \otimes beta) = 
    m_mu * KL(nu, beta) + m_nu * KL(mu, alpha) + (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: vector or matrix
    nu: vector or matrix
    alpha: vector or matrix with the same size as mu
    beta: vector or matrix with the same size as nu

    Returns
    ----------
    KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)

    return m_nu * kl(mu, alpha) + m_mu * kl(nu, beta) + const

def uot_ent(cost, init_dual, tuple_log_p, params, n_iters, tol, eval_freq):
    """
    Solve entropic UOT using Sinkhorn algorithm.
    Allow rho1 and/or rho2 to be infinity but epsilon must be strictly positive.
    """

    rho1, rho2, eps = params
    log_a, log_b, ab = tuple_log_p
    f, g = init_dual

    tau1 = 1 if torch.isinf(rho1) else rho1 / (rho1 + eps)
    tau2 = 1 if torch.isinf(rho2) else rho2 / (rho2 + eps)

    for idx in range(n_iters):
        f_prev = f.detach().clone()
        if rho2 == 0: # semi-relaxed
            g = torch.zeros_like(g)
        else:
            g = -tau2 * ((f + log_a)[:, None] - cost / eps).logsumexp(dim=0)

        if rho1 == 0: # semi-relaxed
            f = torch.zeros_like(f)
        else:
            f = -tau1 * ((g + log_b)[None, :] - cost / eps).logsumexp(dim=1)

        if (idx % eval_freq == 0) and (f - f_prev).abs().max().item() < tol:
            break

    pi = ab * (f[:, None] + g[None, :] - cost / eps).exp()

    return (f, g), pi

# def uot_mm(cost, init_pi, tuple_log_p, params, n_iters, tol, eval_freq):
#     """
#     Solve (entropic) UOT using the max-min algorithm.
#     Allow epsilon to be 0 but rho1 and rho2 can't be infinity.
#     """
#
#     log_a, log_b, _ = tuple_log_p
#     rho1, rho2, eps = params
#     sum_param = rho1 + rho2 + eps
#     tau1, tau2, rho_r = rho1 / sum_param, rho2 / sum_param, eps / sum_param
#     log_K = (tau1 + rho_r) * log_a[:,None] + (tau2 + rho_r) * log_b[None,:] - cost / sum_param
    
#     log_pi = torch.log(init_pi + 1.0 * (init_pi == 0))
#     log_m1, log_m2 = init_pi.sum(1).log(), init_pi.sum(0).log()

#     for idx in range(n_iters):
#         log_m1_old, log_m2_old = log_m1.detach().clone(), log_m2.detach().clone()
#         log_pi = (tau1 + tau2) * log_pi - (tau1 * log_m1[:,None] + tau2 * log_m2[None,:]) + log_K
#         pi = log_pi.exp()
#         log_m1, log_m2 = pi.sum(1).log(), pi.sum(0).log()
#         if (idx % eval_freq == 0) and \
#             max((log_m1 - log_m1_old).abs().max(), (log_m2 - log_m2_old).abs().max()) < tol:
#             break
    
#     return None, pi

def uot_mm(cost, init_pi, tuple_p, params, n_iters, tol, eval_freq):
    """
    Solve (entropic) UOT using the max-min algorithm.
    Allow epsilon to be 0 but rho1 and rho2 can't be infinity.
    Note that if the parameters are small so that numerically, the exponential of 
    negative cost will contain zeros and this serves as sparsification of the optimal plan. 
    If the parameters are large, then the resulting optimal plan is more dense than the one 
    obtained from Sinkhorn algo. 
    But the parameters should not be too small, otherwise the kernel will contain too many zeros 
    and consequently, the optimal plan will contain NaN (because the Kronecker sum of two marginals 
    will eventually contain zeros, and divided by zero will result in undesirable result).
    """

    a, b, _ = tuple_p
    rho1, rho2, eps = params
    sum_param = rho1 + rho2 + eps
    tau1, tau2, rho_r = rho1 / sum_param, rho2 / sum_param, eps / sum_param
    K = a[:,None]**(tau1 + rho_r) * b[None,:]**(tau2 + rho_r) * (- cost / sum_param).exp()
    
    m1, m2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(n_iters):
        m1_old, m2_old = m1.detach().clone(), m2.detach().clone()
        pi = pi**(tau1 + tau2) / (m1[:,None]**tau1 * m2[None,:]**tau2) * K
        m1, m2 = pi.sum(1), pi.sum(0)
        if (idx % eval_freq == 0) and \
            max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break
    
    return None, pi

def get_local_cost(data, pi, tuple_p, hyperparams, entropic_mode):
    """
    Calculate cost of the UOT.
    cost = (X**2 * P_#1 + Y**2 * P_#2 - 2 * X * P * Y.T) + 
            rho1 * approx_kl(P_#1 | a) + rho2 * approx_kl(P_#2 | b) +
            eps * approx_kl(P | a \otimes b)
    """

    rho, eps = hyperparams
    rho1, rho2, _, _, _, _ = rho
    a, b, ab = tuple_p
    X_sqr, Y_sqr, X, Y, alpha_D = data

    pi1, pi2 = pi.sum(1), pi.sum(0)
    B = Y_sqr @ pi2

    if isinstance(X_sqr, tuple):
        X1, X2 = X_sqr
        A = batch_calculation(X1, X2, pi1)
        cost = A[:, None] + B[None, :] - 2 * X1 @ ((X2.T @ pi) @ Y.T) + alpha_D

    elif torch.is_tensor(X_sqr):        
        A = X_sqr @ pi1
        cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T + alpha_D

    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * approx_kl(pi1, a)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * approx_kl(pi2, b)
    if entropic_mode == "joint":
        cost = cost + eps[0] * approx_kl(pi, ab)

    return cost

def get_cost(pi_samp, pi_feat, data, data_T, tuple_pxy_samp, tuple_pxy_feat, hyperparams, entropic_mode):
    """
    Calculate complete UCOOT cost.
    """

    rho, eps = hyperparams
    eps_samp, eps_feat = eps
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    px_samp, py_samp, pxy_samp = tuple_pxy_samp
    px_feat, py_feat, pxy_feat = tuple_pxy_feat
    X_sqr, Y_sqr, X, Y, alpha_D_samp = data
    _, _, _, _, alpha_D_feat = data_T

    pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)
    pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)

    B_sqr = (Y_sqr @ pi2_feat).dot(pi2_samp)

    # UGW part
    if isinstance(X_sqr, tuple):
        X1, X2 = X_sqr
        A = batch_calculation(X1, X2, pi1_feat)
        A_sqr = A.dot(pi1_samp)
        AB = (X1 @ ((X2.T @ pi_feat) @ Y.T)) * pi_samp
        cost = A_sqr + B_sqr - 2 * AB.sum()

    elif torch.is_tensor(X_sqr):
        A_sqr = (X_sqr @ pi1_feat).dot(pi1_samp)
        AB = (X @ pi_feat @ Y.T) * pi_samp
        cost = A_sqr + B_sqr - 2 * AB.sum()

    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * quad_kl(pi1_samp, pi1_feat, px_samp, px_feat)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * quad_kl(pi2_samp, pi2_feat, py_samp, py_feat)

    # UOT part
    if torch.is_tensor(alpha_D_samp):
        cost = cost + (alpha_D_samp * pi_samp).sum()
    if rho1_samp != float("inf") and rho1_samp != 0:
        cost = cost + rho1_samp * kl(pi1_samp, px_samp)
    if rho2_samp != float("inf") and rho2_samp != 0:
        cost = cost + rho2_samp * kl(pi2_samp, py_samp)            

    if torch.is_tensor(alpha_D_feat):
        cost = cost + (alpha_D_feat * pi_feat).sum()            
    if rho1_feat != float("inf") and rho1_feat != 0:
        cost = cost + rho1_feat * kl(pi1_feat, px_feat)
    if rho2_feat != float("inf") and rho2_feat != 0:
        cost = cost + rho2_feat * kl(pi2_feat, py_feat)

    # Entropic part
    ent_cost = cost
    if entropic_mode == "joint" and eps_samp != 0:
        ent_cost = ent_cost + eps_samp * quad_kl(pi_samp, pi_feat, pxy_samp, pxy_feat)
    elif entropic_mode == "independent":
        if eps_samp != 0:
            ent_cost = ent_cost + eps_samp * kl(pi_samp, pxy_samp)
        if eps_feat != 0:
            ent_cost = ent_cost + eps_feat * kl(pi_feat, pxy_feat)

    return cost.item(), ent_cost.item()

def solver(
    X,
    Y,
    px=(None, None),
    py=(None, None),
    rho=(float("inf"), float("inf"), 0, 0, 0, 0),
    uot_mode=("entropic", "entropic"),
    eps=(1e-2, 1e-2),
    entropic_mode="joint",
    alpha=(1, 1),
    D=(None, None),
    init_pi=(None, None),
    init_dual=(None, None),
    log=False,
    verbose=False,
    early_stopping_tol=1e-6,
    mass_rescaling=True,
    eval_bcd=10,
    eval_uot=1,
    tol_bcd=1e-7,
    nits_bcd=100,
    tol_uot=1e-7,
    nits_uot=500
):
    """
    Parameters
    ----------
    X: matrix of size nx x dx. First input data.
    Y: matrix of size ny x dy. Second input data.
    D: matrix of size nx x ny. Sample matrix, in case of fused GW
    px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        Uniform distributions by default.
    py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        Uniform distributions by default.
    rho: tuple of 6 relaxation marginal-relaxation parameters for UGW and UOT.
    uot_mode: string or tuple of strings. Uot modes for each update of couplings
        uot_mode = "entropic": use Sinkhorn algorithm in each BCD iteration.
        uot_mode = "mm": use maximisation-minimisation algorithm in each BCD iteration.
    eps: scalar or tuple of scalars.
        Regularisation parameters for entropic approximation of sample and feature couplings.
    entropic_mode:
        entropic_mode = "joint": use UGW-like regularisation.
        entropic_mode = "independent": use COOT-like regularisation.
    alpha: scaler or tuple of nonnegative scalars. 
        Interpolation parameter for fused UGW w.r.t the sample and feature couplings.
    D: tuple of matrices of size (nx x ny) and (dx x dy). The linear terms in UOT. 
        By default, set to None.
    init_pi: tuple of matrices of size nx x ny and dx x dy if not None.
        Initialisation of sample and feature couplings.
    init_dual: tuple of tuple of vectors of size (nx,ny) and (dx, dy) if not None.
        Initialisation of sample and feature dual vectos if using Sinkhorn algorithm.
    log: True if the cost is recorded, False otherwise.
    verbose: if True then print the recorded cost.
    early_stopping_tol: threshold for the early stopping.
    eval_bcd: multiplier of iteration at which the cost is calculated. For example, if eval_bcd = 10, then the
        cost is calculated at iteration 10, 20, 30, etc...
    eval_bcd: multiplier of iteration at which the old and new duals are compared in the Sinkhorn 
        algorithm.
    tol_bcd: tolerance of BCD scheme.
    nits_bcd: number of BCD iterations.
    tol_uot: tolerance of Sinkhorn or MM algorithm.
    nits_uot: number of Sinkhorn or MM iterations.

    Returns
    ----------
    pi_samp: matrix of size nx x ny. Sample matrix.
    pi_feat: matrix of size dx x dy. Feature matrix.
    dual_samp: tuple of vectors of size (nx, ny). Pair of dual vectors when using Sinkhorn algorithm 
        to estimate the sample coupling. Used in case of faster solver. 
        If use MM algorithm then dual_samp = None.
    dual_feat: tuple of vectors of size (dx, dy). Pair of dual vectors when using Sinkhorn algorithm 
        to estimate the feature coupling. Used in case of faster solver. 
        If use MM algorithm then dual_feat = None.
    log_cost: if log is True, return a list of cost (without taking into account the regularisation term).
    log_ent_cost: if log is True, return a list of entropic cost.
    """

    if isinstance(X, tuple):
        X1, X2 = X
        nx, dx = X1.shape[0], X2.shape[0]
    elif torch.is_tensor(X):
        nx, dx = X.shape
    else:
        raise ValueError("Invalid type of input.")

    ny, dy = Y.shape
    device, dtype = Y.device, Y.dtype

    # hyper-parameters
    if isinstance(eps, float) or isinstance(eps, int):
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError("Epsilon must be either a scalar or a tuple of scalars.")
    # if use joint penalisation for couplings, then only use the first value epsilon.
    if entropic_mode == "joint":
        eps = (eps[0], eps[0])

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha = (alpha, alpha)
    if not isinstance(alpha, tuple):
        raise ValueError("Alpha must be either a scalar or a tuple of scalars.")

    if isinstance(uot_mode, str):
        uot_mode = (uot_mode, uot_mode)
    if not isinstance(uot_mode, tuple):
        raise ValueError("uot_mode must be either a string or a tuple of strings.")

    # some constants
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    eps_samp, eps_feat = eps
    uot_mode_samp, uot_mode_feat = uot_mode
    if eps_samp == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0:
        raise ValueError("Invalid values for epsilon and rho of sample coupling. \
                        Cannot contain zero in epsilon AND infinity in rho at the same time.")
    else:
        if eps_samp == 0:
            uot_mode_samp = "mm"
        if torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0:
            uot_mode_samp = "entropic"

    if eps_feat == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0:
        raise ValueError("Invalid values for epsilon and rho of feature coupling. \
                        Cannot contain zero in epsilon AND infinity in rho at the same time.")
    else:
        if eps_feat == 0:
            uot_mode_feat = "mm"
        if torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0:
            uot_mode_feat = "entropic"
    uot_mode = (uot_mode_samp, uot_mode_feat)

    # measures on rows and columns
    px_samp, px_feat = px
    py_samp, py_feat = py

    if px_samp is None:
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
    if px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
    if py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy
    pxy_samp = px_samp[:, None] * py_samp[None, :]
    pxy_feat = px_feat[:, None] * py_feat[None, :]

    tuple_pxy_samp = (px_samp, py_samp, pxy_samp)
    tuple_pxy_feat = (px_feat, py_feat, pxy_feat)
    tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)
    tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    # constant data variables
    alpha_samp, alpha_feat = alpha
    D_samp, D_feat = D
    if D_samp is None or alpha_samp == 0:
        D_samp, alpha_samp = 0, 0
    if D_feat is None or alpha_feat == 0:
        D_feat, alpha_feat = 0, 0
    alpha_D_samp, alpha_D_feat = alpha_samp * D_samp, alpha_feat * D_feat

    Y_sqr = Y ** 2
    if isinstance(X, tuple):
        data = (X, Y_sqr, None ,Y, alpha_D_samp)
        data_T = ((X2, X1), Y_sqr.T, None, Y.T, alpha_D_feat)
        
    elif torch.is_tensor(X):
        X_sqr = X ** 2
        data = (X_sqr, Y_sqr, X, Y, alpha_D_samp)
        data_T = (X_sqr.T, Y_sqr.T, X.T, Y.T, alpha_D_feat)

    # initialise coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy

    if "entropic" in uot_mode:
        self_uot_ent = partial(uot_ent, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

        duals_samp, duals_feat = init_dual
        if uot_mode_samp == "entropic" and duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp), torch.zeros_like(py_samp))  # shape nx, ny
        if uot_mode_feat == "entropic" and duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat), torch.zeros_like(py_feat))  # shape dx, dy

    elif "mm" in uot_mode:
        self_uot_mm = partial(uot_mm, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

    hyperparams = (rho, eps)
    self_get_local_cost = partial(get_local_cost, hyperparams=hyperparams, entropic_mode=entropic_mode)
    self_get_cost = partial(get_cost, data=data, data_T=data_T, tuple_pxy_samp=tuple_pxy_samp, \
                tuple_pxy_feat=tuple_pxy_feat, hyperparams=hyperparams, entropic_mode=entropic_mode)

    # initialise log
    log_cost = []
    log_ent_cost = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in tqdm(range(nits_bcd)):
        pi_samp_prev = pi_samp.detach().clone()

        # Update pi_feat (feature coupling)
        mass = pi_samp.sum()
        new_rho1 = rho1 * mass + rho1_feat
        new_rho2 = rho2 * mass + rho2_feat
        new_eps = mass * eps_feat if entropic_mode == "joint" else eps_feat
        uot_cost = self_get_local_cost(data_T, pi_samp, tuple_pxy_samp)  # size dx x dy
        uot_params = (new_rho1, new_rho2, new_eps)

        if uot_mode_feat == "entropic":
            duals_feat, pi_feat = self_uot_ent(uot_cost, duals_feat, tuple_log_pxy_feat, uot_params)
        elif uot_mode_feat == "mm":
            duals_feat, pi_feat = self_uot_mm(uot_cost, pi_feat, tuple_pxy_feat, uot_params)

        if mass_rescaling:
            pi_feat = (mass / pi_feat.sum()).sqrt() * pi_feat  # shape dx x dy

        # Update pi (sample coupling)
        mass = pi_feat.sum()
        new_rho1 = rho1 * mass + rho1_samp
        new_rho2 = rho2 * mass + rho2_samp
        new_eps = mass * eps_samp if entropic_mode == "joint" else eps_samp
        uot_cost = self_get_local_cost(data, pi_feat, tuple_pxy_feat)  # size nx x ny
        uot_params = (new_rho1, new_rho2, new_eps)

        if uot_mode_samp == "entropic":
            duals_samp, pi_samp = self_uot_ent(uot_cost, duals_samp, tuple_log_pxy_samp, uot_params)
        elif uot_mode_samp == "mm":
            duals_samp, pi_samp = self_uot_mm(uot_cost, pi_samp, tuple_pxy_samp, uot_params)

        if mass_rescaling:
            pi_samp = (mass / pi_samp.sum()).sqrt() * pi_samp  # shape nx x ny

        if idx % eval_bcd == 0:
            # Update error
            err = (pi_samp - pi_samp_prev).abs().sum().item()
            cost, ent_cost = self_get_cost(pi_samp, pi_feat)
            log_cost.append(cost)
            log_ent_cost.append(ent_cost)

            if err < tol_bcd or abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_tol:
                break

            if verbose:
                print("Cost at iteration {}: {}".format(idx+1, cost))

    if pi_samp.isnan().any() or pi_feat.isnan().any():
        print("There is NaN in coupling")

    if log:
        return (pi_samp, pi_feat), (duals_samp, duals_feat), log_cost, log_ent_cost[1:]
    else:
        return (pi_samp, pi_feat), (duals_samp, duals_feat)