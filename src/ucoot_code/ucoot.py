import torch
from functools import partial

from ucoot_code.utils import quad_kl, kl, approx_kl

# require torch >= 1.9

# This implementation is a special case of FUCOOT.
# - supports both type of regularisation: independent or joint.
# - supports semi-relaxed (by setting rho=0) and COOT (by setting rho=infty).
# - supports using pretrained models (use warmstart dual and coupling).
# - supports epsilon-scaling trick for small epsilon.
# - supports C_linear for both sample and feature (similar to experiement HDA in COOT).
class UCOOT:
    def __init__(self, nits=100, nits_sinkhorn=100, tol=1e-7, tol_sinkhorn=1e-7):
        self.nits = nits
        self.nits_sinkhorn = nits_sinkhorn
        self.tol = tol
        self.tol_sinkhorn = tol_sinkhorn

    def sinkhorn(cost, init_duals, tuple_p, params, n_iters, tol, eval_freq):
        """
        Sinkhorn algorithm
        Allow rho1 and/or rho2 to be infinity but epsilon must be strictly positive.
        """
        rho1, rho2, eps = params
        a, b, ab = tuple_p
        f, g = init_duals
        log_a, log_b = a.log(), b.log()

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

    def get_local_cost(self, data, pi, tuple_p):
        """
        """

        (rho_x, rho_y), (eps_samp, _), _ = self.hyperparams
        a, b, ab = tuple_p
        X_sqr, Y_sqr, X, Y = data

        pi1, pi2 = pi.sum(1), pi.sum(0)
        A = X_sqr @ pi1
        B = Y_sqr @ pi2
        ucoot_cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T

        if rho_x != float("inf") and rho_x != 0:
            ucoot_cost += rho_x * approx_kl(pi1, a)
        if rho_y != float("inf") and rho_y != 0:
            ucoot_cost += rho_y * approx_kl(pi2, b)

        if self.reg_mode == "joint":
            ucoot_cost += eps_samp * approx_kl(pi, ab)

        return ucoot_cost

    def get_loss(self, C_linear, data, tuple_pxy_samp, tuple_pxy_feat, pi_samp, pi_feat):
        """
        """

        (rho_x, rho_y), (eps_samp, eps_feat) = self.hyperparams
        C_samp, C_feat = C_linear
        px_samp, py_samp, pxy_samp = tuple_pxy_samp
        px_feat, py_feat, pxy_feat = tuple_pxy_feat
        X_sqr, Y_sqr, X, Y = data

        pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)
        pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)

        A_sqr = (X_sqr @ pi1_feat).dot(pi1_samp)
        B_sqr = (Y_sqr @ pi2_feat).dot(pi2_samp)
        AB = (X @ pi_feat @ Y.T) * pi_samp
        ucoot_cost = A_sqr + B_sqr - 2 * AB.sum()

        if C_samp != 0:
            ucoot_cost += (pi_samp * C_samp).sum()
        if C_feat != 0:
            ucoot_cost += (pi_feat * C_feat).sum()

        if rho_x != float("inf") and rho_x != 0:
            ucoot_cost += rho_x * quad_kl(pi1_samp, pi1_feat, px_samp, px_feat)
        if rho_y != float("inf") and rho_y != 0:
            ucoot_cost += rho_y * quad_kl(pi2_samp, pi2_feat, py_samp, py_feat)

        if self.reg_mode == "joint":
            ent_cost = ucoot_cost + eps_samp * quad_kl(pi_samp, pi_feat, pxy_samp, pxy_feat)
        elif self.reg_mode == "independent":
            ent_cost = ucoot_cost + eps_samp * kl(pi_samp, pxy_samp) + \
                eps_feat * kl(pi_feat, pxy_feat)

        return ucoot_cost, ent_cost

    def get_barycentre(self, Xt, pi_samp):
        """
        Calculate the barycentre by the following formula: diag(1 / P1_{n_2}) P Xt
        (need to be typed in latex).

        Parameters
        ----------
        Xt: target data of size ny x dy, NOT to be confused with the target distance matrix.
        pi_samp: optimal plan of size nx x ny.

        Returns
        ----------
        Barycentre of size nx x dy
        """

        barycentre = pi_samp @ Xt / pi_samp.sum(1).reshape(-1, 1)

        return barycentre

    def solver(
        self,
        X,
        Y,
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf")),
        eps=(1e-2, 1e-2),
        C_linear=(0, 0),
        reg_mode="joint",
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=False,
        early_stopping_threshold=1e-6,
        eval_freq_loss=10,
        eval_freq_sinkhorn=1
    ):
        """
        Parameters for some frequently used modes:
        - Entropic fused COOT or FGW: rho=(infty, infty), mode="independent" and C_linear=(tensor, 0)
        - Entropic GW: rho=(infty, infty), mode="independent".
        - Ent-semi-relaxed FGW: rho=(0, infty) or rho=(infty,0).

        Parameters
        ----------
        X: matrix of size nx x dx.
        Y: matrix of size ny x dy.
        px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        rho: tuple of 2 relaxation parameters (rho_x, rho_y) for UCOOT.
        eps: tuple of regularisation parameters (eps_samp, eps_feat) for entropic approximation.
            If mode == "joint" then only eps_samp is used.
        C_linear: tuple of tensors of size (nx x ny) and (dx x dy): the linear terms in the fused formulation.
        reg_mode:
            reg_mode="joint": use UGW-like regularisation term.
            reg_mode = "independent": use COOT-like regularisation.
        init_pi: tuple of initialisation for sample and feature couplings:
            matrices of size (nx x ny) and (dx x dy). If not available then assign None.
        init_duals: tuple of two tuples containing initialisation of duals for Sinkhorn algorithm.
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        early_stopping_threshold: trigger early stopping if the absolute difference between the two most
            recent loss is smaller than this threshold.
        eval_freq_loss: The multiplier of iteration at which the loss is calculated, i.e.
            if eval_freq_loss = 10, then the loss is calculated at iteration 10, 20, 30, etc...
        eval_freq_sinkhorn: The multiplier of iteration at which the change of dual is calculated
            in the Sinkhorn algorithm.

        Returns
        ----------
        (pi_samp, pi_feat): tuple of sample (size nx x ny) and feature (size dx x dy) matrices.
        (duals_samp, duals_feat): tuple of two tuples of duals.
        log_cost: if log is True, return a list of loss (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        self.reg_mode = reg_mode

        nx, dx = X.shape
        ny, dy = Y.shape
        device, dtype = X.device, X.dtype

        # hyper-parameters
        if isinstance(eps, float) or isinstance(eps, int):
            eps = (eps, eps)
        if not isinstance(eps, tuple):
            raise ValueError("Epsilon must be either a scalar or a tuple.")

        # some constants
        C_samp, C_feat = C_linear
        if C_samp is None:
            C_samp = 0
        if C_feat is None:
            C_feat = 0

        eps_samp, eps_feat = eps
        rho_x, rho_y = rho
        self.hyperparams = (rho, eps)

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

        # constant data variables
        X_sqr = X ** 2
        Y_sqr = Y ** 2
        data = (X_sqr, Y_sqr, X, Y)
        data_T = (X_sqr.T, Y_sqr.T, X.T, Y.T)

        # initialise coupling and dual vectors
        pi_samp, _ = init_pi
        if pi_samp is None:
            pi_samp = px_samp[:, None] * py_samp[None, :]  # size nx x ny

        duals_samp, duals_feat = init_duals
        if duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp), torch.zeros_like(py_samp))  # shape nx, ny
        if duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat), torch.zeros_like(py_feat))  # shape dx, dy

        sinkhorn_solver = partial(self.sinkhorn, n_iters=self.nits_sinkhorn, tol=self.tol_sinkhorn, \
                                eval_freq=eval_freq_sinkhorn)

        # initialise log
        log_cost = []
        log_ent_cost = []
        i = 1
        err = self.tol + 1e-3

        while (err > self.tol) and (i <= self.nits):
            pi_samp_prev = pi_samp.detach().clone()

            # Update pi_feat (feature coupling)
            mass = pi_samp.sum()
            new_rho_x, new_rho_y = rho_x * mass, rho_y * mass
            new_eps = mass * eps[0] if reg_mode == "joint" else eps_feat
            new_cost = self.get_local_cost(data_T, pi_samp, tuple_pxy_samp) + C_feat  # size dx x dy

            duals_feat, pi_feat = sinkhorn_solver(new_cost, duals_feat, tuple_pxy_feat, new_rho_x, new_rho_y, new_eps)
            pi_feat = (mass / pi_feat.sum()).sqrt() * pi_feat  # shape dx x dy

            # Update pi (sample coupling)
            mass = pi_feat.sum()
            new_rho_x, new_rho_y = rho_x * mass, rho_y * mass
            new_eps = mass * eps[0] if reg_mode == "joint" else eps_samp
            new_cost = self.get_local_cost(data, pi_feat, tuple_pxy_feat) + C_samp  # size nx x ny

            duals_samp, pi_samp = sinkhorn_solver(new_cost, duals_samp, tuple_pxy_samp, new_rho_x, new_rho_y, new_eps)
            pi_samp = (mass / pi_samp.sum()).sqrt() * pi_samp  # shape nx x ny

            # Update error
            err = (pi_samp - pi_samp_prev).abs().sum().item()
            if log and (i % eval_freq_loss == 0):
                cost, ent_cost = self.get_loss(C_linear, data, tuple_pxy_samp, tuple_pxy_feat, pi_samp, pi_feat)

                log_cost.append(cost.item())
                log_ent_cost.append(ent_cost.item())
                if (
                    len(log_ent_cost) >= 2
                    and abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_threshold
                ):
                    break

                if verbose:
                    print("Entropic cost at iteration {}: {}.".format(i, cost.item()))

            i += 1

        if pi_samp.isnan().any() or pi_feat.isnan().any():
            print("There is NaN in coupling.")

        if log:
            return (pi_samp, pi_feat), (duals_samp, duals_feat), log_cost, log_ent_cost
        else:
            return (pi_samp, pi_feat), (duals_samp, duals_feat)

    def faster_solver(
        self,
        X,
        Y,
        C_linear=(0, 0),
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf")),
        eps=(1e-2, 1e-2),
        reg_mode="joint",
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=False,
        early_stopping_threshold=1e-6,
        eval_freq_loss=10,
        eval_freq_sinkhorn=1,
        eps_step=10,
        init_eps=1e-1,
        niter_warmstart_sinkhorn=10
    ):
        """
        Solver with warm start for small epsilon, for reg_mode="joint",
        or "independent" with two equals epsilons.
        """

        if isinstance(eps, float) or isinstance(eps, int):
            eps = (eps, eps)
        if not isinstance(eps, tuple):
            raise ValueError("Epsilon must be either a scalar or a tuple")

        if eps[0] < init_eps:

            nits = self.nits
            nits_sinkhorn = self.nits_sinkhorn

            self.nits = niter_warmstart_sinkhorn
            self.nits_sinkhorn = niter_warmstart_sinkhorn

            while (init_eps > eps[0]):
                init_pi, init_duals = self.solver(X=X, Y=Y, px=px, py=py, C_linear=C_linear, rho=rho, \
                                    eps=init_eps, reg_mode=reg_mode, init_pi=init_pi, \
                                    init_duals=init_duals, log=False, verbose=False, \
                                    eval_freq_loss=niter_warmstart_sinkhorn, \
                                    eval_freq_sinkhorn=niter_warmstart_sinkhorn)
                init_eps /= eps_step

            self.nits = nits
            self.nits_sinkhorn = nits_sinkhorn

        return self.solver(X=X, Y=Y, C_linear=C_linear, px=px, py=py, rho=rho, eps=eps, \
                    reg_mode=reg_mode, init_pi=init_pi, init_duals=init_duals, log=log, \
                    verbose=verbose, early_stopping_threshold=early_stopping_threshold, \
                    eval_freq_loss=eval_freq_loss, eval_freq_sinkhorn=eval_freq_sinkhorn)


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # generate simulated data
    nx = 2087
    dx = 3
    ny = 1561
    dy = 2

    x = torch.rand(nx, dx).to(device)
    y = torch.rand(ny, dy).to(device)
    Cx = torch.cdist(x, x, p=2)**2
    Cy = torch.cdist(y, y, p=2)**2

    # important parameters
    rho_x = float("inf")
    rho_y = float("inf")
    rho = (rho_x, rho_y)

    eps_samp = 1e-2
    eps_feat = 1e-1
    eps = (eps_samp, eps_feat)

    mode = "joint"

    # optional parameters
    C_samp = torch.rand(nx, ny).to(device)
    C_feat = torch.rand(nx, ny).to(device)
    C_linear = (C_samp, C_feat)

    ucoot = UCOOT(nits=100, nits_sinkhorn=500, tol=1e-7, tol_sinkhorn=1e-7)
    (pi_samp, pi_feat), (duals_samp, duals_feat), log_cost, log_ent_cost = \
        ucoot.solver(X=Cx, Y=Cy, C_linear=C_linear, rho=rho, eps=eps, reg_mode=mode, log=True, \
            verbose=True, early_stopping_threshold=1e-6, eval_freq_loss=1, eval_freq_sinkhorn=10)

    print("===============")

    # test faster solver
    eps_step = 5
    eps = 1e-2
    init_eps = 1
    niter_warmstart_sinkhorn=10
    (pi_samp, pi_feat), (duals_samp, duals_feat), log_cost_ws, log_ent_cost_ws = \
        ucoot.faster_solver(X=Cx, Y=Cy, C_linear=C_linear, rho=rho, eps=eps, reg_mode=mode,
                            log=True, verbose=True, early_stopping_threshold=1e-6, \
                            eval_freq_loss=1, eval_freq_sinkhorn=10, eps_step=eps_step, \
                            init_eps=init_eps, niter_warmstart_sinkhorn=niter_warmstart_sinkhorn)
