from warnings import WarningMessage
from .utils import solver
import torch

class MegaWass:
    def __init__(self, nits_bcd=100, tol_bcd=1e-7, eval_bcd=5, nits_uot=100, tol_uot=1e-7, eval_uot=1):
        """
        Init
        """

        self.nits_bcd = nits_bcd
        self.tol_bcd = tol_bcd
        self.eval_bcd = eval_bcd

        self.nits_uot = nits_uot
        self.tol_uot = tol_uot
        self.eval_uot = eval_uot

    def get_barycentre(self, Xt, pi_samp):
        """
        Calculate the barycentre by the following formula: diag(1 / P1_{n_2}) P Xt
        (need to be typed in latex).

        Parameters
        ----------
        Xt: target data of size ny x dy.
        pi_samp: optimal plan of size nx x ny.

        Returns
        ----------
        Barycentre of size nx x dy
        """

        barycentre = pi_samp @ Xt / pi_samp.sum(1).reshape(-1, 1)

        return barycentre

    def solver_megawass(
        self,
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
        mass_rescaling=True
    ):
        """
        Parameters for mode:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho1 != infty, rho2 != infty. No need to care about rho1_samp and rho2_samp.
        - EGW: alpha = 1, mode = "independent", rho1 = rho2 = infty. No need to care about rho1_samp and rho2_samp.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent", rho1 = rho2 = infty (so rho1_samp = rho2_samp = infty)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent", (rho1 = 0, rho2 = infty), or (rho1 = infty, rho2 = 0).
        No need to care about rho1_samp and rho2_samp.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent", (rho1 = rho1_samp = 0, rho2 = rho2_samp = infty),
        or (rho1 = rho1_samp = infty, rho2 = rho2_samp = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None, rho1 != infty, rho2 != infty, rho1_samp != infty, rho2_samp != infty.

        Parameters
        ----------
        X: matrix of size nx x dx
        Y: matrix of size ny x dy
        D: matrix of size nx x ny. Sample matrix, in case of fused GW
        px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        rho: tuple of 6 relaxation parameters for UGW and UOT.
        eps: regularisation parameter for entropic approximation.
        alpha: between 0 and 1. Interpolation parameter for fused UGW.
        entropic_mode:
            entropic_mode="joint": use UGW-like regularisation term
            entropic_mode = "independent": use COOT-like regularisation
        init_n: matrix of size nx x ny if not None. Initialisation matrix for sample coupling.
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        eval_bcd: The multiplier of iteration at which the loss is calculated. For example, if eval_bcd = 10, then the
                    loss is calculated at iteration 10, 20, 30, etc...

        Returns
        ----------
        pi_samp: matrix of size nx x ny. Sample matrix.
        pi_feat: matrix of size dx x dy. Feature matrix.
        log_cost: if log is True, return a list of loss (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """

        return solver(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, init_pi, \
                    init_dual, log, verbose, early_stopping_tol, mass_rescaling, \
                    eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, \
                    nits_bcd=self.nits_bcd, tol_uot=self.tol_uot, nits_uot=self.nits_uot)

    def solver_fcoot(
        self,
        X,
        Y,
        px=(None, None),
        py=(None, None),
        eps=(1e-2, 1e-2),
        alpha=(1, 1),
        D=(None, None),
        init_pi=(None, None),
        init_dual=(None, None),
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        mass_rescaling=True
    ):
        """
        If you want to use fused COOT, it is recommended to use COOT from POT because it is
        much more optimised.
        """

        rho = (float("inf"), float("inf"), 0, 0, 0, 0)
        uot_mode = ("entropic", "entropic")
        entropic_mode = "independent"

        return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
                                init_pi, init_dual, log, verbose, early_stopping_tol, mass_rescaling)

    def solver_fucoot(
        self,
        X,
        Y,
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf")),
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
        mass_rescaling=True
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
        init_dual: tuple of two tuples containing initialisation of duals for Sinkhorn algorithm.
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

        rho1, rho2 = rho
        rho = (rho1, rho2, 0, 0, 0, 0)

        return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
                                init_pi, init_dual, log, verbose, early_stopping_tol, mass_rescaling)

    def solver_fgw(
        self,
        X,
        Y,
        px=None,
        py=None,
        eps=1e-2,
        alpha=1,
        D=None,
        init_pi=None,
        init_dual=None,
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        mass_rescaling=True
    ):
        """
        If you want to use fused GW, it is recommendeded to use COOT from POT because it is
        more optimised.
        """

        if isinstance(X, tuple):
            X1, X2 = X
            nx, dx = X1.shape[0], X2.shape[0]
        elif torch.is_tensor(X):
            nx, dx = X.shape
        else:
            raise ValueError("Invalid type of input.")

        ny, dy = Y.shape
        if nx != dx or ny != dy:
            raise ValueError("Input matrix is not squared.")

        px, py, D, alpha = (px, px), (py, py), (D, D), (alpha, alpha)
        init_dual = (init_dual, init_dual)
        init_pi = (init_pi, init_pi)
        uot_mode = ("entropic", "entropic")
        entropic_mode = "independent"
        rho = (float("inf"), float("inf"), 0, 0, 0, 0)

        return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
                                init_pi, init_dual, log, verbose, early_stopping_tol, mass_rescaling)

    def solver_fugw_simple(
        self,
        X,
        Y,
        px=None,
        py=None,
        rho=(float("inf"), float("inf")),
        uot_mode="entropic",
        eps=1e-2,
        entropic_mode="joint",
        alpha=1,
        D=None,
        init_pi=(None, None),
        init_dual=(None, None),
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        mass_rescaling=True
    ):
        """
        Simple Fused UGW (no KL term in the UOT). Similar to FUCOOT.
        """

        if isinstance(X, tuple):
            X1, X2 = X
            nx, dx = X1.shape[0], X2.shape[0]
        elif torch.is_tensor(X):
            nx, dx = X.shape
        else:
            raise ValueError("Invalid type of input.")

        ny, dy = Y.shape
        if nx != dx or ny != dy:
            raise ValueError("The input matrix is not squared.")

        px, py, D, eps, alpha = (px, px), (py, py), (D, D), (eps, eps), (alpha, alpha)
        uot_mode = (uot_mode, uot_mode)
        rho1, rho2 = rho
        rho = (rho1, rho2, 0, 0, 0, 0)

        return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
                                init_pi, init_dual, log, verbose, early_stopping_tol, mass_rescaling)

    def solver_fugw_full(
        self,
        X,
        Y,
        px=None,
        py=None,
        rho=(float("inf"), float("inf"), 0, 0),
        uot_mode="entropic",
        eps=1e-2,
        entropic_mode="joint",
        alpha=1,
        D=None,
        init_pi=(None, None),
        init_dual=(None, None),
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        mass_rescaling=True
    ):
        """
        Complete Fused UGW
        """

        if isinstance(X, tuple):
            X1, X2 = X
            nx, dx = X1.shape[0], X2.shape[0]
        elif torch.is_tensor(X):
            nx, dx = X.shape
        else:
            raise ValueError("Invalid type of input.")

        ny, dy = Y.shape
        if nx != dx or ny != dy:
            raise ValueError("The input matrix is not squared.")

        px, py, D, eps, alpha = (px, px), (py, py), (D, D), (eps, eps), (alpha, alpha)
        uot_mode = (uot_mode, uot_mode)
        rho1, rho2, rho3, rho4 = rho
        rho = (rho1, rho2, rho3, rho4, rho3, rho4)

        return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
                                    init_pi, init_dual, log, verbose, early_stopping_tol, mass_rescaling)

    ##################################
    ##################################
    # For uot_mode = "entropic",
    # use warmstart for small epsilon
    ##################################
    ##################################

    # def faster_solver_megawass(
    #     self,
    #     X,
    #     Y,
    #     px=(None, None),
    #     py=(None, None),
    #     rho=(float("inf"), float("inf"), 0, 0, 0, 0),
    #     uot_mode=("entropic", "entropic"),
    #     eps=(1e-2, 1e-2),
    #     entropic_mode="joint",
    #     alpha=(1, 1),
    #     D=(None, None),
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Solver with warm start for small epsilon,
    #     for entropic_mode="joint", or "independent" with two equals epsilons.
    #     """

    #     if isinstance(eps, float) or isinstance(eps, int):
    #         eps = (eps, eps)
    #     if not isinstance(eps, tuple):
    #         raise ValueError("Epsilon must be either a scalar or a tuple.")
    #     # if use joint penalisation for couplings, then only use the first value epsilon.
    #     if entropic_mode == "joint":
    #         eps = (eps[0], eps[0])

    #     if isinstance(eps_step, float) or isinstance(eps_step, int):
    #         eps_step = (eps_step, eps_step)
    #     if not isinstance(eps_step, tuple):
    #         raise ValueError("eps_step must be either a scalar or a tuple.")
    #     # if use joint penalisation for couplings, then only use the first value epsilon.
    #     if entropic_mode == "joint":
    #         eps_step = (eps_step[0], eps_step[0])

    #     if isinstance(init_eps, float) or isinstance(init_eps, int):
    #         init_eps = (init_eps, init_eps)
    #     if not isinstance(init_eps, tuple):
    #         raise ValueError("init_eps must be either a scalar or a tuple.")
    #     # if use joint penalisation for couplings, then only use the first value epsilon.
    #     if entropic_mode == "joint":
    #         init_eps = (init_eps[0], init_eps[0])

    #     if isinstance(uot_mode, str):
    #         uot_mode = (uot_mode, uot_mode)
    #     if not isinstance(uot_mode, tuple):
    #         raise ValueError("uot_mode must be either a string or a tuple of strings.")

    #     # some constants
    #     rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    #     eps_samp, eps_feat = eps
    #     init_eps_samp, init_eps_feat = init_eps
    #     eps_step_samp, eps_step_feat = eps_step

    #     uot_mode_samp, uot_mode_feat = uot_mode
    #     if (eps_samp == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0) or \
    #         (eps_feat == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0):
    #         raise ValueError("Invalid values for epsilon and rho. \
    #                         Cannot contain zero in epsilon AND infinity in rho at the same time.")
    #     else:
    #         if eps_samp == 0:
    #             uot_mode_samp = "mm"
    #         if eps_feat == 0:
    #             uot_mode_feat = "mm"
    #         if torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0:
    #             uot_mode_samp = "entropic"
    #         if torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0:
    #             uot_mode_feat = "entropic"
    #     uot_mode = (uot_mode_samp, uot_mode_feat)

    #     if uot_mode == "entropic" and eps[0] < init_eps:

    #         nits_bcd = self.nits_bcd
    #         nits_uot = self.nits_uot
    #         eval_bcd = self.eval_bcd
    #         eval_uot = self.eval_uot

    #         self.nits_bcd = niter_warmstart_uot
    #         self.nits_uot = niter_warmstart_uot
    #         self.eval_bcd = niter_warmstart_uot
    #         self.eval_bcd = niter_warmstart_uot

    #         while (init_eps > eps[0]):
    #             init_pi, init_dual = \
    #                 self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                     D, init_pi, init_dual, log=False, verbose=False)

    #             init_eps /= eps_step

    #         self.nits_bcd = nits_bcd
    #         self.nits_uot = nits_uot
    #         self.eval_bcd = eval_bcd
    #         self.eval_uot = eval_uot

    #     return self.solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, \
    #         init_pi, init_dual, log, verbose, early_stopping_tol)

    # def faster_solver_fcoot(
    #     self,
    #     X,
    #     Y,
    #     px=(None, None),
    #     py=(None, None),
    #     eps=(1e-2, 1e-2),
    #     entropic_mode="joint",
    #     alpha=(1, 1),
    #     D=(None, None),
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Faster solver for Fused COOT
    #     """

    #     rho = (float("inf"), float("inf"), 0, 0, 0, 0)
    #     uot_mode = "entropic"

    #     return self.faster_solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                                     D, init_pi, init_dual, log, verbose, early_stopping_tol, \
    #                                     eps_step, init_eps, niter_warmstart_uot)

    # def faster_solver_fucoot(
    #     self,
    #     X,
    #     Y,
    #     px=(None, None),
    #     py=(None, None),
    #     rho=(float("inf"), float("inf")),
    #     uot_mode=("entropic", "entropic"),
    #     eps=(1e-2, 1e-2),
    #     entropic_mode="joint",
    #     alpha=(1, 1),
    #     D=(None, None),
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Faster solver for Fused UCOOT
    #     """

    #     rho1, rho2 = rho
    #     rho = (rho1, rho2, 0, 0, 0, 0)
    #     uot_mode = "entropic"

    #     return self.faster_solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                                     D, init_pi, init_dual, log, verbose, early_stopping_tol, \
    #                                     eps_step, init_eps, niter_warmstart_uot)

    # def faster_solver_fgw(
    #     self,
    #     X,
    #     Y,
    #     px=None,
    #     py=None,
    #     eps=1e-2,
    #     alpha=1,
    #     D=None,
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Faster solver for fused GW
    #     """

    #     px, py, D = (px, px), (py, py), (D, D)
    #     rho = (float("inf"), float("inf"), 0, 0, 0, 0)
    #     entropic_mode = "joint"
    #     uot_mode = "entropic"

    #     return self.faster_solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                                     D, init_pi, init_dual, log, verbose, early_stopping_tol, \
    #                                     eps_step, init_eps, niter_warmstart_uot)

    # def faster_solver_fugw_simple(
    #     self,
    #     X,
    #     Y,
    #     px=None,
    #     py=None,
    #     rho=(float("inf"), float("inf")),
    #     eps=(1e-2, 1e-2),
    #     entropic_mode="joint",
    #     alpha=1,
    #     D=None,
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Faster solver for simple Fused UGW
    #     """

    #     px, py, D = (px, px), (py, py), (D, D)
    #     rho1, rho2 = rho
    #     rho = (rho1, rho2, 0, 0, 0, 0)
    #     uot_mode = "entropic"

    #     return self.faster_solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                                     D, init_pi, init_dual, log, verbose, early_stopping_tol, \
    #                                     eps_step, init_eps, niter_warmstart_uot)

    # def faster_solver_fugw_full(
    #     self,
    #     X,
    #     Y,
    #     px=None,
    #     py=None,
    #     rho=(float("inf"), float("inf"), 0, 0),
    #     eps=(1e-2, 1e-2),
    #     entropic_mode="joint",
    #     alpha=1,
    #     D=None,
    #     init_pi=(None, None),
    #     init_dual=(None, None),
    #     log=False,
    #     verbose=False,
    #     early_stopping_tol=1e-6,
    #     eps_step=10,
    #     init_eps=1,
    #     niter_warmstart_uot=10
    # ):
    #     """
    #     Faster solver for simple Fused UGW
    #     """

    #     px, py, D = (px, px), (py, py), (D, D)
    #     rho1, rho2, rho3, rho4 = rho
    #     rho = (rho1, rho2, rho3, rho4, rho3, rho4)
    #     uot_mode = "entropic"

    #     return self.faster_solver_megawass(X, Y, px, py, rho, uot_mode, eps, entropic_mode, alpha, \
    #                                     D, init_pi, init_dual, log, verbose, early_stopping_tol, \
    #                                     eps_step, init_eps, niter_warmstart_uot)
