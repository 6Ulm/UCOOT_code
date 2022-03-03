from functools import partial
import torch
from ucoot_code.megawass import MegaWass

# just to ignore Scikit-learn depreciation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.cluster import KMeans


class Barycenter(MegaWass):
    def __init__(self, nits_bcd=100, tol_bcd=1e-7, eval_bcd=5, \
                nits_uot=100, tol_uot=1e-7, eval_uot=1, \
                nits_bary=100, tol_bary=1e-7):
        """
        Init
        """

        self.nits_bcd = nits_bcd
        self.tol_bcd = tol_bcd
        self.eval_bcd = eval_bcd

        self.nits_uot = nits_uot
        self.tol_uot = tol_uot
        self.eval_uot = eval_uot

        self.nits_bary = nits_bary
        self.tol_bary = tol_bary

    @staticmethod
    def get_factored_distance(X, Y):
        """
        Write square Euclidean distance matrix as product of two low rank matrices.
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

    @staticmethod
    def update_bary(list_pi, list_weight, list_C):
        bary = 0
        # pi_samp, pi_feat: both of size (ns, n)
        for pi, w, C in zip(list_pi, list_weight, list_C):
            pi_samp, pi_feat = pi
            if isinstance(C, tuple):
                C1, C2 = C
                bary = bary + w * pi_samp.T @ (C1 @ (C2.T @ pi_feat)) / (pi_samp.sum(0)[:,None] * pi_feat.sum(0)[None,:]) # shape (n, n)
            elif torch.is_tensor(C):
                bary = bary + w * pi_samp.T @ C @ pi_feat / (pi_samp.sum(0)[:,None] * pi_feat.sum(0)[None,:]) # shape (n, n)
        return bary

    @staticmethod
    def update_bary_attr(list_pi, list_weight, list_attr):
        bary_attr = 0
        for pi, w, attr in zip(list_pi, list_weight, list_attr):
            pi_samp = pi[0]
            if attr is not None:
                bary_attr = bary_attr + w * (pi_samp.T @ attr) * pi_samp.sum(0).reshape(-1, 1)

        return bary_attr

    @staticmethod
    def get_dim(C):
        if isinstance(C, tuple):
            return C[0].shape[0]
        elif torch.is_tensor(C):
            return C.shape[0]

    @staticmethod
    def get_device_dtype(C):
        if isinstance(C, tuple):
            return C[0].device, C[0].dtype
        elif torch.is_tensor(C):
            return C.device, C.dtype

    def train_individual(self, list_C, list_attr, list_p, list_params, \
                        list_pi, list_dual, bary_attr, bary):

        list_pi = []
        list_dual = []
        list_cost = []

        for C, C_attr, px, params, pi, dual in zip(list_C, list_attr, list_p, list_params, list_pi, list_dual):
            rho, eps, alpha = params
            D = torch.cdist(C_attr, bary_attr) if C_attr is not None else None
            pi, dual, _, cost = self.fugw_solver(X=C, Y=bary, px=px, rho=rho, eps=eps, alpha=alpha, \
                        D=D, init_pi=pi, init_dual=dual)

            list_pi.append(pi)
            list_dual.append(dual)
            list_cost.append(cost[-1])

        return list_pi, list_dual, list_cost

    def init_bary_attr(self, list_weight, list_attr, bary_dim, random_state):
        list_weighted_attr = [w * attr for (w, attr) in zip(list_weight, list_attr) \
                            if attr is not None]
        attr_data = torch.concat(list_weighted_attr, dim=0).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=bary_dim, random_state=random_state).fit(attr_data)
        bary_attr = torch.from_numpy(kmeans.cluster_centers_)
        return bary_attr

    def barycenter(self, list_weight, list_C, list_rho, list_eps, list_p=None, \
                list_alpha=None, list_attr=None, bary_dim=2, bary=None, bary_p=None, \
                bary_attr=None, log=True, verbose=True, early_stopping_tol=1e-6, \
                uot_mode="entropic", entropic_mode="joint", mass_rescaling=True, random_state=42):
        """
        (Cs, C)
        """

        device, dtype = self.get_device_dtype(list_C[0])

        # init of lists
        list_dim = [self.get_dim(C) for C in list_C]
        if list_p is None:
            list_p = [torch.ones(n).to(device).to(dtype) / n for n in list_dim]

        if list_alpha is None:
            list_alpha = [0] * len(list_C)

        have_attr = True
        if list_attr is None:
            have_attr = False
            list_attr = [None] * len(list_C)

        list_params = list(zip(list_rho, list_eps, list_alpha))

        # init methods
        train = partial(self.train_individual, list_C=list_C, list_attr=list_attr, \
                        list_p=list_p, list_params=list_params)
        megawass = MegaWass(nits_bcd=self.nits_bcd, tol_bcd=self.tol_bcd, eval_bcd=self.eval_bcd, \
                            nits_uot=self.nits_uot, tol_uot=self.tol_uot, eval_uot=self.eval_uot)
        self.fugw_solver = partial(megawass.solver_fugw_full, py=bary_p, log=True, uot_mode=uot_mode, \
                                entropic_mode=entropic_mode, verbose=False, \
                                early_stopping_tol=early_stopping_tol, mass_rescaling=mass_rescaling)

        # init of barycenter
        if bary_p is None:
            bary_p = torch.ones(bary_dim).to(device).to(dtype) / bary_dim
        if bary is None:
            bary = torch.ones(bary_dim, bary_dim).to(device).to(dtype)
        if bary_attr is None and have_attr:
            bary_attr = self.init_bary_attr(list_weight, list_attr, bary_dim, random_state)
            bary_attr = bary_attr.to(device).to(dtype)

        # init training variables
        list_pi = [p[:,None] * bary_p[None,:] for p in list_p]
        list_pi = [(pi, pi) for pi in list_pi]

        list_dual = [(torch.zeros(n).to(device).to(dtype), \
                    torch.zeros(bary_dim).to(device).to(dtype)) for n in list_dim]
        list_dual = [(dual, dual) for dual in list_dual]

        log_cost = [float("inf")]

        for idx in range(self.nits_bary):

            list_pi, list_dual, list_cost = train(list_pi=list_pi, list_dual=list_dual, \
                                                bary_attr=bary_attr, bary=bary)

            bary = self.update_bary(list_pi, list_weight, list_C)
            if have_attr:
                bary_attr = self.update_bary_attr(list_pi, list_weight, list_attr)

            cost = sum(cost * w for (cost, w) in zip(list_cost, list_weight))

            if verbose:
                print("Cost at iteration {} = {}".format(idx+1, cost))

            log_cost.append(cost)
            if abs(log_cost[-2] - log_cost[-1]) < self.tol_bary:
                break

        if log:
            return bary, bary_attr, log_cost[1:]
        else:
            return bary, bary_attr
