"""
Implementation of Accelerated Iterative Thresholding
"""

import numpy as np
import torch

class BaseAIHT:
    def __init__(self, y, A, w, K, L=None, is_sparse=False, support=None , tol=1e-5, max_iter_num=300):
        self.y =y
        self.A = A
        self.w = w
        self.K = K
        self.L = L
        self.is_sparse = is_sparse
        self.support = support
        self.tol = tol
        self.max_iter_num = max_iter_num




def restore_selected_support(w_projected, selected_support, N):
    w_restored = torch.zeros([N, 1], dtype=w_projected.dtype, device=w_projected.device)
    w_restored[selected_support] = w_projected
    return w_restored


def get_selected_support(w, K):
    index_w = torch.argsort(w.squeeze(), descending=True)
    selected_support = index_w[:K].tolist()
    return selected_support


def project_w_to_k_sparse(w, selected_support):
    w_projected = torch.zeros_like(w)
    w_projected[selected_support] = w[selected_support].clamp(min=0)
    return w_projected


def project_w_to_l_constrained(w_selected, L):
    accumulate = torch.cumsum(w_selected, dim=0)
    rho = torch.argmax(w_selected > (accumulate - L) / torch.arange(1, len(w_selected) + 1))
    tau = (w_selected[:rho+1].sum() - L) / (rho + 1)
    w_projected = w_selected - tau
    w_projected = w_projected.clamp(min=0)
    return w_projected



class AcceleratedIHT(BaseAIHT):
    def __init__(self, y, A, w, K, L=None, is_sparse=False, support=None , tol=1e-5, max_iter_num=300):
        super().__init__(y, A, w, K, L,is_sparse, support, tol, max_iter_num)

        """
        The optimization objective is
            argmin_w ||y - A w||^2    s.t.    ||w||_0 <= K    and    w >= 0    (optional: and sum(w) = L)
                where   y is of shape (M, 1),
                        A is of shape (M, N),
                        w is of shape (N, 1),
                        K is a positive integer,
                        L is a positive number.
        """


        self.y =y
        self.A = A
        self.w = w
        self.K = K
        self.L = L
        self.is_sparse = is_sparse
        self.support = support
        self.tol = tol
        self.max_iter_num = max_iter_num


    def iterative_hardThresholdingObj(self):
        """
        Calculate the quadratic objective value given w
        :param y: numpy.ndarray of shape (M, 1)
        :param A: numpy.ndarray of shape (M, N)
        :param w: numpy.ndarray of shape (N, 1)
        :return: float objective value
        """
        return np.linalg.norm(self.y - self.A.dot(self.w), ord=2)


    def l2_projection_torch_2(self,w,K, is_sparse=False,support = None, L=None):
        """
        If L is None, project w to the K-sparsity constrained and non-negative region;
        if L is not None, project w the K-sparsity constrained, non-negative and the sum(w) = L region;
        the projection is optimal in l2 distance.
        :param w: torch.tensor of shape (N, 1)
        :param K: int (sparsity constraint), positive
        :param L: float, positive
        :param already_K_sparse: bool. If the input w has been already K-sparse, put 'True' to for a faster projection
        :param K_sparse_supp: list. If the input w has been already K-sparse, put its support here
        :return: w: torch.tensor of shape (N, 1). A new vector that is the projected w
                selected_support: list of integer indexes (the support of the w).
        """
        self.is_sparse = is_sparse
        self.support = support
        self.w = w
        self.K = K
        self.L = L
        device, dtype, N = self.w.device, self.w.dtype, self.w.shape[0]


        if self.L is None:
            if self.is_sparse:
                w_projected = self.w.clone()
                w_projected[w_projected < 0] = 0
                return w_projected, self.support

            selected_support = get_selected_support(self.w, self.K)
            w_projected = project_w_to_k_sparse(self.w, selected_support)
            return w_projected, selected_support
        else:
            if self.is_sparse:
                w_selected = self.w[self.support]
            else:
                self.support = get_selected_support(self.w, self.K)
                w_selected = self.w[self.support]

            w_projected = project_w_to_l_constrained(w_selected, self.L)
            w_projected = restore_selected_support(w_projected, self.support, N)
            return w_projected, self.support





    def l2_projection_numpy(self,w,K, is_sparse=False,support = None, L=None):
        """
        If L is None, project w to the K-sparsity constrained and non-negative region;
        if L is not None, project w the K-sparsity constrained, non-negative and the sum(w) = L region;
        the projection is optimal in l2 distance.
        :param w: numpy.ndarray of shape (N, 1)
        :param K: int (sparsity constraint), positive
        :param L: float, positive
        :param already_K_sparse: bool. If the input w has been already K-sparse, put 'True' to for a faster projection
        :param K_sparse_supp: list. If the input w has been already K-sparse, put its support here
        :return: w: numpy.ndarray of shape (N, 1). A new vector that is the projected w
                selected_support: list of integer indexes (the support of the w).
        """
        self.is_sparse = is_sparse
        self.support = support
        self.w = w
        self.K = K
        self.L = L
        N = self.w.shape[0]
        if self.L is None:
            if self.is_sparse:
                w_projected = self.w.copy()
                w_projected[w_projected < 0] = 0
                return w_projected, self.support
            else:
                index_w = np.flip(np.argsort(np.squeeze(self.w)))
                index_w = np.squeeze(index_w).tolist()
                w_projected = np.zeros([N, 1])
                selected_support = index_w[0:self.K]
                w_projected[selected_support] = self.w[index_w[0:self.K]]  # projection
                w_projected[w_projected < 0] = 0  # truncate negative entries
                return w_projected, selected_support
        else:
            if self.is_sparse:
                w_selected = self.w[self.support]
            else:
                index_w = np.flip(np.argsort(np.squeeze(self.w)))
                index_w = np.squeeze(index_w).tolist()
                K_sparse_supp = index_w[:self.K]
                w_selected = self.w[self.support]

            w_projected = np.zeros([N, 1])
            accumulate = 0
            rho = 0
            for j in range(self.K):
                w_j = w_selected[j].item()
                accumulate += w_j
                if w_j > (accumulate - self.L) / (j + 1):
                    rho = j
                else:
                    break
            tau = (w_selected[:(rho + 1)].sum() - self.L) / (rho + 1)
            w_selected = w_selected - tau
            w_selected[w_selected < 0] = 0
            w_projected[self.support] = w_selected
            return w_projected, self.support



    def a_iht_i(self):
        """
        A-IHT I implemented by numpy
        :param y: numpy.ndarray of shape (M, 1)
        :param A: numpy.ndarray of shape (M, N)
        :param K: int (sparsity constraint)
        :param tol: float (tolerance of the ending criterion)
        :param max_iter_num: int (maximum iteration number)
        :param verbose: boolean (controls intermediate text output)
        :return: w: numpy.ndarray of shape (N, 1)
                supp: list of integer indexes (the support of the w)
        """
        (M, N) = self.A.shape
        if len(self.y.shape) != 2:
            raise ValueError('y should have shape (M, 1)')

        # Initialize transpose of measurement matrix
        A_t = self.A.T

        # Initialization
        w_cur = np.zeros([N, 1])
        y_cur = np.zeros([N, 1])
        # x_cur = np.random.random([N, 1])
        # y_cur = np.random.random([N, 1])

        A_w_cur = np.zeros([M, 1])
        Y_i = []

        # auxiliary variables
        complementary_Yi = np.ones([N, 1])
        i = 1

        while i <= self.max_iter_num:
            w_prev = w_cur
            if i == 1:
                res = self.y
                der = A_t.dot(res)  # compute gradient
            else:
                res = self.y - A_w_cur - tau * A_diff
                der = A_t.dot(res)  # compute gradient
            A_w_prev = A_w_cur
            complementary_Yi[Y_i] = 0
            ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
            complementary_Yi[Y_i] = 1
            S_i = Y_i + np.squeeze(ind_der[0:self.K]).tolist()  # identify active subspace
            ider = der[S_i]
            Pder = self.A[:, S_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
            b = y_cur + mu_bar * der  # gradient descent
            w_cur, X_i = self.l2_projection_numpy(b, self.K, L=self.L)

            A_w_cur = self.A[:, X_i].dot(w_cur[X_i])
            res = self.y - A_w_cur

            if i == 1:
                A_diff = A_w_cur
            else:
                A_diff = A_w_cur - A_w_prev

            temp = A_diff.T.dot(A_diff)
            if temp > 0:
                tau = res.T.dot(A_diff) / temp
            else:
                tau = res.T.dot(A_diff) / 1e-6

            y_cur = w_cur + tau * (w_cur - w_prev)
            Y_i = np.nonzero(y_cur)[0].tolist()

         
            # stop criterion
            if i > 1 and (np.linalg.norm(w_cur - w_prev) < self.tol * np.linalg.norm(w_cur)):
                break
            i = i + 1

        # finished
        w = w_cur
        supp = np.nonzero(w_cur)[0].tolist()  # support of the output solution
        print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                                self.iterative_hardThresholdingObj()))
        return w, supp

    def a_iht_ii(self):
        """
        A-IHT II implemented by numpy
        :param y: numpy.ndarray of shape (M, 1)
        :param A: numpy.ndarray of shape (M, N)
        :param K: int (sparsity constraint)
        :param tol: float (tolerance of the ending criterion)
        :param max_iter_num: int (maximum iteration number)
        :param verbose: boolean (controls intermediate text output)
        :return: w: numpy.ndarray of shape (N, 1)
                supp: list of integer indexes (the support of the w)
        """
        (M, N) = self.A.shape
        if len(self.y.shape) != 2:
            raise ValueError('y should have shape (M, 1)')
        # Initialize transpose of measurement matrix
        A_t = self.A.T

        # Initialize to zero vector
        w_cur = np.zeros([N, 1])
        y_cur = np.zeros([N, 1])
        # w_cur = np.random.random([N, 1])
        # y_cur = np.random.random([N, 1])

        A_w_cur = np.zeros([M, 1])
        Y_i = []

        # auxiliary variables
        complementary_Yi = np.ones([N, 1])
        i = 1

        while i <= self.max_iter_num:
            w_prev = w_cur
            if i == 1:
                res = self.y
                der = A_t.dot(res)  # compute gradient
            else:
                res = self.y - A_w_cur - tau * A_diff
                der = A_t.dot(res)  # compute gradient

            A_w_prev = A_w_cur
            complementary_Yi[Y_i] = 0
            ind_der = np.flip(np.argsort(np.absolute(np.squeeze(der * complementary_Yi))))
            complementary_Yi[Y_i] = 1
            S_i = Y_i + np.squeeze(ind_der[0:self.K]).tolist()  # identify active subspace
            ider = der[S_i]
            Pder = self.A[:, S_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
            b = y_cur + mu_bar * der  # gradient descent
            w_cur, X_i = self.l2_projection_numpy(b, self.K, L=self.L)

            A_w_cur = self.A[:, X_i].dot(w_cur[X_i])
            res = self.y - A_w_cur
            der = A_t.dot(res)  # compute gradient
            ider = der[X_i]
            Pder = self.A[:, X_i].dot(ider)
            mu_bar = ider.T.dot(ider) / Pder.T.dot(Pder) / 2  # step size selection
            w_cur[X_i] = w_cur[X_i] + mu_bar * ider  # debias
            w_cur, _ = self.l2_projection_numpy(w_cur, self.K, is_sparse=True, support=X_i, L=self.L)

            A_w_cur = self.A[:, X_i].dot(w_cur[X_i])
            res = self.y - A_w_cur

            if i == 1:
                A_diff = A_w_cur
            else:
                A_diff = A_w_cur - A_w_prev

            temp = A_diff.T.dot(A_diff)
            if temp > 0:
                tau = res.T.dot(A_diff) / temp
            else:
                tau = res.T.dot(A_diff) / 1e-6

            y_cur = w_cur + tau * (w_cur - w_prev)
            Y_i = np.nonzero(y_cur)[0].tolist()

      
            # stop criterion
            if (i > 1) and (np.linalg.norm(w_cur - w_prev) < self.tol * np.linalg.norm(w_cur)):
                break
            i = i + 1

        # finished
        w = w_cur
        supp = np.nonzero(w_cur)[0].tolist()  # support of the output solution
        print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp),
                                                                                                self.iterative_hardThresholdingObj()))
        return w, supp

    def accelrated_IHT_II(self):
        """
        :param y: torch.tensor of shape (M, 1)
        :param A: torch.tensor of shape (M, N)
        :param K: int (sparsity constraint)
        :param tol: float (tolerance of the ending criterion)
        :param max_iter_num: int (maximum iteration number)
        :param verbose: boolean (controls intermediate text output)
        :return: w: torch.tensor of shape (N, 1)
                supp: list of integer indexes (the support of the w)
        """
        
        (M, N) = self.A.shape
        # Initialize transpose of measurement matrix
        A_t = self.A.T
        # Initialize to zero vector
        current_weights = torch.zeros([N, 1], dtype=self.y.dtype, device=self.y.device)
        current_ys = torch.zeros([N, 1], dtype=self.y.dtype)

        A_current_weights = torch.zeros([M, 1], dtype=self.y.dtype, device=self.y.device)
        Y_i = []

        # auxiliary variables
        Yi_bar = torch.ones([N, 1], dtype=self.y.dtype, device=self.y.device)
        i = 1

        for i in range(1, self.max_iter_num + 1):
            previous_weights = current_weights
            if i == 1:
                res = self.y
                der = A_t.mm(res)
            else:
                res = self.y - A_current_weights - tau * A_diff
                der = A_t.mm(res)

            A_previous_weights = A_current_weights
            Yi_bar[Y_i] = 0
            ind_der = torch.argsort(torch.abs((der * Yi_bar).squeeze()))
            ind_der = ind_der.flip(0)
            Yi_bar[Y_i] = 1
            S_i = Y_i + (ind_der[:self.K]).squeeze().tolist()
            ider = der[S_i]
            Pder = self.A[:, S_i].mm(ider)
            mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
            b = current_ys + mu_bar * der
            current_weights, X_i = self.l2_projection_torch_2(b, self.K, L=self.L)

            A_current_weights = self.A[:, X_i].mm(current_weights[X_i])
            res = self.y - A_current_weights
            der = A_t.mm(res)
            ider = der[X_i]
            Pder = self.A[:, X_i].mm(ider)
            mu_bar = ider.T.mm(ider) / Pder.T.mm(Pder) / 2
            current_weights[X_i] = current_weights[X_i] + mu_bar * ider
            current_weights, _ = self.l2_projection_torch_2(current_weights, self.K, self.is_sparse==True, support=X_i, L=self.L)

            A_current_weights = self.A[:, X_i].mm(current_weights[X_i])
            res = self.y - A_current_weights
            if i == 1:
                A_diff = A_current_weights
            else:
                A_diff = A_current_weights - A_previous_weights
            temp = A_diff.T.mm(A_diff)
            if temp > 0:
                tau = res.T.mm(A_diff) / temp
            else:
                tau = res.T.mm(A_diff) / 1e-6
            current_ys = current_weights + tau * (current_weights - previous_weights)
            Y_i = current_ys.squeeze().nonzero().squeeze().tolist()
            if isinstance(Y_i, int):
                Y_i = [Y_i]


            # stop criterion
            if (i > 1) and (torch.norm(current_weights - previous_weights) < self.tol * torch.norm(current_weights)):
                break

        w = current_weights
        supp = current_weights.squeeze().nonzero().squeeze().tolist()  # support of the output solution
        if isinstance(supp, int):
            supp = [supp]
        obj_value = torch.norm(self.y - self.A.mm(current_weights))
        print('Stopped at iteration {}. {} items are selected. The objective value is: {}'.format(i, len(supp), obj_value))
        return w, supp
