from casadi.tools import *
import osqp
import numpy as np
from scipy import sparse
import sys
sys.path.append('../../')
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time


def dlqr(A, B, Q, R):
    '''
    dlqr solves the discrete time LQR controller
    Inputs:     A, B: system dynamics
    Outputs:    K: optimal feedback matrix
    '''


    P = solve_discrete_are(A, B, Q, R)
    K = - np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))

    return K

class base_real_time_MPC(object):

    def __init__(self, koopman_model, args):


        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['MPC_pred_horizon']
        self.a_dim = args['act_dim']
        self.LQT_gamma = 0.1  # for cost discount

        self.end_weight = args['end_weight']
        self.args = args
        self.model = koopman_model
        self._build_matrices(args)


    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim'] + args['state_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']

        self._set_point_prob = Opti()
        self._set_point_prob.solver('sqpmethod', dict(qpsol='osqp'))
        # self._set_point_prob.solver('ipopt')

        self.ref_us = self._set_point_prob.parameter(self.state_dim)

    def _build_controller(self):
        ## called after the model parameters are imported, building the controller with the model
        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.u_zero = (np.zeros(self.a_dim) - self.shift_u) / self.scale_u
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = np.hstack((np.eye(self.state_dim), np.zeros([self.state_dim, self.args['latent_dim']])))

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_set_point_u_prob()
        self._create_prob(self.args)

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_low'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
            else:
                self.a_bound_low = None
                self.a_bound_high = None

    def _set_LQR_controller(self):

        A_1 = np.sqrt(self.LQT_gamma) * block_diag(self.A, np.eye(self.state_dim))
        B_1 = np.sqrt(self.LQT_gamma) * np.vstack((self.B, np.zeros([self.state_dim, self.a_dim])))
        # projection = np.array([,,,,0.,0.,0.,0.],[,,,,0.,0.,0.,0.],[,,,,0.,0.,0.,0.])
        C_1 = np.hstack((self.C, -np.eye(self.state_dim)))
        self.CQC = CQC = np.dot(C_1.T, np.dot(self.Q, C_1))
        # self.LQR_Q = np.eye(self.latent_dim) ## could also be
        self.K = dlqr(A_1, B_1, CQC, self.R)
        self.P = self._get_trm_cost(A_1, B_1, CQC, self.R, self.K)

    def _create_set_point_u_prob(self):

        self.u_s_var = self._set_point_prob.variable(self.a_dim)
        phi_s = self._set_point_prob.variable(self.latent_dim - self.state_dim)
        x_s = vertcat(self.ref_us, phi_s)

        self._set_point_prob.subject_to(self.C @ x_s == self.C @ (self.A @ x_s + self.B @ self.u_s_var))
        objective = sumsqr(self.u_s_var)

        self._set_point_prob.minimize(objective)

    def _get_set_point_u(self, reference):
        self._set_point_prob.set_value(self.ref_us, reference)
        sol = self._set_point_prob.solve()
        self.u_s = sol.value(self.u_s_var)
        print(sol.value(10*sumsqr((self.C @ self.phi_s - self.ref_us)*self.scale +self.shift)))

        # if self._set_point_prob.status is 'optimal':
        #     self.u_s = self.u_s_var.value
        # else:
        #     print('no suitable set point control input')
        #     self.u_s = np.zeros([self.a_dim])
    def _get_trm_cost(self, A, B, Q, R, K):
        '''
        get_trm_cost returns the matrix P associated with the terminal cost
        Outputs:    P: the matrix associated with the terminal cost in the objective
        '''

        A_lyap = (A + np.dot(B, K)).T
        Q_lyap = Q + np.dot(K.T, np.dot(R, K))

        P = solve_discrete_lyapunov(A_lyap, Q_lyap)

        return P


    def _create_prob(self, args):
        N = self.pred_horizon
        x0 = np.zeros(self.latent_dim)
        nx = self.latent_dim
        ny = self.state_dim
        nu = self.a_dim
        CQC = sparse.csc_matrix(np.dot(self.C.T, np.dot(self.Q, self.C)))
        QN = self.P[:self.latent_dim, :self.latent_dim]
        self.qN = qN = self.P[:self.latent_dim, self.latent_dim:]
        if args['apply_action_constraints']:
            umin = self.a_bound_low
            umax = self.a_bound_high
        else:
            umin = -np.inf * np.ones([nu])
            umax = np.inf * np.ones([nu])

        if args['apply_state_constraints']:
            xmin = self.s_bound_low
            xmax = self.s_bound_high
        else:
            xmin = -np.inf * np.ones([ny])
            xmax = np.inf * np.ones([ny])

        P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), QN,
                               sparse.kron(sparse.eye(N), self.R)], format='csc')

        self.q = q = np.hstack([np.kron(np.ones(N), -self.C.T.dot(self.Q.dot(self.reference))),
                       -qN.dot(self.reference),
                       np.kron(np.ones(N), -self.R.dot(self.u_zero))])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), self.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self.B)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])
        ueq = leq

        # - input and state constraints

        Aineq = sparse.block_diag([sparse.kron(sparse.eye(N + 1, k=-1), self.C), sparse.eye(N * nu)])
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l_cons = l = np.hstack([leq, lineq])
        self.u_cons = u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, l, u, warm_start=True)


    def choose_action(self, x_0, reference, *args):
        pass

    def reset(self):
        pass

    def update_reference(self, reference):
        self.reference = reference
        self._get_set_point_u(self.reference)


    def check_controllability(self):

        gamma = [self.model.B_result.T]
        A = self.model.A_result.T
        for d in range(self.latent_dim-1):

            gamma.append(np.matmul(A, gamma[d]))

        gamma = np.concatenate(np.array(gamma), axis=1)
        rank = np.linalg.matrix_rank(gamma)
        print('rank of controllability matrix is ' + str(rank) +'/'+ str(self.latent_dim))

    def restore(self, path):
        success = self.model.restore(path)
        self._build_controller()
        return success


class real_time_MPC(base_real_time_MPC):

    def __init__(self, model, args):

        super(real_time_MPC, self).__init__(model, args)

    def choose_action(self, x_0, reference, *args):

        if hasattr(self, 'last_state'):
            if len(self.last_state)<1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])

        self.l_cons[:self.latent_dim] = phi_0
        self.u_cons[:self.latent_dim] = phi_0
        res = self.prob.solve()
        self.prob.update(l=self.l_cons, u=self.u_cons)

        u = res.x[-self.pred_horizon*self.a_dim:-(self.pred_horizon-1)*self.a_dim]
        u = u * self.scale_u + self.shift_u
        return u

    def reset(self):
        self.inital_step = True
        if hasattr(self, 'last_state'):
            self.last_state = []

class real_time_MPC_dynamic(real_time_MPC):

    def __init__(self, model, args):

        super(real_time_MPC_dynamic, self).__init__(model, args)

    def choose_action(self, x_0, reference, *args):

        if hasattr(self, 'last_state'):
            if len(self.last_state)<1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])

        if self.inital_step:
            self.mu_hat = phi_0
            self.inital_step = False
        reference = (reference - self.shift) / self.scale
        self.l_cons[:self.latent_dim] = self.mu_hat
        self.u_cons[:self.latent_dim] = self.mu_hat
        self.q[:(self.pred_horizon + 1) * self.latent_dim] = \
            np.hstack([np.reshape(-self.C.T.dot(self.Q.dot(reference[:self.pred_horizon].T)).T,
                                  [self.latent_dim * self.pred_horizon]),
                       -self.qN.dot(reference[-1])])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()


        u = res.x[-self.pred_horizon*self.a_dim:-(self.pred_horizon-1)*self.a_dim]

        error = np.concatenate([phi_0-self.mu_hat, np.zeros(self.state_dim)])
        u = u + np.dot(self.K, error)
        # self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(u)
        u = u * self.scale_u + self.shift_u
        return u


class robust_MPC(base_real_time_MPC):

    def __init__(self, model, args):
        super(robust_MPC, self).__init__(model, args)


    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']
        # self.opti.solver('ipopt')
        self._set_point_prob = Opti()
        self._set_point_prob.solver('sqpmethod', dict(qpsol='osqp'))
        # self._set_point_prob.solver('ipopt')

        self.ref_us = self._set_point_prob.parameter(self.state_dim)
    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.u_zero = (np.zeros(self.a_dim) - self.shift_u) / self.scale_u
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_set_point_u_prob()
        self._get_set_point_u(self.reference)
        self._create_prob(self.args)

    def _create_set_point_u_prob(self):

        self.u_s_var = self._set_point_prob.variable(self.a_dim)
        self.phi_s = phi_s = self._set_point_prob.variable(self.latent_dim)

        self._set_point_prob.subject_to(phi_s == self.A @ phi_s + self.B @ self.u_s_var)
        self._set_point_prob.subject_to(self.u_s_var >= self.a_bound_low)
        self._set_point_prob.subject_to(self.u_s_var <= self.a_bound_high)
        # self._set_point_prob.subject_to()

        objective = 0.1 *sumsqr(self.u_s_var-self.u_zero) + 10*sumsqr((self.C @ phi_s - self.ref_us)*self.scale +self.shift)
        self._set_point_prob.minimize(objective)
        print()


    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, _] = self.model.encode([x_0])
        if self.inital_step:
            self.mu_hat = mean[0]
            self.inital_step = False

        reference = (reference - self.shift) / self.scale
        self.l_cons[:self.latent_dim] = self.mu_hat
        self.u_cons[:self.latent_dim] = self.mu_hat
        self.q[:(self.pred_horizon+1) * self.latent_dim] = \
            np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))), -self.qN.dot(reference)])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()
        c_t = res.x[-self.pred_horizon * self.a_dim:-(self.pred_horizon - 1) * self.a_dim]
        error = np.concatenate([mean[0]-self.mu_hat, np.zeros_like(reference)])
        u = c_t + np.dot(self.K, error)
        # # u = self.u_s
        # error = np.concatenate([mean[0], self.reference])
        # u = np.dot(self.K, error)
        u = u * self.scale_u + self.shift_u


        # print(str(time_for_solving))
        self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(c_t)
        t6 = time.time()
        overall_time = t6 - t1
        print(str(overall_time))
        return u

    def reset(self):
        self.inital_step = True


class robust_MPC_dynamic_tracking(robust_MPC):

    def __init__(self, model, args):
        super(robust_MPC_dynamic_tracking, self).__init__(model, args)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']
        # self.opti.solver('ipopt')
        # self._set_point_prob.solver('ipopt')


    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.u_zero = (np.zeros(self.a_dim) - self.shift_u) / self.scale_u
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_prob(self.args)

    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, _] = self.model.encode([x_0])

        if self.inital_step:
            self.mu_hat = mean[0]
            self.inital_step = False
        reference = (reference - self.shift) / self.scale
        self.l_cons[:self.latent_dim] = -self.mu_hat
        self.u_cons[:self.latent_dim] = -self.mu_hat
        self.q[:(self.pred_horizon+1) * self.latent_dim] = \
            np.hstack([np.reshape(-self.C.T.dot(self.Q.dot(reference[:self.pred_horizon].T)).T, [self.latent_dim*self.pred_horizon]),
                       -self.qN.dot(reference[self.pred_horizon])])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()
        c_t = res.x[-self.pred_horizon * self.a_dim:-(self.pred_horizon - 1) * self.a_dim]
        error = np.concatenate([mean[0]-self.mu_hat, np.zeros(self.state_dim)])
        u = c_t + np.dot(self.K, error)
        # # u = self.u_s
        # error = np.concatenate([mean[0], reference[0]])
        # u = np.dot(self.K, error)



        # print(str(time_for_solving))
        self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(u)
        u = u * self.scale_u + self.shift_u
        t6 = time.time()
        overall_time = t6 - t1
        print(str(overall_time))
        return u

class robust_MPC_dynamic_tracking_partial_obs(robust_MPC_dynamic_tracking):

    def __init__(self, model, args):
        self.start_dim = args['reconstruct_dims'][0]
        self.end_dim = args['reconstruct_dims'][-1] + 1
        self.obs_dim = self.end_dim-self.start_dim
        super(robust_MPC_dynamic_tracking_partial_obs, self).__init__(model, args)

    def _set_LQR_controller(self):

        A_1 = np.sqrt(self.LQT_gamma) * block_diag(self.A, np.eye(self.obs_dim))
        B_1 = np.sqrt(self.LQT_gamma) * np.vstack((self.B, np.zeros([self.obs_dim, self.a_dim])))
        # projection = np.array([,,,,0.,0.,0.,0.],[,,,,0.,0.,0.,0.],[,,,,0.,0.,0.,0.])
        C_1 = np.hstack((self.C, -np.eye(self.obs_dim)))
        self.CQC = CQC = np.dot(C_1.T, np.dot(self.Q, C_1))
        # self.LQR_Q = np.eye(self.latent_dim) ## could also be
        self.K = dlqr(A_1, B_1, CQC, self.R)
        self.P = self._get_trm_cost(A_1, B_1, CQC, self.R, self.K)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q'][self.start_dim:self.end_dim, self.start_dim:self.end_dim]
        self.R = self.args['R']

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.u_zero = (np.zeros(self.a_dim) - self.shift_u) / self.scale_u
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.reference = self.reference[self.start_dim:self.end_dim]
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_prob(self.args)

    def _create_prob(self, args):
        N = self.pred_horizon
        x0 = np.zeros(self.latent_dim)
        nx = self.latent_dim
        ny = len(args['reconstruct_dims'])
        nu = self.a_dim
        CQC = sparse.csc_matrix(np.dot(self.C.T, np.dot(self.Q, self.C)))
        QN = self.P[:self.latent_dim, :self.latent_dim]
        self.qN = qN = self.P[:self.latent_dim, self.latent_dim:]
        if args['apply_action_constraints']:
            umin = self.a_bound_low
            umax = self.a_bound_high
        else:
            umin = -np.inf * np.ones([nu])
            umax = np.inf * np.ones([nu])

        if args['apply_state_constraints']:
            xmin = self.s_bound_low
            xmax = self.s_bound_high
        else:
            xmin = -np.inf * np.ones([ny])
            xmax = np.inf * np.ones([ny])

        P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), QN,
                               sparse.kron(sparse.eye(N), self.R)], format='csc')

        self.q = q = np.hstack([np.kron(np.ones(N), -self.C.T.dot(self.Q.dot(self.reference))),
                       -qN.dot(self.reference),
                       np.kron(np.ones(N), -self.R.dot(self.u_zero))])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), self.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self.B)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])
        ueq = leq

        # - input and state constraints

        Aineq = sparse.block_diag([sparse.kron(sparse.eye(N + 1, k=-1), self.C), sparse.eye(N * nu)])
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l_cons = l = np.hstack([leq, lineq])
        self.u_cons = u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, l, u, warm_start=True)

    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, _] = self.model.encode([x_0])

        if self.inital_step:
            self.mu_hat = mean[0]
            self.inital_step = False
        reference = (reference - self.shift) / self.scale
        reference = reference[:, self.start_dim:self.end_dim]
        self.l_cons[:self.latent_dim] = -self.mu_hat
        self.u_cons[:self.latent_dim] = -self.mu_hat
        self.q[:(self.pred_horizon+1) * self.latent_dim] = \
            np.hstack([np.reshape(-self.C.T.dot(self.Q.dot(reference[:self.pred_horizon].T)).T, [self.latent_dim*self.pred_horizon]),
                       -self.qN.dot(reference[self.pred_horizon])])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()
        c_t = res.x[-self.pred_horizon * self.a_dim:-(self.pred_horizon - 1) * self.a_dim]
        error = np.concatenate([mean[0]-self.mu_hat, np.zeros(self.end_dim - self.start_dim)])
        u = c_t + np.dot(self.K, error)
        # # u = self.u_s
        # error = np.concatenate([mean[0], reference[0]])
        # u = np.dot(self.K, error)



        # print(str(time_for_solving))
        self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(u)
        u = u * self.scale_u + self.shift_u
        t6 = time.time()
        overall_time = t6 - t1
        print(str(overall_time))
        return u

class robust_MPC_dynamic_tracking_V2(robust_MPC):

    def __init__(self, model, args):
        super(robust_MPC_dynamic_tracking_V2, self).__init__(model, args)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']
        # self.opti.solver('ipopt')
        # self._set_point_prob.solver('ipopt')

    def _create_prob(self, args):
        N = self.pred_horizon
        x0 = np.zeros(self.latent_dim)
        nx = self.latent_dim
        ny = self.state_dim
        nu = self.a_dim
        CQC = sparse.csc_matrix(np.dot(self.C.T, np.dot(self.Q, self.C)))
        QN = self.P
        self.qN = qN = self.P
        if args['apply_action_constraints']:
            umin = self.a_bound_low
            umax = self.a_bound_high
        else:
            umin = -np.inf * np.ones([nu])
            umax = np.inf * np.ones([nu])

        if args['apply_state_constraints']:
            xmin = self.s_bound_low
            xmax = self.s_bound_high
        else:
            xmin = -np.inf * np.ones([ny])
            xmax = np.inf * np.ones([ny])

        P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), QN,
                               sparse.kron(sparse.eye(N), self.R)], format='csc')

        self.q = q = np.hstack([np.kron(np.ones(N), -self.C.T.dot(self.Q.dot(self.reference))),
                       -qN.dot(np.zeros([self.latent_dim])),
                       np.kron(np.ones(N), -self.R.dot(self.u_zero))])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), self.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self.B)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])
        ueq = leq

        # - input and state constraints

        Aineq = sparse.block_diag([sparse.kron(sparse.eye(N + 1, k=-1), self.C), sparse.eye(N * nu)])
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l_cons = l = np.hstack([leq, lineq])
        self.u_cons = u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, l, u, warm_start=True)

    def _set_LQR_controller(self):


        self.CQC = CQC = np.dot(self.C.T, np.dot(self.Q, self.C))
        # self.LQR_Q = np.eye(self.latent_dim) ## could also be
        self.K = dlqr(self.A, self.B, CQC, self.R)
        self.P = self._get_trm_cost(self.A, self.B, CQC, self.R, self.K)

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.u_zero = (np.zeros(self.a_dim) - self.shift_u) / self.scale_u
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_prob(self.args)

    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, _] = self.model.encode([x_0])
        [mean_ref, _] = self.model.encode(reference)
        if self.inital_step:
            self.mu_hat = mean[0]
            self.inital_step = False
        reference = (reference - self.shift) / self.scale
        self.l_cons[:self.latent_dim] = -self.mu_hat
        self.u_cons[:self.latent_dim] = -self.mu_hat
        self.q[:(self.pred_horizon+1) * self.latent_dim] = \
            np.hstack([np.reshape(-self.C.T.dot(self.Q.dot(reference[:self.pred_horizon].T)).T, [self.latent_dim*self.pred_horizon]),
                       -self.qN.dot(mean_ref[self.pred_horizon])])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()
        c_t = res.x[-self.pred_horizon * self.a_dim:-(self.pred_horizon - 1) * self.a_dim]
        error = mean[0]-self.mu_hat
        u = c_t + np.dot(self.K, error)
        # # u = self.u_s
        # error = np.concatenate([mean[0], self.reference])
        # u = np.dot(self.K, error)
        u = u * self.scale_u + self.shift_u


        # print(str(time_for_solving))
        # self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(c_t)
        t6 = time.time()
        overall_time = t6 - t1
        print(str(overall_time))
        return u

class robust_MPC_dynamic_tracking_with_jerk_penalty(robust_MPC_dynamic_tracking):

    def __init__(self, model, args):
        super(robust_MPC_dynamic_tracking_with_jerk_penalty, self).__init__(model, args)

    def _create_prob(self, args):

        N = self.pred_horizon
        x0 = np.zeros(self.latent_dim)
        nx = self.latent_dim
        ny = self.state_dim
        nu = self.a_dim
        CQC = sparse.csc_matrix(np.dot(self.C.T, np.dot(self.Q, self.C)))
        QN = self.P[:self.latent_dim, :self.latent_dim]
        self.qN = qN = self.P[:self.latent_dim, self.latent_dim:]
        if args['apply_action_constraints']:
            umin = self.a_bound_low
            umax = self.a_bound_high
        else:
            umin = -np.inf * np.ones([nu])
            umax = np.inf * np.ones([nu])

        if args['apply_state_constraints']:
            xmin = self.s_bound_low
            xmax = self.s_bound_high
        else:
            xmin = -np.inf * np.ones([ny])
            xmax = np.inf * np.ones([ny])
        component1 = sparse.hstack([np.zeros([N*nu,(N-1)*nu]),sparse.vstack([np.zeros([(N-1)*nu,nu]), -self.R])])
        component2 = sparse.hstack([sparse.vstack([-self.R, np.zeros([(N-1)*nu,nu])]), np.zeros([N*nu,(N-1)*nu])])
        P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), QN,
                               sparse.kron(sparse.eye(N), 3*self.R) +
                               sparse.kron(sparse.eye(N, N, -1), -self.R) +
                               sparse.kron(sparse.eye(N, N, 1), -self.R) + component1 + component2], format='csc')
        component3 = np.hstack([-self.R.dot(self.u_zero), np.zeros([(N-1)*nu])])
        self.q = q = np.hstack([np.kron(np.ones(N), -self.C.T.dot(self.Q.dot(self.reference))),
                                -qN.dot(self.reference),
                                np.kron(np.ones(N), -self.R.dot(self.u_zero))+component3])

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(N + 1, k=-1), self.A)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self.B)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N * nx)])
        ueq = leq

        # - input and state constraints

        Aineq = sparse.block_diag([sparse.kron(sparse.eye(N + 1, k=-1), self.C), sparse.eye(N * nu)])
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l_cons = l = np.hstack([leq, lineq])
        self.u_cons = u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, l, u, warm_start=True)

    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, _] = self.model.encode([x_0])

        if self.inital_step:
            self.mu_hat = mean[0]
            self.inital_step = False

        reference = (reference - self.shift) / self.scale
        self.l_cons[:self.latent_dim] = -self.mu_hat
        self.u_cons[:self.latent_dim] = -self.mu_hat

        component3 = np.hstack([-self.R.dot(self.u_last_step), np.zeros([(self.pred_horizon-1)*self.a_dim])])
        self.q = np.hstack([np.reshape(-self.C.T.dot(self.Q.dot(reference[:self.pred_horizon].T)).T,
                                       [self.latent_dim * self.pred_horizon]), -self.qN.dot(reference[-1]),
                            np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero)) + component3])
        # self.q = np.hstack([np.kron(np.ones(self.pred_horizon), -self.C.T.dot(self.Q.dot(reference))),
        #                -self.qN.dot(reference),
        #                np.kron(np.ones(self.pred_horizon), -self.R.dot(self.u_zero))])
        self.prob.update(q=self.q, l=self.l_cons, u=self.u_cons)
        res = self.prob.solve()
        c_t = res.x[-self.pred_horizon * self.a_dim:-(self.pred_horizon - 1) * self.a_dim]
        error = np.concatenate([mean[0] - self.mu_hat, np.zeros(self.state_dim)])
        u = c_t + np.dot(self.K, error)
        # # u = self.u_s
        # error = np.concatenate([mean[0], self.reference])
        # u = np.dot(self.K, error)
        self.u_last_step = u
        u = u * self.scale_u + self.shift_u

        # print(str(time_for_solving))
        # self.mu_hat = self.A.dot(self.mu_hat) + self.B.dot(c_t)
        t6 = time.time()
        overall_time = t6 - t1
        print(str(overall_time))

        return u

    def reset(self):
        self.inital_step = True
        self.u_last_step = self.u_zero


