import os
import inspect
import numpy as np
from .test_function import TestFunction

# auxiliary function for the standard form
def g(x, beta, eta, x0):
    g1 = beta * np.mean(np.square(x[:, 1:] - x0), axis=1) * (1 - np.exp(-eta * x[:, 0]))
    g2 = beta * np.mean(np.square(x[:, 1:] - x0), axis=1) * (1 - np.exp(-eta * (1 - x[:, 0])))
    
    return g1, g2

# auxiliary function for the `hard` form
def g_H(x, beta, eta, x0):
    g1 = beta * (1-np.mean(np.square(1-np.abs(x[:, 1:] - x0)), axis=1)) * (1 - np.exp(-eta * x[:, 0]))
    g2 = beta * (1-np.mean(np.square(1-np.abs(x[:, 1:] - x0)), axis=1)) * (1 - np.exp(-eta * (1 - x[:, 0])))
    
    return g1, g2


class RLSFBase(TestFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.eps = 0.01                             # small value to avoid division by zero
        self.alpha = self._instance_params['alpha'] # controls the sharpness of the PF bend
        self.gamma = self._instance_params['gamma'] # controls the overall scaling of the function
        self.beta = self._instance_params['beta']     # controls the size of the feasible region
        self.eta = self._instance_params['eta']   # controls the shape of the feasible region
        self.x0 = self._instance_params['x0']       # value of x2 to obtain PF
        self.g = self._instance_params['g_func']    # auxiliary function


class rRLSF(RLSFBase):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.25, 'beta':6, 'gamma':0.5, 'eta':5, 'x0':0.5, 'g_func':g}
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        # compute shift for the "legs" of the PF, so that the ends sit directly on the axes
        self.leg_shift = abs(self.gamma * (1 - 1 / (self.eps ** self.alpha)))
        
    def _evaluate(self, x, out, *args, **kwargs):        
        # growth terms to expand the feasible region        
        g1, g2 = self.g(x, self.eta, self.beta, self.x0)
        
        f1 = self.gamma * (1 - 1 / (x[:, 0] + self.eps) ** self.alpha) + g1 + self.leg_shift
        f2 = self.gamma * (1 - 1 / (1 - x[:, 0] + self.eps) ** self.alpha) + g2 + self.leg_shift

        out["F"] = np.column_stack((f1, f2))


class RLSF(RLSFBase):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.4, 'beta':6, 'gamma':0.5, 'eta':50, 'x0':0.5, 'g_func':g}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # compute shift for the "legs" of the PF, so that the asymptotes sit directly on the axes
        self.leg_shift = self.gamma / (1 + self.eps) ** self.alpha

    def _evaluate(self, x, out, *args, **kwargs):
        # growth terms to expand the feasible region
        g1, g2 = self.g(x, self.beta, self.eta, self.x0)
        
        # objective functions for non-inverted "L" shape with shifts
        f1 = self.gamma / (x[:, 0] + self.eps) ** self.alpha + g1 - self.leg_shift
        f2 = self.gamma / (1 - x[:, 0] + self.eps) ** self.alpha + g2 - self.leg_shift

        out["F"] = np.column_stack((f1, f2))


class rRLSF_H(rRLSF):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.25, 'beta':2, 'gamma':0.5, 'eta':5, 'x0':0.5, 'g_func':g_H}
    
    
class RLSF_H(RLSF):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.4, 'beta':2, 'gamma':0.5, 'eta':50, 'x0':0.5, 'g_func':g_H}
        




