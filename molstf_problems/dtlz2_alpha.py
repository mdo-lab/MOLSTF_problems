import numpy as np
from .test_function import TestFunction
from pymoo.problems.many.dtlz import DTLZ2

# auxiliary function for the standard form
def g(X_M, beta, x0):
    return beta * (np.mean(np.square(X_M - x0), axis=1))

# auxiliary function for the `hard` form
def g_H(X_M, beta, x0):
    return beta * (1-np.mean(np.square(1-np.abs(X_M - x0)), axis=1))

class DTLZ2mu(DTLZ2):
    def g2(self, X_M):
        return g(X_M, 1, 0.5)

class DTLZ2mu_H(DTLZ2):
    def g2(self, X_M):
        return g_H(X_M, 1, 0.5)

class PymooDTLZ2(TestFunction):
    _default_params = {'n_var':2, 'n_obj':2, 'x0': 0.5}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dltz2 = DTLZ2(n_var=self.n_var, n_obj=self.n_obj)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.dltz2.evaluate(x)

class PymooDTLZ2mu(PymooDTLZ2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dltz2 = DTLZ2mu(n_var=self.n_var, n_obj=self.n_obj)

class PymooDTLZ2mu_H(PymooDTLZ2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dltz2 = DTLZ2mu_H(n_var=self.n_var, n_obj=self.n_obj)

class DTLZ2AlphaBase(TestFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = self._instance_params['alpha']     # controls the sharpness of the PF bend
        self.beta = self._instance_params['beta']       # controls the range of the feasible region
        self.gamma = self._instance_params['gamma']     # controls the scaling of the entire function
        self.x0 = self._instance_params['x0']           # value of x[:,1:] to obtain PF
        self.g = self._instance_params['g_func']        # auxiliary function
    
    def obj_func(self, X_, g):
        f = []
        for i in range(0, self.n_obj):
            _f = (1 + g) #if not self.reflect else (1 - g)
            _f *= self.gamma
            _f *= np.prod(np.cos((X_[:, :X_.shape[1] - i]) * np.pi / 2.0), axis=1)**self.alpha
            if i > 0:
                _f *= np.sin((X_[:, X_.shape[1] - i]) * np.pi / 2.0)**self.alpha

            f.append(_f)

        f = np.column_stack(f) #if not self.reflect else np.column_stack([-np.array(col)+self.gamma for col in f[::-1]])
        return f
    
    
class DTLZ2Alpha(DTLZ2AlphaBase):      
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.3, 'beta':8, 'gamma':1, 'x0':0.5, 'g_func':g}
    
    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g(X_M, self.beta, self.x0)
        out["F"] = self.obj_func(X_, g) - (self.gamma - 1)

        
class rDTLZ2Alpha(DTLZ2AlphaBase):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.3, 'beta':3, 'gamma':3, 'x0':0.5, 'g_func':g}
    
    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = -self.g(X_M, self.beta, self.x0)
        F = self.obj_func(X_, g)
        out["F"] = -F[:, ::-1] + self.gamma
                
class DTLZ2Alpha_H(DTLZ2Alpha):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.3, 'beta':8, 'gamma':0.3, 'x0':0.5, 'g_func':g_H}
        
    
class rDTLZ2Alpha_H(rDTLZ2Alpha):
    _default_params = {'n_var':2, 'n_obj':2, 'alpha':0.3, 'beta':1, 'gamma':3, 'x0':0.5, 'g_func':g_H}
