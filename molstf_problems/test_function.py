from abc import ABC, abstractmethod
from copy import deepcopy
from pymoo.core.problem import Problem
from molstf_problems.pf_tools import get_test_fn_pf

# function to compare parameter dictionaries, accounting for function objects
def compare_params(params1, params2):
    for k in params1:
        v1, v2 = params1.get(k), params2.get(k)
        if callable(v1) and callable(v2):
            if v1 is not v2:
                return False
        else:
            if v1 != v2:
                return False
    return True


class TestFunction(Problem, ABC):
    @property
    @abstractmethod
    def _default_params(self):
        """Subclasses must define their own _default_params dictionary"""
        pass
    
    def __init__(self, *args, **kwargs):
        # Ensure _default_params is defined
        if not self._default_params:
            raise NotImplementedError(f"{self.__class__.__name__} cannot be instantiated, subclasses must define _default_params dictionary.")
        
        # copy class-level default parameters to instance level
        self._default_params = deepcopy(self._default_params)
        
        # update with default bounds if not provided
        self._default_params.update({k:v for k,v in [('xl',0),('xu',1)] if self._default_params.get(k) is None})
        
        # set instance params and remove from kwargs to avoid conflicts with superclass
        self._instance_params = {
            k: (kwargs.pop(k) if k in kwargs and kwargs[k] is not None else self._default_params[k])
            for k in self._default_params
        }
        
        # check if default parameters are used
        self.is_default_instance = True if compare_params(self._instance_params, self._default_params) else False
                
        # initialise Pymoo Problem superclass
        super().__init__(n_var=self._instance_params['n_var'], 
                         n_obj=self._instance_params['n_obj'], 
                         xl=self._instance_params['xl'], 
                         xu=self._instance_params['xu'], 
                         **kwargs)
    
    def _calc_pareto_front(self, n_pareto_points=500, return_pareto_set=False, initial_multiplier=10, fill_missing_points=True):        
        F, X = get_test_fn_pf(self, 
                              n_pareto_points=n_pareto_points, 
                              initial_multiplier=initial_multiplier, 
                              x0=self._instance_params['x0'], 
                              fill_missing_points=fill_missing_points)
        if return_pareto_set:
            return F, X
        return F
