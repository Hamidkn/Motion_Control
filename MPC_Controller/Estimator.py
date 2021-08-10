
import numpy as np
from systems import Error

class KalmanStateObserver( object ):
    def __init__ ( self, system, x0=None ):
        # set attributes
        self._A = system.A
        self._B = system.B
        self._C = system.C
        self._D = system.D
        self._Sw = system.Sw
        self._Sv = system.Sv
        self._n_states = system.n_states
        self._n_inputs = system.n_inputs
        self._n_outputs = system.n_outputs
        
        if x0 is None:
            self._xhat = system.x + np.matrix( np.random.multivariate_normal( np.zeros((system.n_states,)), system.Sw, 1 ).reshape(system.n_states,1) )
        else:
            x0_ = np.matrix( x0 )
            if not x0_.shape == ( self._n_states, 1 ):
                raise Error('wrong shape of initial state vector')
            self._xhat = x0_
        
        self.P = self._Sw.copy()
        
    def get_state_estimate( self, y, u ):
        """Get estimate of the systems state."""
        y = np.asmatrix(y).reshape(self._n_outputs, 1)
        
        self._xhat = self._A * self._xhat + self._B * u
        
        inn = y - self._C*self._xhat
        
        s = self._C*self.P*self._C.T + self._Sv
        
        K = self._A*self.P*self._C.T * np.linalg.inv(s)
        
        self._xhat += K*inn
        
        self.P = self._A*self.P*self._A.T -  \
                 self._A*self.P*self._C.T * np.linalg.inv(s)*\
                 self._C*self.P*self._A.T + self._Sw
        
        return self._xhat


class ExtendedKalmanFilter( object ):
    def __init__(self, system, x0= None):
        # set attributes
        self._A = system.A
        self._B = system.B
        self._C = system.C
        self._D = system.D
        self._Sw = system.Sw
        self._Sv = system.Sv
        self._n_states = system.n_states
        self._n_inputs = system.n_inputs
        self._n_outputs = system.n_outputs
    
    def get_state_estimate(self, y, u):
        pass