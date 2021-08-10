import numpy as np
import scipy
import scipy.linalg


class DtSystem( object ):
    def __init__ ( self, n_states, n_inputs, n_outputs, Ts, x0 ):
       
        # number of states
        self.n_states = n_states
        
        # number of inputs
        self.n_inputs = n_inputs
        
        # number of outputs
        self.n_outputs = n_outputs
        
        # sampling time
        self.Ts = Ts
        
        # check initial condition
        x0 = np.asmatrix(x0)
        if not x0.shape == ( self.n_states, 1):
            raise Error('wrong shape of initial state vector')
            
        self.x = np.matrix(x0)
    
    def simulate( self, u ):
        
        # set data to proper shape
        u = np.asmatrix(u)
        
        # initialize outputs array. Each column is the output at time k.
        y = np.matrix( np.zeros( (self.n_outputs, u.shape[1] ) ) )
        
        for i in range(u.shape[1]):
            # get measurements of the system
            y[:,i] = self.measure_outputs()
                    
            # compute new state vector
            self._apply_input( u[:,i] )
            
        return np.array( y )
        
    def measure_outputs( self ):
        
        raise NotImplementedError('Use derived classes instead')
    
    def _apply_input( self, u ) :

        raise NotImplementedError('Use derived classes instead')


class DtNLSystem( DtSystem ):
    def __init__ ( self, f, g, n_states, n_inputs, n_outputs, Ts, x0):

        # state equation function
        self.f = f
        
        # outputs equation function
        self.g = g
        
        DtSystem.__init__ ( self, Ts, x0 )


class DtLTISystem( DtSystem ):
    def __init__ ( self, A, B, C, D, Ts, x0 ):

        # set state-space matrices
        self.A = np.matrix(A, copy=True)
        self.B = np.matrix(B, copy=True)
        self.C = np.matrix(C, copy=True)
        self.D = np.matrix(D, copy=True)
        
        # checks
        if not self.A.shape[0] == self.A.shape[1]:
            raise Error('matrix A must be square')
        
        if not self.B.shape[0] == self.A.shape[0]:
            raise Error('matrix B must be have the same number of rows as matrix A')
            
        if not self.C.shape[1] == self.A.shape[0]:
            raise Error('matrix D must be have the same number of columns as matrix A')
            
        if not self.D.shape[0] == self.C.shape[0]:
            raise Error('matrix D must be have the same number of rows as matrix C')

        DtSystem.__init__( self, n_outputs = self.C.shape[0],
                                 n_states = self.A.shape[0],
                                 n_inputs = self.B.shape[1],
                                 Ts = Ts,
                                 x0 = x0  )

    def measure_outputs( self ):

        return self.C * self.x 
    
    def _apply_input( self, u ) :

        self.x = self.A * self.x + self.B * u 


class NoisyDtLTISystem( DtLTISystem ):
    def __init__ ( self, A, B, C, D, Ts, Sw, Sv, x0 ):
        
        # set process and measurement noise covariance matrices
        self.Sw = Sw
        self.Sv = Sv
        
        # call parent __init__
        DtLTISystem.__init__( self, A, B, C, D, Ts, x0 )
        
    def _measurement_noise( self ):

        return np.matrix( np.random.multivariate_normal( np.zeros((self.n_outputs,)), self.Sv, 1 ).reshape( self.n_outputs, 1) )
    
    def _process_noise( self ):

        return np.matrix( np.random.multivariate_normal( np.zeros((self.n_states,)), self.Sw, 1 ).reshape(self.n_states,1) )
               
    def measure_outputs( self ):

        return self.C * self.x + self._measurement_noise()
    
    def _apply_input( self, u ):

        self.x = self.A * self.x + self.B * u + self._process_noise()


class Error( Exception ):

    pass


class ObservabilityError( Exception ):

    pass

