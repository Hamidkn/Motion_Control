import numpy as np
from systems import ObservabilityError

class SimEnv( object ):
    def __init__ ( self, system, controller=None, observer=None ):
        
        # the system we want to simulate
        self.system = system
        
        # the controller ( state feedback )
        self.controller = controller

        # the state observer
        self.observer = observer
        
    def simulate( self, Tsim ):
        
        # run for sufficient steps
        n_steps = int( Tsim / self.system.Ts ) + 1
        
        # Preallocate matrices
        # This matrix is for the state 
        xhat = np.zeros( (self.system.n_states, n_steps) )
        
        # control input
        u = np.zeros( (self.system.n_inputs, n_steps) )

        # measure setpoint
        setpoint = np.zeros( (self.system.n_inputs, n_steps))
        sp = 0
        
        # measurements
        y = np.zeros( (self.system.n_outputs, n_steps) )
        
        
        if self.observer:
            # if we have an observer estimate system's state
            xhat[:,0] = self.observer.get_state_estimate( 
                self.system.measure_outputs().ravel(), u[:,0]  ).ravel()
        else:
            # try to compute measurements, but only if the system is observable.
            try:
                xhat[:,0] = (np.linalg.inv(self.system.C) * self.system.measure_outputs()).ravel()
            except np.linalg.LinAlgError:
                raise ObservabilityError( "System is not observable. Cannot compute system's state." )
                
        # run simulation
        for k in range( n_steps-1 ):
                        
            # compute control move based on the state at this time. 
            u[:,k] = self.controller.compute_control_input( xhat[:,k].reshape(self.system.n_states,1) )
            
            # apply input 
            self.system._apply_input( u[:,k].reshape(self.system.n_inputs, 1) )
            
            # get measuremts
            y[:,k+1] = self.system.measure_outputs().ravel()

            if self.observer:
                xhat[:,k+1] = self.observer.get_state_estimate( y[:,k+1], u[:,k] ).ravel()
            else:
                xhat[:,k+1] = (np.linalg.inv(self.system.C) * self.system.measure_outputs() ).ravel()

            if( k > 9):
                sp = u[:,k] - sp / k
            setpoint[:, k] = sp
                
        return SimulationResults(xhat, u, y, self.system.Ts, setpoint)


class SimulationResults():
    def __init__ ( self, x, u, y, Ts, setpoint ):

        self.x = x
        self.u = u
        self.y = y
        self.t = np.arange(x.shape[1]) * Ts
        self.sp = setpoint
