import numpy as np
from scipy.linalg import block_diag, solve_discrete_are, solve_continuous_are
import time


class Controller( object ):

    def compute_control_input( self, x ):
        self.value = 1
        return np.matrix([[self.value]])
        # raise NotImplemented( 'Call derived classes instead.')


class LQController( Controller ):
    def __init__ ( self, system, Q, R ):

        # make two local variables
        Ql = np.matrix( Q )
        Rl = np.matrix( R )
        
        # solve Algebraic Riccati Equation
        P = solve_discrete_are( system.A, system.B, Ql, Rl )
        # P = np.matrix(solve_continuous_are( system.A, system.B, Ql, Rl ))
        
        # create static state feedback matrix
        self.K  = (Rl + system.B.T * P * system.B).I * (system.B.T * P * system.A)
        # self.K  = np.matrix((Rl).I * (system.B.T * P ))
        
    def compute_control_input( self, x ):
        
        output = -np.dot( self.K, x)
        return output

class MPCController( Controller ):
    def __init__(self, system, horizon=5, ref_path=10):
        super().__init__()
        self.horizon = horizon
        self.num_red_path = ref_path

    def compute_control_input(self, x):
        pass

class PIDController( Controller ):
    def __init__(self, system, P=1.2, I=1.0, D=0.001, current_time=None, sample_time=0.01):
        
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.lasterror = 0
        self.P = 0
        self.I = 0
        self.D = 0
        
        # Windup Guard
        '''
        Integral windup, also known as integrator windup or reset windup, refers to the situation in a PID feedback 
        controller where a large change in setpoint occurs (say a positive change) and the integral term accumulates a 
        significant error during the rise (windup), thus overshooting and continuing to increase as this accumulated error 
        is unwound (offset by errors in the other direction).
        '''
        self.windup = 1.0
        
        if current_time is not None:
            self.current_time = current_time
        else:
            self.current_time = time.time()

        self.sample_time = sample_time
        self.last_time = self.current_time
        self.output = 0
        self.setpoint = 1.0

    def compute_control_input(self, x):
        # feedback = self.output
        feedback = 0
        if( self.setpoint > 0 ):
                feedback += self.output
        self.setpoint = 1
        error = self.setpoint - feedback

        self.current_time = time.time()

        dt = self.current_time - self.last_time
        derror = error - self.lasterror

        if(dt >= self.sample_time):
            self.P = self.Kp * error
            self.I += error * dt

            # if(self.I < -self.windup):
            #     self.I = -self.windup
            
            # elif(self.I > self.windup):
            #     self.I = self.windup
            
            if(dt > 0):
                self.D = derror / dt
            
            self.last_time = self.current_time
            self.lasterror = error

            self.output = self.P + self.Ki * self.I + self.Kd * self.D

        return self.output