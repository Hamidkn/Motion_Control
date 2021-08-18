import numpy as np
from scipy.linalg import block_diag, solve_discrete_are, solve_continuous_are
import time
import copy
import torch
from gym import spaces


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

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
    def __init__(self, system, horizon=15):
        super().__init__()
        self.horizon = horizon
        self.reference = [11, 11, 0]
        self.dt = 0.2
        self.gamma = 0.99
        self.epsilon = 0.2
        self.alpha = 0.01
        self.init_var = 10
        self.sol_dim = 15
        self.popsize = 40000
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.upper_bound = np.array(self.action_space.high)
        self.lower_bound = np.array(self.action_space.low)
        self.iter = 20
        self.num_elites = 50
        # self.cost_function = None
        self.size = [self.popsize, self.sol_dim]
        self.action_dim = 1
        self.particle = 1

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((-1 + 1) / 2, [self.horizon])
        self.init_var = np.tile(np.square(-1 - 1) / 16, [self.horizon])

    def compute_control_input(self, x):

        self.reset()
        self.state = x
        soln, var = self.obtain_solution(self.prev_sol, self.init_var)
        # if self.type == "CEM":
        #     self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        # else:
        #     pass
        action = soln[0]
        return action
        # pass

    def cost_function(self, actions):
        # trajectory_cost = 0
        # for i in range(len(actions)):
        #     trajectory_cost += cost_function(states[i], actions[i], next_states[i])
        # return trajectory_cost
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            state_next = self.predict(state, action) + state

            cost = -self.cost_predict(state_next, action)  # compute cost
            # cost = cost.reshape(costs.shape)
            costs += cost[:, 0] * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs
    
    def obtain_solution(self, *args, **kwargs):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.

        Returns: None
        """

        self.ub, self.lb = torch.FloatTensor(self.upper_bound), torch.FloatTensor(self.lower_bound)
        self.sampler = torch.distributions.uniform.Uniform(self.lb, self.ub)

        """Optimizes the cost function provided in setup().

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        solutions = self.sampler.sample(self.size).cpu().numpy()[:,:,0]
        #solutions = np.random.uniform(self.lb, self.ub, [self.popsize, self.sol_dim])
        costs = self.cost_function(solutions)
        return solutions[np.argmin(costs)], None
    
    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            delta_state = inputs
            delta_state = CPU(delta_state).numpy()
        return delta_state[:, 0:3]

    def cost_predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            reward = inputs
            reward = CPU(reward).numpy()
        return reward
class StochasticMPCController( Controller ):
    def __init__(self, system):
        pass

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