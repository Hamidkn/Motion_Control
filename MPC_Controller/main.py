import numpy as np
from pylab import *
from Controllers import LQController, MPCController, PIDController, Controller
from systems import NoisyDtLTISystem, DtLTISystem
from Estimator import KalmanStateObserver
from simulation import SimEnv
from plant import Plant


if __name__ == '__main__':
    
    # sampling time. 
    Ts = 0.05

    # Process and measurement noise covariance matrice
    Sv = np.matrix( [[1**2]] )
    
    # introduce a random variation in the robot's accelleration.
    Sw = 1e1 * np.matrix( [[10, 10, 10],
                            [10, 10, 10],
                            [10, 10, 10]] )
    # Sw = 1e-4 * np.eye(3)
    Q= np.eye(3)
    R= 0.1 * np.eye(1)

    # define the system
    plant = Plant( Ts=Ts, Sw=Sw, Sv=Sv, x0 = np.matrix( [[0.0],[0.0],[0.0]] ) )
    
    # define controller. 
    # controller = LQController(plant, Q, R)
    controller = PIDController(plant)
    # controller = Controller()
    
    # create kalman state observer object
    kalman_observer = KalmanStateObserver( plant )
    
    # Create simulator.
    sim = SimEnv( plant, controller=controller, observer=kalman_observer )
        
    # run simulation for 10 seconds
    res = sim.simulate( 2 )

    # Plotting displacement and velocity
    rcParams['figure.figsize'] = (15, 12)
    rcParams['font.size'] = 10

    # now do plots
    subplot(311)
    plot ( res.t, res.y[0], 'b.', label='Position measured' )
    plot ( res.t, res.x[0], 'r-', label='Position estimated' )
    legend(loc=2, numpoints=1)
    ylabel( 'x [m]' )
    grid()
    
    subplot(312, sharex=gca())
    plot ( res.t, res.x[1], 'r-', label='Velocity estimated' )
    ylabel( 'v [m/s]' )
    legend(loc=2, numpoints=1)
    grid()
    
    subplot(313, sharex=gca())
    plot ( res.t, res.u[0], 'b-', label='Input' )
    plot ( res.t, res.sp[0], 'r-', label='setpoint')
    xlabel( 't [s]' )
    ylabel( 'u' )
    legend(loc=2, numpoints=1)
    grid()
    
    show()