import numpy as np


def acting_force(x,phase):
    if phase == 1:
        Fx = 0.0002706*np.sin(3143*x+1.57) + 3.811e-05*np.sin(9430*x-1.574) \
             + 1.294e-05*np.sin(1.572e+04*x+1.566) + 5.519e-06*np.sin(2.2e+04*x-1.577)    
    elif phase == 2:
        Fx = 0.0002707*np.sin(3142*x-2.138e-12) + 3.815e-05*np.sin(9425*x-6.356e-12)\
             + 1.295e-05*np.sin(1.571e+04*x-1.05e-11) + 5.544e-06*np.sin(2.199e+04*x-1.471e-11)
    elif phase == 3:
        Fx = 0.0002706*np.sin(3143*x-1.572 ) + 3.811e-05*np.sin(9430*x+1.568)\
             + 1.294e-05*np.sin(1.572e+04*x-1.575) + 5.519e-06*np.sin(2.2e+04*x+1.564)
    else:
        Fx = 0.0002707*np.sin(3142*x+3.142) + 3.815e-05*np.sin(9425*x+3.142)\
             + 1.295e-05*np.sin(1.571e+04*x+3.142) + 5.544e-06*np.sin(2.199e+04*x+3.142)
    return Fx