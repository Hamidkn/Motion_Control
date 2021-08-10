import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import pandas as pd



def current(i, t, phase, V):
        R=100
        L=1.3e-3

        if phase == 1 or phase == 2:
            V = V[0]
        elif phase == 3 or phase == 4:
            V = V[1]
    
        didt = V / L - (R / L) * i

        return didt

def position(x, t, m, v,u):
        '''
        x = [position, velocity, current]
        '''
        # m = 0.0375
        # m = 0.0362 * 1e-3
        g = 9.81
        k = 0.213
        R = 100
        L = 1.3e-3
        pos, vel, curr = x
        # u=len(t)
        # m = m
        for i in range(u):
            if i < 25000:
                Fx = 0.0002706*np.sin(3143*pos+1.57) + 3.811e-05*np.sin(9430*pos-1.574) \
             + 1.294e-05*np.sin(1.572e+04*pos+1.566) + 5.519e-06*np.sin(2.2e+04*pos-1.577)

            # Fz = 0.0004986*np.sin(72.31*x[0]+0.1257) + 0.0002715*np.sin(6287*x[0]-1.573) + \
            #         0.0002042*np.sin(144.7*x[0]+1.82) + 3.469e-05*np.sin(290.5*x[0]+2.048) + \
            #         2.151e-05*np.sin(1.886e+04*x[0]-1.573) + 1.017e-05*np.sin(439.2 *x[0]+2.219) + \
            #         1.798e-06*np.sin(596.1*x[0]+2.225) + 9.211e-06*np.sin(2.515e+04*x[0]-1.552)
                V = v

            elif 25000 < i and i < 50000:
                Fx = 0.0002707*np.sin(3142*pos-2.138e-12) + 3.815e-05*np.sin(9425*pos-6.356e-12)\
                + 1.295e-05*np.sin(1.571e+04*pos-1.05e-11) + 5.544e-06*np.sin(2.199e+04*pos-1.471e-11)
            
            # Fz=0.0005039*np.sin(78.54*x[0]-1.038e-07) + 0.0002713*np.sin(6283*x[0]+1.571) + \
            #     0.000214*np.sin(157.1*x[0]+1.571) + 4.287e-05*np.sin(314.2*x[0]+1.571) + \
            #     2.112e-05*np.sin(1.885e+04*x[0]+1.571) + 1.842e-05*np.sin(471.2*x[0]+1.571) \
            #         + 1.028e-05*np.sin(628.3*x[0]+1.571) + 9.519e-06*np.sin(2.513e+04*x[0]-1.571)
                V = v
            elif 50000 < i and i < 75000:
                Fx = 0.0002706*np.sin(3143*pos-1.572 ) + 3.811e-05*np.sin(9430*pos+1.568)\
                + 1.294e-05*np.sin(1.572e+04*pos-1.575) + 5.519e-06*np.sin(2.2e+04*pos+1.564)
            
            # Fz = 0.0004986*np.sin(72.31*x[0]+0.1257) + 0.0002715*np.sin(6287*x[0]-1.573) + \
            #         0.0002042*np.sin(144.7*x[0]+1.82) + 3.469e-05*np.sin(290.5*x[0]+2.048) + \
            #         2.151e-05*np.sin(1.886e+04*x[0]-1.573) + 1.017e-05*np.sin(439.2 *x[0]+2.219) + \
            #         1.798e-06*np.sin(596.1*x[0]+2.225) + 9.211e-06*np.sin(2.515e+04*x[0]-1.552)
                V = -v
            else:
                Fx = 0.0002707*np.sin(3142*pos+3.142) + 3.815e-05*np.sin(9425*pos+3.142)\
                 + 1.295e-05*np.sin(1.571e+04*pos+3.142) + 5.544e-06*np.sin(2.199e+04*pos+3.142)
            
            # Fz=0.0005039*np.sin(78.54*x[0]-1.038e-07) + 0.0002713*np.sin(6283*x[0]+1.571) + \
            #     0.000214*np.sin(157.1*x[0]+1.571) + 4.287e-05*np.sin(314.2*x[0]+1.571) + \
            #     2.112e-05*np.sin(1.885e+04*x[0]+1.571) + 1.842e-05*np.sin(471.2*x[0]+1.571) \
            #         + 1.028e-05*np.sin(628.3*x[0]+1.571) + 9.519e-06*np.sin(2.513e+04*x[0]-1.571)
                V = -v

        # Fn = Fx - k * (Fz - m *g)
        didt = V / L - (R / L) * curr
        dx1dt = vel
        dx2dt = (Fx * didt) / (0.1 * m)
        dxdt = [dx1dt, dx2dt, didt]

        return dxdt


dt=1e-6
t = np.arange(0, 0.1, dt)
u=len(t)
g = 9.81
k = 0.213
step=5*1e3
Sstep= 4*step
R = 100
L = 1.3e-3
m = 0.0362 * 1e-3
# x = np.array([[0], [0], [0]])
x = []
v = 18
m = 0.0362 * 1e-3
start = time.time()
print(f'Start time: {start}')
st = 4
# for i in range(u):
#     if(st == 4):
#         phase = 1
#         x = odeint(position,[0,0,0], t, args=(phase,m))[-1]
#         print(x)
#         plt.plot(t,x)
#         plt.show()
#         st = 1

    
#     elif(st == 1):
#         phase = 2
#         ind = len(x)
#         print(x[ind-1])
#         x.append(odeint(position,x[-1], t, args=(phase,m)))
#         st = 2
    
#     elif(st == 2):
#         phase = 3
#         x.append(odeint(position,x[-1], t, args=(phase,m)))
#         st = 3
    
#     elif (st==3):
#         phase = 4
#         x.append(odeint(position,x[-1], t, args=(phase,m)))
#         st = -1
#     elif(st==-1):
#         break

x = odeint(position,[0,0,0], t = np.arange(0, 0.1, dt), args=(m,v,u))
end = time.time()
print(f'elapsed time: {end - start} s')

print(np.shape(x))
print(x)

# pos = x[:,0]
# vel = x[:,1]
# cur = x[:,2]

nparray = np.array(x)
transpose = nparray.transpose()
xrange = transpose.tolist()
df = pd.DataFrame(x)

print(df)
# Position
plt.plot(t,x[:,0])
# plt.savefig('position.png')
plt.show()

# Current
plt.plot(t,x[:,2])
# plt.plot(x1,curr2)
# plt.savefig('current.png')
plt.show()

# Velocity
plt.plot(t,x[:,1])
# plt.savefig('velocity.png')
plt.show()