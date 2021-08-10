from acting_force import force
import numpy as np

def profile(Step,dt, u, x1, x2, Ftotal, curr1, curr2):
    R=100
    L=1.3e-3
    m=(0.0075 + 5 * 0.006) * 1e-3;
    g = 9.81
    k = 0.213
    for i in range(u - 1):
        x1[i+1] = dt*x2[i] + x1[i]
        if (np.mod(x1[i], Step) < 0.5*1e-3):
            phase = 1
            curr = curr1[i]
            curr2[i]=0
            
            v1=18
            v2=0
            Fx, Fz = force(x1[i], phase)
            Fn = Fx - k * (Fz - m *g)
            # Ftotal[i] = np.abs(Fx)
            Ftotal[i] = np.abs(Fn)
            # if Ftotal[i] == 0 :
            #     continue
        elif ( 0.5*1e-3 < np.mod(x1[i], Step) and np.mod(x1[i], Step) < 1.0*1e-3):
            # print('Phase 2 ....')
            phase = 2
            curr = curr2[i]
            curr1[i] = 0
            v1 = 0
            v2 = 18
            Fx, Fz = force(x1[i], phase)
            Fn = Fx - k * (Fz - m *g)
            Ftotal[i] = np.abs(Fn)
        #     if Ftotal[i] == 0:
        #         continue
        elif (1.0*1e-3 < np.mod(x1[i], Step) and np.mod(x1[i], Step) < 1.5*1e-3):
            # print('Phase 3 ....')
            phase = 3
            curr = curr1[i]
            curr2[i] = 0
            v2 = 0
            v1 = -18
            Fx, Fz = force(x1[i], phase)
            Fn = Fx - k * (Fz - m *g)
            Ftotal[i] = np.abs(Fn)
        #     if Ftotal[i] == 0:
        #         continue
        elif (1.5*1e-3 < np.mod(x1[i], Step) and np.mod(x1[i], Step) < 2.0*1e-3):
            # print('phase 4 .....')
            phase = 4
            curr = curr2[i]
            curr1[i] = 0
            v1 = 0
            v2 = -18
            Fx, Fz = force(x1[i], phase)
            Fn = Fx - k * (Fz - m *g)
            Ftotal[i] = np.abs(Fn)
        # else:
        #     break

        curr1[i+1]= dt*(((-R/L)*curr1[i]) + v1*(1/L)) + curr1[i]
        curr2[i+1]= dt*(((-R/L)*curr2[i]) + v2*(1/L)) + curr2[i]
        x2[i+1]= (dt*Ftotal[i]*(curr/0.1))/m + x2[i]

    return x1, x2, Ftotal, curr1, curr2