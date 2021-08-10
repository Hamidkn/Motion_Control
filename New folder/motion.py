from acting_force import acting_force

def motion_profile(dt, u, x1, x2, Ftotal, curr1, curr2):
    R=100
    L=1.3e-3
    m=(0.0075 + 5 * 0.006);
    for i in range(u - 1):
        x1[i+1] = dt*x2[i] + x1[i]
        if (x1[i] < 0.0005):
            phase = 1
            curr = curr1[i]
            curr2[i]=0
            v1=18
            v2=0
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx
            # if Ftotal[i] == 0 :
            #     continue
        elif ( 0.0005 < x1[i] and x1[i] < 0.001):
            # print('Phase 2 ....')
            phase = 2
            curr = curr2[i]
            curr1[i] = 0
            v1 = 0
            v2 = 18
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx
        #     if Ftotal[i] == 0:
        #         continue
        elif (0.001 < x1[i] and x1[i] < 0.0015):
            # print('Phase 3 ....')
            phase = 3
            curr = curr1[i]
            curr2[i] = 0
            v2 = 0
            v1 = -18
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx
        #     if Ftotal[i] == 0:
        #         continue
        elif (0.0015 < x1[i] and x1[i] < 0.002):
            # print('phase 4 .....')
            phase = 4
            curr = curr2[i]
            curr1[i] = 0
            v1 = 0
            v2 = -18
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx
        # else:
        #     break

        curr1[i+1]= dt*(((-R/L)*curr1[i]) + v1*(1/L)) + curr1[i]
        curr2[i+1]= dt*(((-R/L)*curr2[i]) + v2*(1/L)) + curr2[i]
        x2[i+1]= (dt*Ftotal[i]*(curr/0.1))/m + x2[i]

    return x1, x2, Ftotal, curr1, curr2