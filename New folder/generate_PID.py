import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import random

# PID Parameters
Kc   = 6.0
tauI = 75.0 # sec
tauD = 0.0  # sec
dataframe = 'E:/UniversityFiles/Control-20210210T123743Z-001/Control/homework/MainModel_Version1/Mainmodel_vnew/Motion_Control/LSTM_model/dataframe1/dataframe1.txt'
df = pd.read_csv(dataframe)
dataset = df.values
# ggg = dataset[:,2]
# # print(ggg)
# plt.plot(ggg)
# plt.show()

#-----------------------------------------
# PID Controller
#-----------------------------------------
# inputs ---------------------------------
# sp = setpoint
# pv = current temperature
# pv_last = prior temperature
# ierr = integral error
# dt = time increment between measurements
# outputs --------------------------------
# op = output of the PID controller
# P = proportional contribution
# I = integral contribution
# D = derivative contribution
def pid(sp,pv,pv_last,ierr,dt):
    # PID coefficients in terms of tuning parameters
    KP = Kc
    KI = Kc / tauI
    KD = Kc * tauD
    
    # ubias for controller (initial heater)
    op0 = 0 
    
    # upper and lower bounds on heater level
    ophi = 100
    oplo = 0
    
    # calculate the error
    error = sp - pv
    
    # calculate the integral error
    ierr = ierr + KI * error * dt
    
    # calculate the measurement derivative
    dpv = (pv - pv_last) / dt
    
    # calculate the PID output
    P = KP * error
    I = ierr
    D = -KD * dpv
    op = op0 + P + I + D
    
    # implement anti-reset windup
    if op < oplo or op > ophi:
        I = I - KI * error * dt
        # clip output
        op = max(oplo,min(ophi,op))
        
    # return the controller output and PID terms
    return [op,P,I,D]


##### Set up run parameters #####

# Run time in minutes
run_time = len(dataset)

# Number of cycles
loops = int(run_time)

# arrays for storing data
F = np.zeros(loops) # Force
Pos = np.zeros(loops) # Position
tm = np.zeros(loops) # Time
sp1 = np.ones(loops)
# set point (degC)
for i in range(loops):
    ss = dataset[i,2]
    # print(ss)
    sp1 [i] = sp1 [i] * ss
print(np.shape(sp1))
print(f'sp1: {sp1}')

# vary temperature setpoint
end = 1 # leave 1st 30 seconds of temp set point as room temp
while end <= loops:
    start = end
    # keep new temp set point value for anywhere from 4 to 10 min
    end += random.randint(240,600) 
    sp1[start:end] = random.uniform(0,1) * 1e-3

# Plot
plt.plot(sp1)
plt.xlabel('Time',size=14)
plt.ylabel(r'SP',size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('Images/SP_profile.png')

# with dataset as ds:
# Find current T1, T2
# print('Temperature 1: {0:0.2f} °C'.format(ds[2]))
    # print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

    # Integral error
ierr = 0.0
    # Integral absolute error
iae = 0.0
start_time = time.time()
prev_time = start_time

for i in range(loops):
       # Delay 1 second
    if time.time() > prev_time + 1.0:
            print('Exceeded cycle time by ',time.time()-prev_time-1.0)
    else:
        while time.time() < prev_time + 1.0:
                pass
        
        # Record time and change in time
    t = time.time()
    dt = t - prev_time
    prev_time = t
    tm[i] = t - start_time

    # Read temperatures in Kelvin 
    F[i] = dataset[i,2]

        # Integral absolute error
    iae += np.abs(sp1[i]-F[i])

        # Calculate PID output
    [Pos[i],P,ierr,D] = pid(sp1[i],F[i],F[i-1],ierr,dt)

        # Write heater output (0-100)
        # lab.Q1(Vel[i])
    Pos[i] = Pos[i]

        # Print line of data
    # print(('{:6.1f} {:6.2f} {:6.2f} ' + \
    #           '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
    #               tm[i],sp1[i],F[i], \
    #               Pos[i],P,ierr,D,iae))

stop = time.time()  
print(f'elapsed time: {stop - start_time} s')
       
# Save csv file
data = pd.DataFrame()
data['Pos'] = Pos
data['F'] = F
data['sp'] = sp1
data.to_csv('PID_train_data.csv',index=False)

# Plot
plt.plot(data['Pos'],'b-',label='$pos$ (%)')
plt.plot(data['F'],'r-',label='$F$ $(^oC)$')
plt.plot(data['sp'],'k-',label='SP $(^oC)$')
plt.legend()
plt.savefig('Images/PID_train.png')
plt.show()
