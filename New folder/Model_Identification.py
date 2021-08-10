from gekko import GEKKO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data and parse into columns
url = 'E:/UniversityFiles/Control-20210210T123743Z-001/Control/homework/MainModel_Version1/Mainmodel_vnew/Motion_Control/motion_ode_solver/dataframe.txt'
data = pd.read_csv(url)
t = data['time']
u = data[['current1','current2']]
y = data['position']

# generate time-series model
m = GEKKO(remote=False) # remote=True for MacOS

# system identification
na = 2 # output coefficients
nb = 2 # input coefficients
yp,p,K = m.sysid(t,u,y,na,nb,diaglevel=1,pred='meas')

print(f'yp: {yp}')
print(f'p: {p}')
print(f'K: {K}')

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,u)
plt.legend([r'$u_0$'])
plt.ylabel('MVs')
plt.subplot(2,1,2)
plt.plot(t,y)
plt.plot(t,yp)
plt.legend([r'$y_0$',r'$y_1$'])
plt.ylabel('CVs')
plt.xlabel('Time')
plt.savefig('motion_ode_solver/Images/sysid.png')
plt.show()


# step test model
yc,uc = m.arx(p)

print(f'uc : {uc}')
print(f'yc : {yc}')

# # steady state initialization
# m.options.IMODE = 1
# m.solve(disp=False)

# # dynamic simulation (step tests)
# m.time = np.linspace(0,500,501)
# m.options.TIME_SHIFT=0
# m.options.IMODE = 4
# m.solve(disp=False)

# plt.figure()
# yc = np.transpose(yc)
# # step for first MV (Heater 1)
# uc[0].value = np.zeros(len(m.time))
# uc[0].value[5:] = 100
# uc[1].value = np.zeros(len(m.time))
# m.solve(disp=False)



# plt.subplot(2,2,1)
# plt.title('Step Test 1')
# plt.plot(uc[0].value,'b-',label=r'$H_1$')
# plt.plot(uc[1].value,'r-',label=r'$H_2$')
# # plt.ylabel('Heater (%)')
# # plt.legend()
# plt.subplot(2,2,3)
# plt.plot(yc,'b--',label=r'$T_1$')
# # plt.plot(m.time,yc[1].value,'r--',label=r'$T_2$')
# # plt.ylabel('Temperature (K)')
# # plt.xlabel('Time (sec)')
# # plt.legend()

# # step for second MV (Heater 2)
# uc[0].value = np.zeros(len(m.time))
# uc[1].value = np.zeros(len(m.time))
# uc[1].value[5:] = 100
# m.solve(disp=False)
# plt.subplot(2,2,2)
# plt.title('Step Test 2')
# plt.plot(uc[0].value,'b-',label=r'$H_1$')
# plt.plot(uc[1].value,'r-',label=r'$H_2$')
# # plt.ylabel('Heater (%)')
# # plt.legend()
# plt.subplot(2,2,4)
# plt.plot(yc,'b--',label=r'$T_1$')
# # plt.plot(m.time,yc[1].value,'r--',label=r'$T_2$')
# # plt.ylabel('Temperature (K)')
# # plt.xlabel('Time (sec)')
# # plt.legend()

# plt.show()