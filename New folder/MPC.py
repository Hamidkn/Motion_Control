from scipy.signal import StateSpace, lsim, step
import numpy as np
import matplotlib.pyplot as plt
import time

# initialize system's parameters
R=100
L=1.3e-3
# k = 100 # Nm/A
# r = 0.5 # m
m = 0.0375 # kg
V = 18
dt = 1e-5

# state matrix
A = np.asarray([[0, 0, 0, -0.5001], 
                [0.5, 0, 0, 1.25],
                [0, 1, 0, -2.249],
                [0, 0, 2, 3.499]])

# input matrix
B = np.asarray([[-9.277e-11, -1.068e-9], 
                [-3.365e-11, 3.569e-10], 
                [-6.354e-11, 8.437e-11],
                [2.883e-10, 1.86e-10]] )

# output-matrix
C = np.asarray([[0, 0, 0, 2]])

# feedforward matrix
D = np.asarray([[0, 0]])

micro = StateSpace(A, B, C, D)

print(micro)

# plt.plot(A)
# plt.show()

# plt.plot(B)
# plt.show()

# define simulation steps in time
t = np.arange(0, 1, 1e-5)

# initialize input signal
U = np.zeros((t.shape[0], 2))
U[:, 1] = 1 # "enable" gravitation
U[:, 0] = 1 # set u(t) to 0
# U = (np.cos(2*np.pi*4*t) + 0.6*np.sin(2*np.pi*40*t) + 0.5*np.cos(2*np.pi*80*t))

# simulate the system
p, y, s = lsim(micro, U, t)
# we won't use y, since y=s[:, 0]

plt.plot(y)
plt.grid(alpha=0.3)
plt.show()


# start = time.time()
# for i in range(2):
#     plt.plot(y)
#     plt.grid(alpha=0.3)
#     plt.show()


# stop = time.time()
# print(f'elapsed time: {stop - start}')

# two seperate heater
from scipy.signal import StateSpace, TransferFunction, ss2tf

micro2 = ss2tf(A, B, C, D)

print(f'micro2 : {micro2[0]}')

num = micro2[0]
den = micro2[1]

sys = TransferFunction(num, den)
print(f'sys: {sys}')


p, y, s = lsim(micro2, U[:,0], t)
# we won't use y, since y=s[:, 0]

plt.plot(y)
plt.grid(alpha=0.3)
plt.show()