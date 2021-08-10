import numpy as np
import pandas as pd
from motion import profile
from plot import draw_plots


dt=1e-5
t = np.arange(0, 0.3, dt)
u=len(t)
g = 9.81
k = 0.213
step=0.5*1e-3
Sstep= 4*step

x1 = []
x2 = []
curr2 = []
curr1 = []
Ftotal = []
x1 = np.zeros(int(u))
x2 = np.zeros(int(u))
curr1 = np.zeros(int(u))
curr2 = np.zeros(int(u))
Ftotal = np.zeros(int(u))

x1, x2, Ftotal, curr1, curr2 = profile(Sstep,dt, u, x1, x2, Ftotal, curr1, curr2)
draw_plots(t,x1,x2,Ftotal,curr1,curr2)

inp = [t, curr1, curr2, Ftotal, x2, x1]
print(np.shape(inp))

nparray = np.array(inp)
transpose = nparray.transpose()
inp = transpose.tolist()
df = pd.DataFrame(inp, columns=('time', 'current1','current2','Ftotal','velocity','position'))

print(df)
print(df.shape)

# df.to_csv('motion_profile/Dataframe/dataframe.csv', index=False)