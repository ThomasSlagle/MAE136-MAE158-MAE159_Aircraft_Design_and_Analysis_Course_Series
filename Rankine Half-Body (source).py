import numpy as np
import math
from matplotlib import pyplot as plt

#number of points/itterations
n = 200

#Boundary Conditions
x_i = -4
x_f = 4
y_i = -4
y_f = 4
x = np.linspace(x_i,x_f,n)
y = np.linspace(y_i,y_f,n)
X,Y = np.meshgrid(x,y)

#Freestream Infomration
U_inf = 1

u_free = U_inf*np.ones((n,n),dtype=float)
v_free = U_inf*np.zeros((n,n),dtype=float)
#stream-function of freestream
psi_free = U_inf*Y

#Source Information
def sourceVelocity (str, xs, ys, X, Y):
    u = str/(2*np.pi) * (X-xs)/((X-xs)**2 + (Y-ys)**2)
    v = str/(2*np.pi) * (Y-ys)/((X-xs)**2 + (Y-ys)**2)
    return u, v

def sourceStreamFunction (str, xs, ys, X, Y):
    psi = str/(2*np.pi) * np.arctan2((Y-ys),(X-xs))
    return psi

sourceStrength = 1
x_source, y_source = 0.0,0.0
u_source, v_source = sourceVelocity(sourceStrength, x_source, y_source, X, Y)
psi_source = sourceStreamFunction(sourceStrength, x_source, y_source, X, Y)

#Super Position
u = u_free + u_source
v = v_free + v_source
psi = psi_free + psi_source

#plot
width = 10
height = (y_f-y_i)/(x_f-x_i) * width
plt.figure(figsize=(width,height))
plt.grid(True)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Streamlines')
plt.xlim(x_i,x_f)
plt.ylim(y_i,y_f)
plt.streamplot(X,Y,u,v,density=2, linewidth=1, arrowsize=1, arrowstyle='-')
#plt.scatter(x_source,y_source,color='blue',s=80, marker='o')

#stagpoints
x_stag = x_source - sourceStrength / (2*np.pi*U_inf)
y_stag = y_source
#plt.scatter(x_stag,y_stag, color='g',s=90,marker='o')

plt.contour(X, Y, psi,
               levels=[-sourceStrength / 2, sourceStrength / 2],
               colors='black', linewidths=2, linestyles='solid')
plt.show()
plt.close()

#Cp plot
def c_p (theta):
    return -2*np.cos(theta)*np.sin(theta/2)**2-np.sin(theta/2)**4

Theta = np.linspace(0,2*np.pi,n)
count = 0
C_p = np.array([])
for count in range(len(Theta)):
    x = c_p(Theta.item(count))
    C_p = np.append(C_p,[x])
    count=count+1
plt.plot(Theta,C_p)
plt.xlabel('$\u03B8_{body}$')
plt.ylabel('$c_p$')
plt.title('Body Coefficient of Pressure v Body Angle')
plt.show()
