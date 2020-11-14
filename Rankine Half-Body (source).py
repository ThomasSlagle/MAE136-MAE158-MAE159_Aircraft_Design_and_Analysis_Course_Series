import numpy as np
import math
from matplotlib import pyplot as plt

#number of points/itterations
n = 200

#Boundary Conditions
x_i = -1
x_f = 1
y_i = -1
y_f = 1
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


plt.contour(X, Y, psi,
               levels=[-sourceStrength / 2, sourceStrength / 2],
               colors='black', linewidths=2, linestyles='solid')
plt.savefig("Half Rankine Body")
plt.close()

#Cp plot
b = sourceStrength/(2*np.pi*U_inf)
def c_p (theta):
    r = b*(np.pi-theta)/np.sin(theta)
    Velocity = np.sqrt(U_inf**2*(1+2*b/r*np.cos(theta) + b**2/r**2))
    return 1.0 - Velocity**2/U_inf**2

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
plt.savefig("Cp Half Rankine Body")
