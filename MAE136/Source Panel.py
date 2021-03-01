import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

#Defining Panel Method
class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya=xa,ya
        self.xb, self.yb=xa,yb

        self.xc, self.yc = (xa+xb)/2,(ya+yb)/2
        self.length=np.sqrt((xb-xa)**2+(yb-ya)**2)

        if (xb-xa) <= 0.0:
            self.beta=np.arccos((yb-ya)/self.length)
        elif (xb-xa) > 0.0:
            self.beta=np.pi+np.arccos(-(yb-ya)/self.length)

        self.lambda_ = 0.0
        self.vt = 0.0
        self.cp = 0.0

#Integral of panel contribution at the center-point in the normal direction
def normalIntegral(p_i,p_j):
    def integrand(sourcesheet):
        return (((p_i.xc - (p_j.xa-np.sin(p_j.beta)*sourcesheet))*np.cos(p_i.beta) +
                 (p_i.yc - (p_j.ya+np.cos(p_j.beta)*sourcesheet))*np.sin(p_i.beta))/
                ((p_i.xc - (p_j.xa-np.sin(p_j.beta)*sourcesheet))**2 +
                 (p_i.yc - (p_j.ya+np.cos(p_j.beta)*sourcesheet))**2))
    return (integrate.quad(integrand,0.0,p_j.length)[0])

#Integral of panel contribution at the center-point in the tangential direction
def tangentialIntegral(p_i,p_j):
    def integrand(sourcesheet):
        return ((-(p_i.xc - (p_j.xa-np.sin(p_j.beta)*sourcesheet))*np.sin(p_i.beta) +
                  (p_i.yc - (p_j.ya+np.cos(p_j.beta)*sourcesheet))*np.cos(p_i.beta))/
                 ((p_i.xc - (p_j.xa-np.sin(p_j.beta)*sourcesheet))**2 +
                  (p_i.yc - (p_j.ya+np.cos(p_j.beta)*sourcesheet))**2))
    return (integrate.quad(integrand,0.0,p_j.length)[0])

#Freestream
U_inf = 1

#Definition of Geometry put in Freestream flow
#Cylinder
#delta theta
dtheta=100
#Radius
R = 1
#Center location
x_0, y_0 = 0.0, 0.0
#Theta array
theta_a = np.linspace(0.0,2*np.pi,dtheta)
X_cylinder, Y_cylinder = (x_0 + R*np.cos(theta_a),
                          y_0 + R*np.sin(theta_a))

#Plot of Cylinder
x_figsize = 6
y_figsize = 12
figure_1, (ax1, ax2) = plt.subplots(1,2, figsize=(y_figsize,x_figsize))
figure_1.suptitle('Source Panel Approximation Graphical Representation')
plt.subplots_adjust(top=0.75)
ax1.grid()
ax1.set_title('Coarse Mesh (n=8)')
ax1.set(xlabel='x',ylabel='y')
ax1.plot(X_cylinder,Y_cylinder,linewidth=2)
ax1.set_xlim([-2,2])
ax1.set_ylim([-2,2])

ax2.set_title('Fine Mesh (n=64)')
ax2.grid()
ax2.set(xlabel='x',ylabel='y')
ax2.plot(X_cylinder,Y_cylinder,linewidth=2, label='Circle')
ax2.set_xlim([-2,2])
ax2.set_ylim([-2,2])

### Course ###
#Break Geometry into Panels
number_panels = 8
#Panel endpoints
x_ends = R*np.cos(np.linspace(0.0,2*np.pi,number_panels+1))
y_ends = R*np.sin(np.linspace(0.0,2*np.pi,number_panels+1))
#Create Panels
panels=np.empty(number_panels,dtype=object)
for i in range(number_panels):
    panels[i]=Panel(x_ends[i],y_ends[i],x_ends[i+1],y_ends[i+1])
#Plot of panels
ax1.plot(x_ends,y_ends,linewidth=2)
ax1.scatter([p.xa for p in panels], [p.ya for p in panels],s=40)
ax1.scatter([p.xc for p in panels], [p.yc for p in panels],s=40,zorder=3)


#Source Influence Matrix
Matrix_A = np.empty((number_panels, number_panels), dtype=float)
np.fill_diagonal(Matrix_A, 0.5)
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            Matrix_A[i, j] = 0.5/np.pi * normalIntegral(p_i,p_j)
#RHS of system
b = -U_inf * np.cos([p.beta for p in panels])

lambda_ = np.linalg.solve(Matrix_A, b)
for i, panel in enumerate(panels):
    panel.lambda_ = lambda_[i]

#Source Influence Matrix 2
Matrix_A = np.empty((number_panels, number_panels), dtype=float)
np.fill_diagonal(Matrix_A, 0.0)
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            Matrix_A[i, j] = 0.5/np.pi * tangentialIntegral(p_i,p_j)
#RHS of system
b = -U_inf * np.sin([panel.beta for panel in panels])
#tangent velocity
V_tangential = np.dot(Matrix_A, lambda_) + b
for i, panel in enumerate(panels):
    panel.vt = V_tangential[i]

#Numerical Pressure Coefficient
for panel in panels:
    panel.cp = 1.0 - (panel.vt/U_inf)**2

#Convert Numerical Approximation Results to Polar Coordinates
center_tup=zip([p.xc for p in panels],[p.yc for p in panels])
delta_list=[]
for each_center in (list(center_tup)):
    if each_center[1]>0:
        delta=np.arctan2(each_center[1],each_center[0])
    elif each_center[1]<=0:
        delta=(2*np.pi + np.arctan2(each_center[1],each_center[0]))
    delta_list.append(delta)

#Analytical Pressure Coefficient
c_p = 1.0 - 4*(Y_cylinder/R)**2
figure_2, (axx1, axx2) = plt.subplots(1,2, figsize=(y_figsize,x_figsize))
plt.subplots_adjust(top=0.75)
figure_2.suptitle('Source Panel Approximation Pressure Coefficient Results')
axx1.grid()
axx1.set_title('Coarse Mesh (n=8)')
axx1.set(xlabel='x',ylabel='$C_p$')
axx1.plot(theta_a, c_p)
axx1.scatter(delta_list, [p.cp for p in panels],color='white',marker='v',edgecolors='r',s=100,zorder=2)

### Fine ###
#Break Geometry into Panels
number_panels = 64
#Panel endpoints
x_ends = R*np.cos(np.linspace(0.0,2*np.pi,number_panels+1))
y_ends = R*np.sin(np.linspace(0.0,2*np.pi,number_panels+1))
#Create Panels
panels=np.empty(number_panels,dtype=object)
for i in range(number_panels):
    panels[i]=Panel(x_ends[i],y_ends[i],x_ends[i+1],y_ends[i+1])
#Plot of panels
ax2.plot(x_ends,y_ends,linewidth=2)
ax2.scatter([p.xa for p in panels], [p.ya for p in panels],s=40)
ax2.scatter([p.xc for p in panels], [p.yc for p in panels],s=40,zorder=3, label='Circle Approximation')
ax2.legend(loc='upper right', bbox_to_anchor=(0.15,1.25))

#Source Influence Matrix
Matrix_A = np.empty((number_panels, number_panels), dtype=float)
np.fill_diagonal(Matrix_A, 0.5)
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            Matrix_A[i, j] = 0.5/np.pi * normalIntegral(p_i,p_j)
#RHS of system
b = -U_inf * np.cos([p.beta for p in panels])

lambda_ = np.linalg.solve(Matrix_A, b)
for i, panel in enumerate(panels):
    panel.lambda_ = lambda_[i]

#Source Influence Matrix 2
Matrix_A = np.empty((number_panels, number_panels), dtype=float)
np.fill_diagonal(Matrix_A, 0.0)
for i, p_i in enumerate(panels):
    for j, p_j in enumerate(panels):
        if i != j:
            Matrix_A[i, j] = 0.5/np.pi * tangentialIntegral(p_i,p_j)
#RHS of system
b = -U_inf * np.sin([panel.beta for panel in panels])
#tangent velocity
V_tangential = np.dot(Matrix_A, lambda_) + b
for i, panel in enumerate(panels):
    panel.vt = V_tangential[i]

#Numerical Pressure Coefficient
for panel in panels:
    panel.cp = 1.0 - (panel.vt/U_inf)**2

#Convert Numerical Approximation Results to Polar Coordinates
center_tup=zip([p.xc for p in panels],[p.yc for p in panels])
delta_list=[]
for each_center in (list(center_tup)):
    if each_center[1]>0:
        delta=np.arctan2(each_center[1],each_center[0])
    elif each_center[1]<=0:
        delta=(2*np.pi + np.arctan2(each_center[1],each_center[0]))
    delta_list.append(delta)

#Analytical Pressure Coefficient
c_p = 1.0 - 4*(Y_cylinder/R)**2
axx2.grid()
axx2.set_title('Fine Mesh (n=64)')
axx2.set(xlabel='x',ylabel='$C_p$')
axx2.plot(theta_a, c_p, label='Analytical Solution')
axx2.scatter(delta_list, [p.cp for p in panels],color='white',marker='v',edgecolors='r',s=100,zorder=2, label='Numerical Approximation')
axx2.legend(loc='upper right', bbox_to_anchor=(0.15,1.25))

figure_1.savefig('Source Panel Circle Plot')
figure_2.savefig('Source Panel Cp')
