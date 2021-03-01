import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline, BSpline

alpha = np.array([-2.0,0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0])
c_l = np.array([0.05, 0.25, 0.44, 0.64, 0.85, 1.08, 1.26, 1.43, 1.56])
c_d = np.array([0.006, 0.006, 0.006, 0.007, 0.0075, 0.0092, 0.0115, 0.0150, 0.0186])
c_mc_4 = np.array([-0.042, -0.040, -0.038, -0.036, -0.036, -0.036, -0.034, -0.030, -0.025])


c_n = np.array([])
for i in range (0, len(c_mc_4)):
    x = c_l.item(i)*np.cos(math.radians(alpha[i]))+c_d.item(i)*np.sin(math.radians(alpha[i]))
    c_n = np.append(c_n, [x])

x_cp_c = np.array([])
for i in range (0, len(c_mc_4)):
    y = 1/4-c_mc_4.item(i)/c_n.item(i)
    x_cp_c = np.append(x_cp_c, [y])

#print(x_cp_c)

x_new = np.linspace(-2,14,300)
spl = make_interp_spline(alpha, x_cp_c, k=3)
y_new = spl(x_new)

plt.plot(x_new,y_new, "--", alpha = 0.4, color = 'black')
plt.plot(alpha,x_cp_c, ".")

plt.xlabel("\u03B1 (angle of attack) (\u00b0)")
plt.ylabel("$X_{cp}$/c")
plt.title('Normalized Center of Pressure v Angle of Attack')

plt.savefig("Problem 1.6.png")
