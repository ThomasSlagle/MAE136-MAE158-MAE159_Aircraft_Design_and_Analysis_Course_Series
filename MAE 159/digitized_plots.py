import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.optimize import curve_fit

def fix_data(array, sweep, xy, x):
    for i in x.index:
        array = np.append(array, [x[sweep + ', ' + xy][i]])
    nans=np.where(np.isfinite(array))
    array = array[nans]
    return array

def fix_data2(array, xy, x):
    for i in x.index:
        array = np.append(array, [x[xy][i]])
    nans=np.where(np.isfinite(array))
    array = array[nans]
    return array

def test_lin(x, M, B):
    return M*x+B

def linear (x_val, coefs):
    return coefs[0]*x_val + coefs[1]

def test_2nd(x, a, b, c):
    return a*x**2 + b*x + c

def order_2nd(x_val, coefs):
        return coefs[0]*x_val**2 + coefs[1]*x_val + coefs[2]

def test_3rd(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def order_3rd(x_val, coefs):
        return coefs[0]*x_val**3 + coefs[1]*x_val**2 + coefs[2]*x_val + coefs[3]

def test_s3(x, a, b):
    return a*x**b

def order_s3(x_val, coefs):
    return coefs[0]*x_val**coefs[1]

#Effects of t/c and sweep on the delta_M_div
fig_1a = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 1a')

#0 degree sweep
Mdiv_0 = np.array([])
Mdiv_0 = fix_data(Mdiv_0,'0','Mdiv', fig_1a)

tc_0 = np.array([])
tc_0 = fix_data(tc_0,'0','tc', fig_1a)

#10 degree sweep
Mdiv_10 = np.array([])
Mdiv_10 = fix_data(Mdiv_10,'10','Mdiv', fig_1a)

tc_10 = np.array([])
tc_10 = fix_data(tc_10,'10','tc', fig_1a)

#15 degree sweep
Mdiv_15 = np.array([])
Mdiv_15 = fix_data(Mdiv_15,'15','Mdiv', fig_1a)

tc_15 = np.array([])
tc_15 = fix_data(tc_15,'15','tc', fig_1a)

#20 degree sweep
Mdiv_20 = np.array([])
Mdiv_20 = fix_data(Mdiv_20,'20','Mdiv', fig_1a)

tc_20 = np.array([])
tc_20 = fix_data(tc_20,'20','tc', fig_1a)

#25 degree sweep
Mdiv_25 = np.array([])
Mdiv_25 = fix_data(Mdiv_25,'25','Mdiv', fig_1a)

tc_25 = np.array([])
tc_25 = fix_data(tc_25,'25','tc', fig_1a)

#30 degree sweep
Mdiv_30 = np.array([])
Mdiv_30 = fix_data(Mdiv_30,'30','Mdiv', fig_1a)

tc_30 = np.array([])
tc_30 = fix_data(tc_30,'30','tc', fig_1a)

#35 degree sweep
Mdiv_35= np.array([])
Mdiv_35 = fix_data(Mdiv_35,'35','Mdiv', fig_1a)

tc_35 = np.array([])
tc_35 = fix_data(tc_35,'35','tc', fig_1a)

#40 degree sweep
Mdiv_40 = np.array([])
Mdiv_40 = fix_data(Mdiv_40,'40','Mdiv', fig_1a)

tc_40 = np.array([])
tc_40 = fix_data(tc_40,'40','tc', fig_1a)



coef_1a_0, cov_1a_0 = curve_fit(test_lin, Mdiv_0, tc_0)

coef_1a_10, cov_1a_10 = curve_fit(test_lin, Mdiv_10, tc_10)

coef_1a_15, cov_1a_15 = curve_fit(test_lin, Mdiv_15, tc_15)

coef_1a_20, cov_1a_20 = curve_fit(test_lin, Mdiv_20, tc_20)

coef_1a_25, cov_1a_25 = curve_fit(test_lin, Mdiv_25, tc_25)

coef_1a_30, cov_1a_30 = curve_fit(test_lin, Mdiv_30, tc_30)

coef_1a_35, cov_1a_35 = curve_fit(test_lin, Mdiv_35, tc_35)

coef_1a_40, cov_1a_40 = curve_fit(test_lin, Mdiv_40, tc_40)

#Supercritical
fig_1b = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 1b')

#0 degree sweep
Mdiv_0 = np.array([])
Mdiv_0 = fix_data(Mdiv_0,'0','Mdiv', fig_1b)

tc_0 = np.array([])
tc_0 = fix_data(tc_0,'0','tc', fig_1b)

#5 degree sweep
Mdiv_5 = np.array([])
Mdiv_5 = fix_data(Mdiv_5,'5','Mdiv', fig_1b)

tc_5 = np.array([])
tc_5 = fix_data(tc_5,'5','tc', fig_1b)

#10 degree sweep
Mdiv_10 = np.array([])
Mdiv_10 = fix_data(Mdiv_10,'10','Mdiv', fig_1b)

tc_10 = np.array([])
tc_10 = fix_data(tc_10,'10','tc', fig_1b)

#15 degree sweep
Mdiv_15 = np.array([])
Mdiv_15 = fix_data(Mdiv_15,'15','Mdiv', fig_1b)

tc_15 = np.array([])
tc_15 = fix_data(tc_15,'15','tc', fig_1b)

#20 degree sweep
Mdiv_20 = np.array([])
Mdiv_20 = fix_data(Mdiv_20,'20','Mdiv', fig_1b)

tc_20 = np.array([])
tc_20 = fix_data(tc_20,'20','tc', fig_1b)

#25 degree sweep
Mdiv_25 = np.array([])
Mdiv_25 = fix_data(Mdiv_25,'25','Mdiv', fig_1b)

tc_25 = np.array([])
tc_25 = fix_data(tc_25,'25','tc', fig_1b)

#30 degree sweep
Mdiv_30 = np.array([])
Mdiv_30 = fix_data(Mdiv_30,'30','Mdiv', fig_1b)

tc_30 = np.array([])
tc_30 = fix_data(tc_30,'30','tc', fig_1b)

#35 degree sweep
Mdiv_35= np.array([])
Mdiv_35 = fix_data(Mdiv_35,'35','Mdiv', fig_1b)

tc_35 = np.array([])
tc_35 = fix_data(tc_35,'35','tc', fig_1b)

#40 degree sweep
Mdiv_40 = np.array([])
Mdiv_40 = fix_data(Mdiv_40,'40','Mdiv', fig_1b)

tc_40 = np.array([])
tc_40 = fix_data(tc_40,'40','tc', fig_1b)

coef_1b_0, cov_1b_0 = curve_fit(test_2nd, Mdiv_0, tc_0)

coef_1b_5, cov_1b_5 = curve_fit(test_2nd, Mdiv_5, tc_5)

coef_1b_10, cov_1b_10 = curve_fit(test_2nd, Mdiv_10, tc_10)

coef_1b_15, cov_1b_15 = curve_fit(test_2nd, Mdiv_15, tc_15)

coef_1b_20, cov_1b_20 = curve_fit(test_2nd, Mdiv_20, tc_20)

coef_1b_25, cov_1b_25 = curve_fit(test_2nd, Mdiv_25, tc_25)

coef_1b_30, cov_1b_30 = curve_fit(test_2nd, Mdiv_30, tc_30)

coef_1b_35, cov_1b_35 = curve_fit(test_2nd, Mdiv_35, tc_35)

coef_1b_40, cov_1b_40 = curve_fit(test_2nd, Mdiv_40, tc_40)



#effects of Wing Drag Divergence due to Lift
fig_2 = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 2')

#Conventional
Delta_md_c = np.array([])
Delta_md_c = fix_data2(Delta_md_c, 'Delta Mdiv, c', fig_2)

C_l_c = np.array([])
C_l_c = fix_data2(C_l_c, 'C_L, c', fig_2)

coef_2_c, cov_2_c = curve_fit(test_2nd, C_l_c, Delta_md_c)


#Supercritical
Delta_md_s = np.array([])
Delta_md_s = fix_data2(Delta_md_s, 'Delta Mdiv, s', fig_2)

C_l_s = np.array([])
C_l_s = fix_data2(C_l_s, 'C_L, s', fig_2)

coef_2_s, cov_2_s = curve_fit(test_2nd, C_l_s, Delta_md_s)



# Figure 3 C_L_max Conceptual Design Estimates @M=0.2
fig_3 = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 3')

#Takeoff
C_takeoff = np.array([])
C_takeoff = fix_data2(C_takeoff, 't, cos(lambda)^2 * (t/c)^2 * AR', fig_3)

C_l_takeoff = np.array([])
C_l_takeoff = fix_data2(C_l_takeoff, 't, C_L_max', fig_3)

coef_3_t, cov_3_t = curve_fit(test_2nd, C_takeoff, C_l_takeoff)

#Landing
C_landing = np.array([])
C_landing = fix_data2(C_landing, 'l, cos(lambda)^2 * (t/c)^2 * AR', fig_3)

C_l_landing = np.array([])
C_l_landing = fix_data2(C_l_landing, 'l, C_L_max', fig_3)

coef_3_l, cov_3_l = curve_fit(test_2nd, C_landing, C_l_landing)




# Figure 4 JT8D-9 Fuel Fraction Estimate for All Out Range

fig_4 = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 4')

R = np.array([])
R = fix_data2(R, 'Range', fig_4)

weight_frac = np.array([])
weight_frac = fix_data2(weight_frac, 'Wf / Wto', fig_4)

coef_4, cov_4 = curve_fit(test_3rd, R, weight_frac)

# Figure 5 Jet Aircraft Far Take-off Field Length with Engine Failure
fig_5 = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 5')

# 2 Engines
X2 = np.array([])
X2 = fix_data2(X2, '2, W/S * W/T * 1/Clmax', fig_5)

Y2 = np.array([])
Y2 = fix_data2(Y2, '2, TOFL', fig_5)

coef_5_2e, cov_5_2e = curve_fit(test_2nd, Y2, X2)


# 3 Engines
X3 = np.array([])
X3 = fix_data2(X3, '3, W/S * W/T * 1/Clmax', fig_5)

Y3 = np.array([])
Y3 = fix_data2(Y3, '3, TOFL', fig_5)

coef_5_3e, cov_5_3e = curve_fit(test_2nd, Y3, X3)

# 4 Engines



#Figure 6 incremental profile drag for high lift systems

fig_6 = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Figure 6')

# takeoff
cd_6_to = np.array([])
cd_6_to = fix_data2(cd_6_to, 'to, cd', fig_6)

cl_6_to = np.array([])
cl_6_to = fix_data2(cl_6_to, 'to, cl', fig_6)

coef_6_to, cov_6_to = curve_fit(test_2nd, cl_6_to, cd_6_to)


# landing
cd_6_l = np.array([])
cd_6_l = fix_data2(cd_6_l, 'l, cd', fig_6)

cl_6_l = np.array([])
cl_6_l = fix_data2(cl_6_l, 'l, cl', fig_6)

coef_6_l, cov_6_l = curve_fit(test_2nd, cl_6_l, cd_6_l)


#JT9D-7 Performance

mach = np.array([0,0.15,0.30,0.45])
thrust = np.array([45500,39120,34820,31750])
coef_JT9D, cov_JT9D = curve_fit(test_lin, mach, thrust)



#Shevel_3
fig_s3 =  pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Shevel_3')

rn_s3 = np.array([])
rn_s3 = fix_data2(rn_s3, 'RN', fig_s3)

c_f_s3 = np.array([])
c_f_s3 = fix_data2(c_f_s3, 'C_f', fig_s3)

coef_s3, cov_s3 = curve_fit(test_s3, rn_s3, c_f_s3)


#Shevel_1
fig_s1 =  pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Shevel_1')

tc_s1_35 = np.array([])
tc_s1_35 = fix_data2(tc_s1_35, '35, tc', fig_s1)

kappa_35 = np.array([])
kappa_35 = fix_data2(kappa_35, '35, K', fig_s1)

coef_s1_35, cov_s1_35 = curve_fit(test_2nd, tc_s1_35, kappa_35)

#Shevel_2
fig_s2 =  pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='Shevel_2')

l_d_array = np.array([])
l_d_array = fix_data2(l_d_array, 'L/D', fig_s2)

kappa_fuse_array = np.array([])
kappa_fuse_array = fix_data2(kappa_fuse_array, 'K', fig_s2)

coef_s2, cov_s2 = curve_fit(test_2nd, l_d_array, kappa_fuse_array)



#JT9D Thrust
fig_thrust =  pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\159 Digitized Plots.xlsx',sheet_name='JT9D_Thrust')

thrust_15_M = np.array([])
thrust_15_M = fix_data2(thrust_15_M, '15, M', fig_thrust)

thrust_15_T = np.array([])
thrust_15_T = fix_data2(thrust_15_T, '15, T', fig_thrust)

sfc_15_c = np.array([0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75])
sfc_15_M = np.array([0.044008,0.176616,0.312153,0.440115,0.549997,0.667388,0.775736,0.887076,0.983367])

sfc_25_c = np.array([0.45,0.5,0.55,0.6,0.65,0.7])
sfc_25_M = np.array([0.295317,0.421887,0.557711,0.705871,0.836535,0.968233])

sfc_35_c = np.array([0.6,0.65,0.68,0.7,0.75,0.8,0.9,0.95,1.05])
sfc_35_T = np.array([7000,5000,4000,3000,2500,23000,2000,1600,1000])

thrust_25_M = np.array([])
thrust_25_M = fix_data2(thrust_25_M, '25, M', fig_thrust)

thrust_25_T = np.array([])
thrust_25_T = fix_data2(thrust_25_T, '25, T', fig_thrust)

coef_thrust_15, cov_thrust_15 = curve_fit(test_2nd, thrust_15_M, thrust_15_T)
coef_thrust_25, cov_thrust_25 = curve_fit(test_2nd, thrust_25_M, thrust_25_T)

coef_sfc_15, cov_sfc_15 = curve_fit(test_lin, sfc_15_M, sfc_15_c)
coef_sfc_25, cov_sfc_25 = curve_fit(test_lin, sfc_25_M, sfc_25_c)
coef_sfc_35, cov_sfc_35 = curve_fit(test_2nd, sfc_35_T, sfc_35_c)

