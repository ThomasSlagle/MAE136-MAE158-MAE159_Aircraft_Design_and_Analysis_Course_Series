import numpy as np
from matplotlib import pyplot as plt
import digitized_plots as dgp
#import other_functions as of

# Design Mission Parameters
num_p = 0
W_cargo = 0
air_range = 0
l_takeoff = 0
v_approach = 0
M_cruise = 0
h_cruise_i = 0

mission_type = 0

while (mission_type != 1 or mission_type != 2 or mission_type != 3):
    mission_type = 3#int(input("Enter your mission type (1 for Non-Stop, 2 for One-stop, 3 for DC-10 Sample Code): "))

    if (mission_type == 1):
        num_p = 225 #Class 3
        W_cargo = 6000 #lbs
        air_range = 7400 #nautical miles
        l_takeoff = 10500 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.85
        h_cruise_i = 35000 #ft
        #print('hehe')
        break

    elif (mission_type == 2):
        num_p = 225 #Class 3
        W_cargo = 6000 #lbs
        air_range = 7400 #nautical miles
        l_takeoff = 10500 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.85
        h_cruise_i = 35000 #ft
        #print('hehe2')
        break

    elif (mission_type == 3):
        num_p = 275 #Class 3
        W_cargo = 12000 #lbs
        air_range = 6000 #nautical miles
        l_takeoff = 9000 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.82
        h_cruise_i = 35000 #ft
        break

    else:
        print('Error! Retype!')
print('exited loop; mission type is ', mission_type)

# C_L loop

a = 576.48
x = 0.75
sigma = 0.953
delta = 0.2360

ar = 8
C_l = 0.5
sweep = 35

#figure 2 to get delta_M_div
delta_M_div = dgp.order_2nd(C_l, dgp.coef_2_c)

M_div = (M_cruise + 0.004) - delta_M_div

#figure 1a for t/c
tc = dgp.linear(M_div, dgp.coef_1a_35)

C = np.cos(np.radians(sweep))**2*tc**2*ar
#fig 3 to get CL maxes
c_l_takeoff = dgp.order_2nd(C, dgp.coef_3_t)
c_l_landing = dgp.order_2nd(C, dgp.coef_3_l)

ws_landing = (v_approach/1.3)**2*((sigma*c_l_landing)/296)
R_allout = air_range + 200 + 0.75*M_cruise*(a)#*0.592484)
#fig 4 for wf/wto
wf_wto = dgp.order_3rd(R_allout, dgp.coef_4)
#JT9D to JT8D
jt9d = wf_wto * 0.61/0.78 + .015#1.2307

ws_takeoff = ws_landing / (1-x*jt9d)#wf_wto)
ws_initialcruise = 0.965 * ws_takeoff
c_l_initialcruise = ws_initialcruise/(1481*delta*M_cruise**2)
#print(ws_)
#print(tc)
#print(M_div)
#print(dgp.coef_1a_35)
print(c_l_initialcruise)

#TOFL

engineequation = dgp.order_2nd(l_takeoff, dgp.coef_5_3e)

wt_7vstall = engineequation/ws_initialcruise*sigma*c_l_takeoff
v_lo = 1.2 * (296*ws_takeoff/(sigma*c_l_takeoff))**(1/2)
M_lo = v_lo/661/(sigma**(1/2))*0.7
wt = wt_7vstall * 37200/45500

print(wt)
