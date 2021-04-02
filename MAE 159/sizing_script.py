import numpy as np
from matplotlib import pyplot as plt
import digitized_plots as dgp

# Design Mission Parameters
num_p = 0
W_cargo = 0
air_range = 0
l_takeoff = 0
v_approach = 0
M_cruise = 0
h_cruise_i = 0

mission_type = 0

while (mission_type != 1 or mission_type != 2):
    mission_type = int(input("Enter your mission type (1 for Non-Stop or 2 for One-stop): "))

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

    else:
        print('Error! Retype!')
print('exited loop')

