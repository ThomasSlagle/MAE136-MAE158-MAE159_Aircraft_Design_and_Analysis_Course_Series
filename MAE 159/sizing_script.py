import numpy as np
from matplotlib import pyplot as plt
import digitized_plots as dgp
import other_functions as of

# Design Mission Parameters
num_p = 0
W_cargo = 0
air_range = 0
l_takeoff = 0
v_approach = 0
M_cruise = 0
h_cruise_i = 0

mission_type = 0

#Mission selection loop
while (mission_type != 1 or mission_type != 2 or mission_type != 3 or mission_type != 4):
    mission_type = 3#int(input("Enter your mission type (1 for Non-Stop, 2 for One-stop, 3 for DC-10 Sample Code): "))

    if (mission_type == 1):
        #non Stop
        num_p = 225 #Class 3
        W_cargo = 6000 #lbs
        air_range = 7400 #nautical miles
        l_takeoff = 10500 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.85
        h_cruise_i = 35000. #ft
        fuel_remaning = 0.35
        num_e = 2
        #print('hehe')
        break

    elif (mission_type == 2):
        #One-stop
        num_p = 225 #Class 2
        W_cargo = 3000 #lbs
        air_range = 3700 #nautical miles
        l_takeoff = 6000 #feet
        v_approach = 130 #kts       #PL +35% max fuel
        M_cruise = 0.80
        h_cruise_i = 35000. #ft
        fuel_remaning = 0
        num_e = 2
        #print('hehe2')
        break

    elif (mission_type == 3):
        num_p = 275 #Class 3
        W_cargo = 12000 #lbs
        air_range = 6000 #nautical miles
        l_takeoff = 9000 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.82
        h_cruise_i = 35000. #ft
        fuel_remaning = 0.25
        num_e = 3
        break

    elif (mission_type == 4):
        num_p = 214 #Class 3
        W_cargo = 3000 #lbs
        air_range = 3900 #nautical miles
        l_takeoff = 6300 #feet
        v_approach = 140 #kts       #PL +35% max fuel
        M_cruise = 0.80
        h_cruise_i = 35000. #ft
        fuel_remaning = 0.35
        num_e = 2
        break

    else:
        print('Error! Retype!')
print('exited loop; mission type is ', mission_type)

# General Parameters
a = 576.48 #kts
a_20000 = of.speed_sound[np.where(of.altitude_array == 20000.)[0][0]]*0.592484
#print(a_20000)
x = 1 - fuel_remaning
sigma_0 = 0.953
sigma_1000 = 0.925
sigma_10000 = 0.764
sigma_20000 = 0.533


delta_35000 = 0.2360

airfoil_type = 'conventional'
ar = 8
sweep = 35 # degrees
C_l = 0.5
taper = 0.35
n = 1.5*2.5 #ultimate load factor
k_f = 11.5

#fuselage mounted (1) wing mounted (0)
fuse_engine = 0

num_flightcrew = 2
num_stew = np.ceil(num_p/50)
num_abreast = 8
num_aisles = 2
jt8d = 1

adjust = 0
R_total = 0
R_allout = 100
count_adj = 0

adj_w = 0
T_r_jt9d_2 = 10001

c_l_initialcruise = 0
count_cl=0
count_w=0


#Thrust on top loop
while (T_r_jt9d_2 > 10000):
    #Range Loop
    while ((R_allout - R_total) > 50):

        # C_L loop
        while (np.abs((C_l - c_l_initialcruise)/C_l) > 0.005):
            #figure 2 to get delta_M_div
            if (airfoil_type == 'conventional'):
                delta_M_div = dgp.order_2nd(C_l, dgp.coef_2_c)
            else:
                delta_M_div = dgp.order_3rd(C_l, dgp.coef_2_s)

            M_div = (M_cruise + 0.004) - delta_M_div

            if (airfoil_type == 'conventional'):
                #figure 1a for t/c
                #tc = dgp.linear(M_div, dgp.coef_1a_35)
                if (sweep == 0):
                    tc = dgp.linear(M_div, dgp.coef_1a_0)
                elif (sweep < 10):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_0)], [M_div,dgp.linear(M_div, dgp.coef_1a_10)])
                elif(sweep == 10):
                    tc = dgp.linear(M_div, dgp.coef_1a_10)
                elif (sweep > 10 and sweep < 15):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_10)], [M_div,dgp.linear(M_div, dgp.coef_1a_15)])
                elif(sweep == 15):
                    tc = dgp.linear(M_div, dgp.coef_1a_15)
                elif (sweep > 15 and sweep < 20):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_15)], [M_div,dgp.linear(M_div, dgp.coef_1a_20)])
                elif(sweep == 20):
                    tc = dgp.linear(M_div, dgp.coef_1a_20)
                elif (sweep > 20 and sweep < 25):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_20)], [M_div,dgp.linear(M_div, dgp.coef_1a_25)])
                elif(sweep == 25):
                    tc = dgp.linear(M_div, dgp.coef_1a_25)
                elif (sweep > 25 and sweep < 30):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_25)], [M_div,dgp.linear(M_div, dgp.coef_1a_30)])
                elif(sweep == 30):
                    tc = dgp.linear(M_div, dgp.coef_1a_30)
                elif (sweep > 30 and sweep < 35):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_30)], [M_div,dgp.linear(M_div, dgp.coef_1a_35)])
                elif(sweep == 35):
                    tc = dgp.linear(M_div, dgp.coef_1a_35)
                elif (sweep > 35 and sweep < 40):
                    tc = np.interp(M_div, [M_div,dgp.linear(M_div, dgp.coef_1a_35)], [M_div,dgp.linear(M_div, dgp.coef_1a_40)])
                elif(sweep == 40):
                    tc = dgp.linear(M_div, dgp.coef_1a_40)

                else:
                    print('Sweep value is bad, please check!')
                    exit()
            else:
                if (sweep == 0):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_0)
                elif (sweep > 0 and sweep < 5):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_0)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_5)])
                elif (sweep == 5):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_5)
                elif (sweep > 5 and sweep < 10):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_5)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_10)])
                elif(sweep == 10):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_10)
                elif (sweep > 10 and sweep < 15):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_10)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_15)])
                elif(sweep == 15):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_15)
                elif (sweep > 15 and sweep < 20):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_15)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_20)])
                elif(sweep == 20):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_20)
                elif (sweep > 20 and sweep < 25):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_20)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_25)])
                elif(sweep == 25):
                    tc = dgp.order_2nd(M_div, dgp.coef_1a_25)
                elif (sweep > 25 and sweep < 30):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_25)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_30)])
                elif(sweep == 30):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_30)
                elif (sweep > 30 and sweep < 35):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_30)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_35)])
                elif(sweep == 35):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_35)
                elif (sweep > 35 and sweep < 40):
                    tc = np.interp(M_div, [M_div,dgp.order_2nd(M_div, dgp.coef_1b_35)], [M_div,dgp.order_2nd(M_div, dgp.coef_1b_40)])
                elif(sweep == 40):
                    tc = dgp.order_2nd(M_div, dgp.coef_1b_40)
                else:
                    print('Sweep value is bad, please check!')
                    exit()


            C = np.cos(np.radians(sweep))**2*tc**2*ar
            #fig 3 to get CL maxes
            c_l_takeoff = dgp.order_2nd(C, dgp.coef_3_t)
            c_l_landing = dgp.order_2nd(C, dgp.coef_3_l)

            ws_landing = (v_approach/1.3)**2*((sigma_0*c_l_landing)/296)
            R_allout = air_range + 200 + 0.75*M_cruise*(a)#*0.592484)

            if (jt8d == 1):
                #fig 4 for wf/wto
                #JT9D to JT8D
                wf_wto = dgp.order_3rd(R_allout, dgp.coef_4) * 0.61/0.78 + 0.04*dgp.order_3rd(R_allout, dgp.coef_4) * 0.61/0.78 + adjust#1.2307
            else:
                wf_wto = dgp.order_3rd(R_allout, dgp.coef_4) + adjust



            ws_takeoff = ws_landing / (1-x*wf_wto)#wf_wto)
            ws_initialcruise = 0.965 * ws_takeoff
            c_l_initialcruise = ws_initialcruise/(1481*delta_35000*M_cruise**2)
            if (c_l_initialcruise > C_l):
                C_l = C_l + 0.001
            else:
                C_l = C_l - 0.001
            count_cl = count_cl+1
            if(count_cl > 1000):
                print('(C_l loop) too many itterations, STOPPING!')
                exit()

        #print(ws_takeoff)
        #print(tc)
        #print(M_div)

        #print(dgp.coef_1a_35)
        print('C_L is: '+ str(C_l))
        print('C_L_IC is: '+ str(c_l_initialcruise))
        #print(count)

        #TOFL
        if (num_e == 2):
            engineequation = dgp.order_2nd(l_takeoff, dgp.coef_5_2e)
            #print(engineequation)
        elif(num_e == 3):
            engineequation = dgp.order_2nd(l_takeoff, dgp.coef_5_3e)
            #print(dgp.order_2nd(l_takeoff, dgp.coef_5_2e))
            #print(dgp.order_2nd(l_takeoff, dgp.coef_5_3e))
        else:
            print('engine count error. Please check!')
            exit()

        wt_7vstall = engineequation/ws_takeoff*sigma_0*c_l_takeoff
        v_lo = 1.2 * (296*ws_takeoff/(sigma_0*c_l_takeoff))**(1/2)
        #print(ws_takeoff)
        #print(v_lo)
        M_lo = v_lo/661/(sigma_0**(1/2))*0.7
        #print(M_lo)
        #print(wt_7vstall)
        wt = wt_7vstall * dgp.linear(M_lo, dgp.coef_JT9D)/dgp.linear(0, dgp.coef_JT9D) + adj_w

        print('W/T is: ' + str(wt))




        # Weight
        if (fuse_engine == 1):
            Kw = 1.03
            Kts = 0.25
        else:
            Kw = 1.0
            Kts = 0.17

        W_w = (0.00945*ar**0.8*(1+taper)**0.25*Kw*n**0.5)/((tc)**0.4*np.cos(np.radians(sweep))*ws_takeoff**0.695)
        # ^ tc + 0.03


        l_f = 3.76 * num_p/num_abreast + 33.2
        dia_f = 1.75*num_abreast + 1.58*num_aisles + 1.0
        W_f = 0.6727 * k_f * l_f**0.6 * dia_f**0.72 * n**0.3

        W_lg = 0.040
        W_np = 0.0555/wt
        W_ts = Kts*W_w
        W_pp = 1/(3.58*wt)

        W_fuel = 1.0275*wf_wto
        W_payload = 215*num_p + W_cargo
        W_fixedequip = 0.035
        W_fe_cts = 132*num_p + 300*num_e + 260*num_flightcrew + 170*num_stew

        #w_to = np.linspace(100000,750000, 100000)
        #W = (W_w+W_ts)*w_to**1.195 + W_f*w_to**0.235 + (W_lg+W_np+W_pp+W_fuel+W_fixedequip-1)*w_to + (W_payload + W_fe_cts)
        #plt.plot(w_to,W)
        #plt.show()
        w_to = 100000
        W = 1000
        while(W > 100):
            if ((W_lg+W_np+W_pp+W_fuel+W_fixedequip-1) >= 0):
                print('Aircraft weight fraction is too high, adjust value is too high, retry!')
                exit()

            W = (W_w+W_ts)*w_to**1.195 + W_f*w_to**0.235 + (W_lg+W_np+W_pp+W_fuel+W_fixedequip-1)*w_to + W_payload + W_fe_cts
            w_to = w_to + 5
            count_w = count_w+1
            #print(count)
            if(count_w > 200000):
                print('(weight loop) too many itterations, STOPPING!')
                print(w_to)
                exit()
        print('Takeoff Weight is: '+ str(w_to) + ' lbs')
        #print(W)

        S = w_to/(ws_takeoff)
        b = (ar*S)**(1/2)
        chord_average = S/b
        T = w_to/wt
        T_e = T/num_e



        #Drag Calculation
        #print(of.altitude_array)
        h_drag = 30000.
        M_05 = 0.5
        altitude_ = np.where(of.altitude_array == h_drag)
        altitude_index = altitude_[0][0]
        #print(altitude_index)
        Rn_foot = M_05 * of.speed_sound[altitude_index]/of.k_visc[altitude_index] # /ft
        #print(Rn_foot)
        def Rn (x):
            return Rn_foot*x
        Rn_wing = Rn(chord_average)
        c_f_wing = dgp.order_s3(Rn_wing, dgp.coef_s3)
        #print(c_f_wing)
        Rn_fuse = Rn(l_f)
        c_f_fuse = dgp.order_s3(Rn_fuse, dgp.coef_s3)
        #print(c_f_fuse)
        S_wet_w = 2 * (S - dia_f*30)* 1.02 #might need to do 20*30
        Z = ((2 - M_05**2)*np.cos(np.radians(sweep)))/np.sqrt(1-M_05**2*np.cos(np.radians(sweep))**2)
        kappa_wing = 1 + Z*tc + 100*tc**4 #dgp.order_2nd(tc, dgp.coef_s1_35)
        f_wing = kappa_wing*c_f_wing*S_wet_w

        kappa_fuse = dgp.order_2nd((l_f/dia_f), dgp.coef_s2)
        S_wet_fuse = 0.9*np.pi*dia_f*l_f
        f_fuse = kappa_fuse*S_wet_fuse*c_f_fuse

        # kappa for tail from tail?
        f_tail = 0.38*f_wing

        S_wet_nac = 2.1*(T_e)**(1/2)*num_e

        # kappa for nacelle from fuse?
        f_nac = 1.25*c_f_wing*S_wet_nac
        f_pylon = 0.20*f_nac

        f_total = (f_pylon+f_nac+f_tail+f_fuse+f_wing)*1.06
        c_d_0 = f_total/S

        e = 1/(1.035+0.38*c_d_0*np.pi*ar)

        #Climb
        h_average_climb = 20/35*h_cruise_i
        #print(h_average_climb)
        W = (1+0.965)/2*w_to
        V_climb = 1.3 * 12.9/(f_total*e)**(1/4) * (W/(sigma_20000*b))**(1/2) # sigma needs to change and w
        M_climb = V_climb/a_20000
        #print(M_climb)

        T_r_cl = sigma_20000*f_total*V_climb**2/296 + 94.1/(sigma_20000*e)*(W/b)**2*1/V_climb**2

        T_a_15 = dgp.order_2nd(M_climb, dgp.coef_thrust_15)
        sfc_15 = dgp.linear(M_climb, dgp.coef_sfc_15)
        T_a_25 = dgp.order_2nd(M_climb, dgp.coef_thrust_25)
        sfc_25 = dgp.linear(M_climb, dgp.coef_sfc_25)

        T_a_engine = (T_a_15 + T_a_25)/2
        sfc = (sfc_15 + sfc_25)/2
        T_a = T_e/dgp.linear(0, dgp.coef_JT9D) * T_a_engine
        #print(T_a)

        roc = 101 * (num_e*T_a - T_r_cl)/W * V_climb
        #print(roc)

        time_climb = h_cruise_i/roc
        range_climb = V_climb * time_climb/60
        w_f_climb = num_e * T_a * sfc * time_climb/60

        #Range
        w_0 = w_to - w_f_climb
        w_1 = (1 - wf_wto)*w_to
        c_l_avg = ((w_0 + w_1)/(2*S))/(1481*delta_35000*M_cruise**2)
        c_d_i = c_l_avg**2/(np.pi*ar*e)
        c_d_tot = c_d_0 + c_d_i + 0.0010

        l_d_cruise = c_l_avg/c_d_tot

        T_r_cruise = (w_0 + w_1)/2/l_d_cruise
        T_r_JT9D = T_r_cruise * dgp.linear(0, dgp.coef_JT9D)/T_e/num_e
        sfc_35 = dgp.order_2nd(T_r_JT9D, dgp.coef_sfc_35)
        range_cruise = (M_cruise*973.14/1.68781)/sfc_35 * l_d_cruise * np.log(w_0/w_1)
        R_total = range_climb + range_cruise

        if (R_total < R_allout):
            adjust = adjust + 0.0001
        else:
            adjust = adjust - 0.0001

        count_adj = count_adj + 1
        #reset
        count_cl = 0
        count_w = 0
        c_l_initialcruise = 0
        #print(R_total)
        #print(R_allout)
        #print(adjust)
        #print(wf_wto)
        print('\n')
        if (count_adj > 1000):
            print('adjustment count too high')
            exit()
        #print(w_0)

    print('Total actual range is: ' + str(R_total))
    print('Theoretical range is: ' + str(R_allout))



    #Thrust on top of climb
    c_l_ic_2 = w_0/S/(1481*delta_35000*M_cruise**2)
    c_d_i_2 = c_l_ic_2**2/(np.pi*ar*e)
    c_d_tot_2 = c_d_0 + c_d_i_2 + 0.0010
    l_d_2 = c_l_ic_2/c_d_tot_2
    T_req_toc = w_0/l_d_2/num_e
    T_r_jt9d_2 = T_req_toc*dgp.linear(0, dgp.coef_JT9D)/T_e
    #print(T_req_toc)
    print(T_r_jt9d_2)
    if(T_r_jt9d_2 > 10000):
        print('Required thrust to maintain top of climb is too high')
        adj_w = adj_w + 0.1


#Direct Operating Cost
D = air_range*1.15078
t_gm = D/(11.866 + 0.040669*D)/60#0.25*ar
t_cl = 0.18*ar
t_d = 0
t_am = 0.10
t_cr = (D*1.02 + 20 - range_climb*1.15078)/(M_cruise*973.14/1.68781 * 1.15)

V_block = D / (t_gm + time_climb/60 + 0.10 + t_cr)
time_block = (t_gm + time_climb/60 + 0.10 + t_cr)

#block fuel
F_block = w_f_climb + (T_r_cruise * sfc_35 * (t_cr+0.10))

#flight cost
passanger = ((205*num_p + 50*num_p)+W_cargo)/2000
dollar_hr = 17.849 * M_cruise*973.14/1.68781 * (w_to/10**5)**0.3 + 40.83
ctm_cr = dollar_hr / (V_block*passanger)
#fuel cost
ctm_fuel = 1.02*F_block*0.0438 + num_e*2.15*time_block*0.135/(D*passanger)

#Insurance cost
Wa = w_to*(1-wf_wto)-(215*num_p+W_cargo)-W_pp*w_to
Ca = 2.4*10**6 + 87.5*Wa
Ce = 590000 + 16*T_e
Ct = num_e*Ce + Ca
U = 630 + 4000/(1 + 1/(time_block+0.50))
ctm_hull = 0.01*Ct/(U*V_block*passanger)

#Maintanence
kfh = 4.9169*np.log10(Wa/1000)-6.425
kfca = 0.21256*np.log10(Wa/1000)**3.7375
T_f = time_block - t_gm
ctm_directEq = 8.60*(kfh*T_f)/(V_block*(time_block*passanger))

#Airframe Material
cfha = 1.5994*(Ca/10**6) + 3.4263
cfca = 1.9229*(Ca/10**6) + 2.2504
ctm_airframe = (cfha*T_f+cfca)/(V_block*time_block*passanger)

#Engine Labor
kfhe = num_e*(T_e/10**3)/(0.82715*(T_e/10**3)+13.639)
kfce = 0.20*num_e
ctm_engine = 8.60 * (kfhe*T_f+kfce)/(V_block*time_block*passanger)

#Material Cost
cfhe = (28.2353*(Ce/10**6)-6.5176)*num_e
cfce = (3.6698*(Ce/10**6)+1.3685)*num_e
ctm_material = (cfhe*T_f+cfce)/(V_block*time_block*passanger)

#total maintanence
ctm_total = 2*(ctm_directEq+ctm_airframe+ctm_engine+ctm_material)

#depreciation
ctm_dep = 1/(V_block*passanger) * (Ct + 0.06*(Ct-num_e*Ce)+0.3*num_e*Ce)/(15*U)

doc = ctm_cr + ctm_fuel + ctm_hull + ctm_total + ctm_dep #per ton mile

print(doc*passanger/num_p) # per passangar mile
