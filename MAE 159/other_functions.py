import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import digitized_plots as dgp

#class other_functions:
df = pd.read_excel(r'C:\Users\Tommy Slagle\Documents\MAE136-MAE158-MAE159_Aircraft_Design_and_Analysis_Course_Series\MAE 159\Other data\std_atm_table.xlsx',sheet_name='Sheet1')
#print(df)

altitude_array = np.array([])
altitude_array = dgp.fix_data2(altitude_array, 'alt', df)
#print(altitude_array)

delta_array = np.array([])
delta_array = dgp.fix_data2(delta_array, 'delta', df)
#print(delta_array)

speed_sound = np.array([])
speed_sound = dgp.fix_data2(speed_sound, 'a', df)

k_visc = np.array([])
k_visc = dgp.fix_data2(k_visc, 'k.visc', df)


