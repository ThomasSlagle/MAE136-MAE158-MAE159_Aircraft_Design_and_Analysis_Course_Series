import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler
import seaborn as sns

### Fixed aspect ratio curves, varying sweep ###
#2 aisles, 8 abreast, 2 engines

ar_array = np.arange(6,12,0.1)
#print(ar_array)
sweep_array = np.arange(10, 40, 1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

array_w_ar_onestop = np.load('onestop_w_array.npy')
array_doctm_ar_onestop = np.load('onestop_tm_array.npy')
array_docpm_ar_onestop = np.load('onestop_pm_array.npy')

array_w_ar_nonstop = np.load('nonstop_w_array.npy')
array_doctm_ar_nonstop = np.load('nonstop_tm_array.npy')
array_docpm_ar_nonstop = np.load('nonstop_pm_array.npy')

onestop_w_40_sweep = np.load('onestop_w_40_sweep.npy')
onestop_tm_40_sweep = np.load('onestop_tm_40_sweep.npy')
onestop_pm_40_sweep = np.load('onestop_pm_40_sweep.npy')

nonstop_w_40_sweep = np.load('nonstop_w_40_sweep.npy')
nonstop_tm_40_sweep = np.load('nonstop_tm_40_sweep.npy')
nonstop_pm_40_sweep = np.load('nonstop_pm_40_sweep.npy')

#print(array_doctm_ar_onestop[np.where(sweep_array == 31)].flatten()[np.where(ar_array == find_nearest(ar_array, 9.5))])
#print((array_doctm_ar_nonstop[np.where(sweep_array == 37)]).flatten()[np.where(ar_array == find_nearest(ar_array,7.9))])




#Weight v Sweep
fig1, (ax1, ax2) = plt.subplots(1,2, figsize = (12,9))
fig1.subplots_adjust(bottom = 0.15)

#for i in range(len(array_w_ar_nonstop)):
#       ax1.scatter(ar_array, array_w_ar_nonstop[i], label=('Sweep Angle: ' + str(10+i)))
ax1.scatter(ar_array, array_w_ar_nonstop[0], label=('Sweep Angle: ' + str(10)))
ax1.scatter(ar_array, array_w_ar_nonstop[5], label=('Sweep Angle: ' + str(15)))
ax1.scatter(ar_array, array_w_ar_nonstop[10], label=('Sweep Angle: ' + str(20)))
ax1.scatter(ar_array, array_w_ar_nonstop[15], label=('Sweep Angle: ' + str(25)))
ax1.scatter(ar_array, array_w_ar_nonstop[20], label=('Sweep Angle: ' + str(30)))
ax1.scatter(ar_array, array_w_ar_nonstop[25], label=('Sweep Angle: ' + str(35)))
ax1.scatter(ar_array, array_w_ar_nonstop[27], label=('Sweep Angle: ' + str(37)))
ax1.scatter(ar_array, nonstop_w_40_sweep, label='Sweep Angle: 40')
ax1.set_xlabel('Aspect Ratio', fontsize=14)
ax1.set_ylabel('Takeoff Weight (lbs)', fontsize=14)


#for i in range(len(array_w_ar_onestop)):
#       ax2.scatter(ar_array, array_w_ar_onestop[i], label=('Sweep Angle: ' + str(10+i)))
ax2.scatter(ar_array, array_w_ar_onestop[0], label=('Sweep Angle: ' + str(10)))
ax2.scatter(ar_array, array_w_ar_onestop[5], label=('Sweep Angle: ' + str(15)))
ax2.scatter(ar_array, array_w_ar_onestop[10], label=('Sweep Angle: ' + str(20)))
ax2.scatter(ar_array, array_w_ar_onestop[15], label=('Sweep Angle: ' + str(25)))
ax2.scatter(ar_array, array_w_ar_onestop[20], label=('Sweep Angle: ' + str(30)))
ax2.scatter(ar_array, array_w_ar_onestop[25], label=('Sweep Angle: ' + str(35)))
ax2.scatter(ar_array, array_w_ar_onestop[21], label=('Optimized Sweep Angle'))
ax2.scatter(ar_array, onestop_w_40_sweep, label='Sweep Angle: 40')
ax2.set_xlabel('Aspect Ratio', fontsize=14)

ax2.legend(loc = 'lower center', bbox_to_anchor=(0,0,1,1),bbox_transform = plt.gcf().transFigure, ncol = 3, fontsize=10)
ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)

plt.savefig(r'MAE 159\Other data\Images\Weight v Sweep Angle.png')
plt.close()




#DOC TM
fig2, (axx1, axx2) = plt.subplots(1,2, figsize = (12,9))
fig2.subplots_adjust(bottom = 0.15)

colors = sns.color_palette('tab20')
axx1.set_prop_cycle('color', colors)
axx2.set_prop_cycle('color', colors)

#for i in range(len(array_doctm_ar_nonstop)):
#       axx1.plot(ar_array, array_doctm_ar_nonstop[i], label=('Sweep Angle: ' + str(10+i)),marker='.',markersize = 10)

axx1.scatter(ar_array, array_doctm_ar_nonstop[0], label=('Sweep Angle: ' + str(10)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[5], label=('Sweep Angle: ' + str(15)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[10], label=('Sweep Angle: ' + str(20)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[15], label=('Sweep Angle: ' + str(25)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[20], label=('Sweep Angle: ' + str(30)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[25], label=('Sweep Angle: ' + str(35)))
axx1.scatter(ar_array, array_doctm_ar_nonstop[27], label=('Sweep Angle: ' + str(37)))
axx1.scatter(ar_array, nonstop_tm_40_sweep, label='Sweep Angle: 40')
axx1.set_xlabel('Aspect Ratio', fontsize=14)
axx1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)


#for i in range(len(array_doctm_ar_onestop)):
#       axx2.plot(ar_array, array_doctm_ar_onestop[i], label=('Sweep Angle: ' + str(10+i)),marker='.',markersize = 10)
#axx2.plot(ar_array, onestop_tm_40_sweep, label='Sweep Angle: 40',marker='.',markersize = 10)

axx2.scatter(ar_array, array_doctm_ar_onestop[0], label=('Sweep Angle: ' + str(10)))
axx2.scatter(ar_array, array_doctm_ar_onestop[5], label=('Sweep Angle: ' + str(15)))
axx2.scatter(ar_array, array_doctm_ar_onestop[10], label=('Sweep Angle: ' + str(20)))
axx2.scatter(ar_array, array_doctm_ar_onestop[15], label=('Sweep Angle: ' + str(25)))
axx2.scatter(ar_array, array_doctm_ar_onestop[20], label=('Sweep Angle: ' + str(30)))
axx2.scatter(ar_array, array_doctm_ar_onestop[25], label=('Sweep Angle: ' + str(35)))
axx2.scatter(ar_array, array_doctm_ar_onestop[21], label=('Optimized Sweep Angle'))
axx2.scatter(ar_array, onestop_tm_40_sweep, label='Sweep Angle: 40')

axx2.set_xlabel('Aspect Ratio', fontsize=14)

axx2.legend(loc = 'lower center', bbox_to_anchor=(0,0,1,1),bbox_transform = plt.gcf().transFigure, ncol = 3, fontsize=10)
axx1.set_title('Non-stop Aircraft', fontsize = 14)
axx2.set_title('One-stop Aircraft', fontsize = 14)

plt.savefig(r'MAE 159\Other data\Images\DOCTM v Sweep Angle.png')
#plt.show()
plt.close()
exit()


nonstop_w_conv = (array_w_ar_nonstop[np.where(sweep_array == 37)]).flatten()[np.where(ar_array == find_nearest(ar_array,7.9))]
nonstop_doctm_conv = (array_doctm_ar_nonstop[np.where(sweep_array == 37)]).flatten()[np.where(ar_array == find_nearest(ar_array,7.9))]

onestop_w_conv = array_w_ar_onestop[np.where(sweep_array == 31)].flatten()[np.where(ar_array == find_nearest(ar_array, 9.5))]
onestop_doctm_conv = array_doctm_ar_onestop[np.where(sweep_array == 31)].flatten()[np.where(ar_array == find_nearest(ar_array, 9.5))]


#Conventional v Supercritical DOCTM
nonstop_w_super = 453770
nonstop_doctm_super = 0.08682931402790262

onestop_w_super = 349255
onestop_doctm_super = 0.09653323477019972

#Composite, super
nonstop_w_com = 370555
nonstop_doctm_com = 0.07371624416117747

onestop_w_com = 291090
onestop_doctm_com = 0.08350397612319056

#Composite, conv
nonstop_w_com_c = 428810
nonstop_doctm_com_c = 0.08397394536007215

onestop_w_com_c = 316765
onestop_doctm_com_c = 0.09003464500232948


fig3 = fig3, (ax1, ax2) = plt.subplots(1,2, figsize = (8,4))

ax1.scatter([1,2,3,4],[nonstop_doctm_conv, nonstop_doctm_super,nonstop_doctm_com_c,nonstop_doctm_com])
ax2.scatter([1,2,3,4],[onestop_doctm_conv, onestop_doctm_super,onestop_doctm_com_c,onestop_doctm_com],color = '#ff7f0e')

ax1.annotate('Conventional Airfoil\nAluminum Airframe',xy = (1+0.1,nonstop_doctm_conv),fontsize=7.5)
ax1.annotate('Supercritical Airfoil\nAluminum Airframe',xy = (1+0.3,nonstop_doctm_super+0.0025),fontsize=7.5)
ax1.annotate('Conventional Airfoil\nComposite Airframe',xy = (3-0.2,nonstop_doctm_com_c+0.003),fontsize=7.5)
ax1.annotate('Supercritial Airfoil\nComposite Airframe',xy = (3-0.35,nonstop_doctm_com),fontsize=7.5)

ax2.annotate('Conventional Airfoil\nAluminum Airframe',xy = (1+0.1,onestop_doctm_conv),fontsize=7.5)
ax2.annotate('Supercritical Airfoil\nAluminum Airframe',xy = (1+0.3,onestop_doctm_super+0.0025),fontsize=7.5)
ax2.annotate('Conventional Airfoil\nComposite Airframe',xy = (3-0.2,onestop_doctm_com_c+0.003),fontsize=7.5)
ax2.annotate('Supercritial Airfoil\nComposite Airframe',xy = (3-0.35,onestop_doctm_com),fontsize=7.5)

ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)
ax1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
plt.savefig(r'MAE 159\Other data\Images\tech.png')
plt.close()

fig4 = fig4, (ax1, ax2) = plt.subplots(1,2, figsize = (9.2,4))

ax1.scatter([1,2,3,4],[nonstop_w_conv, nonstop_w_super,nonstop_w_com_c,nonstop_w_com])
ax2.scatter([1,2,3,4],[onestop_w_conv, onestop_w_super,onestop_w_com_c,onestop_w_com],color = '#ff7f0e')

ax1.annotate('Conventional Airfoil\nAluminum Airframe',xy = (1+0.1,nonstop_w_conv-10000),fontsize=7.5)
ax1.annotate('Supercritical Airfoil\nAluminum Airframe',xy = (1+0.3,nonstop_w_super+10000),fontsize=7.5)
ax1.annotate('Conventional Airfoil\nComposite Airframe',xy = (3-0.1,nonstop_w_com_c+10000),fontsize=7.5)
ax1.annotate('Supercritial Airfoil\nComposite Airframe',xy = (3-0.25,nonstop_w_com),fontsize=7.5)

ax2.annotate('Conventional Airfoil\nAluminum Airframe',xy = (1+0.1,onestop_w_conv-10000),fontsize=7.5)
ax2.annotate('Supercritical Airfoil\nAluminum Airframe',xy = (1+0.3,onestop_w_super+5000),fontsize=7.5)
ax2.annotate('Conventional Airfoil\nComposite Airframe',xy = (3-0.1,onestop_w_com_c+5000),fontsize=7.5)
ax2.annotate('Supercritial Airfoil\nComposite Airframe',xy = (3-0.25,onestop_w_com),fontsize=7.5)

ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)
ax1.set_ylabel('Takeoff Weight (Pounds)', fontsize=14)
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
plt.savefig(r'MAE 159\Other data\Images\tech_weight.png')
plt.close()


### Seating Config ###
nonstop_doctm_a2_b6 = 0.07562530010786023
onestop_doctm_a2_b6 = 0.08548894563176762

nonstop_doctm_a2_b8 = 0.07602524742588644
onestop_doctm_a2_b8 = 0.08587211625204444

nonstop_doctm_a2_b9 = 0.07682934206665448
onestop_doctm_a2_b9 = 0.08676861981146972

fig5 = fig5, (ax1, ax2) = plt.subplots(1,2, figsize = (8,4))

ax1.scatter([1,2,3,4],[nonstop_doctm_com, nonstop_doctm_a2_b6,nonstop_doctm_a2_b8,nonstop_doctm_a2_b9])
ax2.scatter([1,2,3,4],[onestop_doctm_com, onestop_doctm_a2_b6,onestop_doctm_a2_b8,onestop_doctm_a2_b9],color = '#ff7f0e')

ax1.annotate('1 Asile\n6 Abreast',xy = (1+0.1,nonstop_doctm_com),fontsize=7.5)
ax1.annotate('2 Asiles\n6 Abreast',xy = (2+0.1,nonstop_doctm_a2_b6),fontsize=7.5)
ax1.annotate('2 Asiles\n8 Abreast',xy = (3+0.1,nonstop_doctm_a2_b8-0.0005),fontsize=7.5)
ax1.annotate('2 Asiles\n9 Abreast',xy = (3+0.3,nonstop_doctm_a2_b9),fontsize=7.5)

ax2.annotate('1 Asile\n6 Abreast',xy = (1+0.1,onestop_doctm_com),fontsize=7.5)
ax2.annotate('2 Asiles\n6 Abreast',xy = (2+0.1,onestop_doctm_a2_b6),fontsize=7.5)
ax2.annotate('2 Asiles\n8 Abreast',xy = (3+0.1,onestop_doctm_a2_b8-0.0005),fontsize=7.5)
ax2.annotate('2 Asiles\n9 Abreast',xy = (3+0.3,onestop_doctm_a2_b9),fontsize=7.5)

ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)
ax1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)

ax1.set_ylim([0.072,0.080])
ax2.set_ylim([0.080,0.094])

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
plt.savefig(r'MAE 159\Other data\Images\seating.png')
plt.close()

#Engine Count

nonstop_doctm_4e = 0.07116377540140915
onestop_doctm_4e = 0.07739057954920024

nonstop_doctm_3e = 0.06863654876850053
onestop_doctm_3e = 0.07894583050396861

fig6 = fig6, (ax1, ax2) = plt.subplots(1,2, figsize = (8,4))

ax1.scatter([1,2,3],[nonstop_doctm_com,nonstop_doctm_3e,nonstop_doctm_4e])
ax2.scatter([1,2,3],[onestop_doctm_com,onestop_doctm_3e,onestop_doctm_4e],color = '#ff7f0e')

ax1.annotate('2 Engines',xy = (1+0.1,nonstop_doctm_com),fontsize=7.5)
ax1.annotate('3 Engines',xy = (2+0.1,nonstop_doctm_3e),fontsize=7.5)
ax1.annotate('4 Engines',xy = (2+0.5,nonstop_doctm_4e-0.0005),fontsize=7.5)


ax2.annotate('2 Engines',xy = (1+0.1,onestop_doctm_com),fontsize=7.5)
ax2.annotate('3 Engines',xy = (2+0.1,onestop_doctm_3e),fontsize=7.5)
ax2.annotate('4 Engines',xy = (2+0.5,onestop_doctm_4e-0.0005),fontsize=7.5)

ax1.set_title('Non-stop Aircraft', fontsize = 14)
ax2.set_title('One-stop Aircraft', fontsize = 14)
ax1.set_ylabel('DOC per Ton per Mile ($)', fontsize=14)

#ax1.set_ylim([0.072,0.080])
#ax2.set_ylim([0.080,0.094])

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
plt.savefig(r'MAE 159\Other data\Images\engines.png')
plt.close()





