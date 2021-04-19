import numpy as np
from matplotlib import pyplot as plt

### Fixed aspect ratio curves, varying sweep ###
#2 aisles, 8 abreast, 2 engines

ar_sweep = np.array([10,15,20,25,30,35,40])

## non stop ##

#AR = 6
# Passanger Mile
ar6_pm = np.array([0.01751076177463328, 0.016889214532717356, 0.016466882152735077,
0.01598032490770467, 0.015687269674111003, ])
# Ton Mile
ar6_tm = np.array([0.12433676999739607, 0.11992341680035994, 0.11692460700166918,
0.113469762658258, 0.11138889709427931, ])
# Weight
ar6_w = np.array([668395, 641890, 624920, 605070, 593470, ])
