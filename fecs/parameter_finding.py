import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
from tweezers_toolbox_modules.models import ini_eWLC
import numpy as np

#%%

df = pd.read_csv("preprocess/260601_SYT1_preprocessed.csv")
tmpdf = df[df.fec_id == "260601_03_01_fecS04"]

d, f = tmpdf.molext.to_numpy(), tmpdf.diffF.to_numpy()
dlf = CK_filtfilt(d, 25)
flf = CK_filtfilt(f, 25 )

#%%

pf = np.arange(0,40,0.1)
pdist = ini_eWLC(1, pf, [40,  4.262,1050], [1, 0.95639,1500])
pdist2 = ini_eWLC(1, pf, [40,  4.262,1050], [1, 0.80,1500])
pdist3 = ini_eWLC(1, pf, [40,  4.262, 1050], [1, 0.215,1500])


plt.xlim(3.8,5.2)
plt.plot(pdist, pf)
plt.plot(pdist2, pf)
plt.plot(pdist3, pf)


plt.plot(dlf+0.248,flf)



#%%