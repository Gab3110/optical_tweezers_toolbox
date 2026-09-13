#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Description  : This script extracts relevant channels from .h5 files
               created by a lumicks C-trap and performs baseline and trap
               compliance corrections to the forces and trap position
               channels, respectively. It also computes the differential
               bead displacement and the diffF channel,
               which combines the info of displacement of the two beads and
               force1x and force2x, respectively, to produce a higher resolution channel.


Author       : Gabriel Jiménez-Avalos, PhD. student.
Affiliation  : T. C. Jenkins department of Biophysics. Johns Hopkins University.
Email        : gjimene5@jhu.edu
Version      : 4.0
Date         : 2025/04/06


'''

#%% IMPORT MODULES

import os, glob, re, sys
import numpy as np
import pandas as pd
import lumicks.pylake as lk

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules import preprocessing as pre

#%%

def gen_baseline_name (fec_id):
    # Getting baseline name
    underscore_ids = [x.start() for x in re.finditer("_", fec_id)]
    dte =  fec_id[0:underscore_ids[0]]
    bset = fec_id[underscore_ids[0]+1:underscore_ids[1]]
    bsl_pthname_current = dte + "_" + bset + "_"+ "fec00.h5"
    bsl_pthname_current_base = os.path.basename(bsl_pthname_current).replace(".h5","")

    return(bsl_pthname_current, bsl_pthname_current_base)

#%%

# Get a list of new path of the files
files = glob.glob("rawdat/*.h5")
folder = "260214_1xBiotBact1_1016-601-Bact2_998NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_1umBeads"
df = pd.read_csv(f"preprocess/{folder}_preprocessed.csv")
dfcalpars = pd.read_csv(f"preprocess/{folder}_calpars.csv")

#%%
#
k1_corrs = []
k2_corrs = []
fecs = []

for i in files:
    if "fec00" not in i:
        print(i)
        fec_file = lk.File(i)
        k1_corr, k2_corr = pre.estimate_stiffness_error(fec_file)
        k1_corrs.append(k1_corr)
        k2_corrs.append(k2_corr)
        fecs.append(i)

k1_corrs = np.array(k1_corrs)
k2_corrs = np.array(k2_corrs)

#%%

overall_corr = np.mean(k2_corrs[ (k2_corrs > 1.05) & (k2_corrs < 1.3)])
fecs = np.array(fecs)

#%%
corr_df = []

for i in files:
    if "fec00" not in i:
        tmpdf = df[df["fec_id"] == i]
        bsl = gen_baseline_name (i)[1]
        k1 = dfcalpars[dfcalpars.baseline_id == bsl]["kappa (pN/nm)"].to_list()[0]*overall_corr
        k2 = dfcalpars[dfcalpars.baseline_id == bsl]["kappa (pN/nm)"].to_list()[1]*overall_corr
        tmpdf.loc[:,"molext"] = (1000*tmpdf.trappos1x - tmpdf.force2x/k2 + tmpdf.force1x/k1)/1000
        corr_df.append(tmpdf)

corr_df = pd.concat(corr_df)
corr_df.to_csv("preprocess/" + folder + "_postprocessed_corrected.csv", index = False)




