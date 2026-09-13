#%% CLEAN SESSION BEFORE START

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

#%% IMPORT MODULES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, dill,re
from datetime import datetime

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
from tweezers_toolbox_modules import postprocessing as post
from tweezers_toolbox_modules import fec_plotting as fplt

#%% DEFINE FUNCTIONS

def get_fd_from_df (fec_id, df, downfact):
    fec_dict = {}
    tmpdf = df[df["fec_id"] == fec_id]
    fec_dict["t"] = tmpdf.relative_time.to_numpy()
    fec_dict["d"] =  tmpdf.molext.to_numpy()
    fec_dict["f"] = tmpdf.diffF.to_numpy()
    fec_dict["dlf"] = CK_filtfilt(tmpdf.molext.to_numpy(), downfact)
    fec_dict["flf"] = CK_filtfilt(tmpdf.diffF.to_numpy(), downfact)
    fec_dict["curve_type"] = tmpdf.curve_type.to_list()[0]
    return(fec_dict)

def get_trace_base (trace):
    underscore_indexes = [x.start() for x in re.finditer("_", trace)]
    trace_base = trace[:underscore_indexes[-1]]
    return(trace_base)

#%% DEFINING GENERAL VARIABLES

# Define folder variable as the name of the folder containing the data (expt name)
folder = "260430_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_20nMChd1_1mMATP_1umBeads_b84_drift"

# Define a reference eWLC/WLC model. This will be used to align the fecs.
# You can define multiple models in series by
# defining multiple lists within the tuple. Just follow [PersistenceLen, ContourLen, StModulus]
# or [PersistenceLen, ContourLen]. If you omit StModulus the model will be a WLC.
# All models generated are from Marko-Siggia.
# NOTE: DEFINE THE UNFOLDED WLC UNLESS YOU KNOW THE SIZE OF THE MOLECULE BEFORE UNFOLDING.
WLCp = ([50,  0.78506],[0.65,0.01296])
downfact = 25

# True to remove the section of the trace after rupture and to reduce the end-state
# force below a particular value.
autotrim = True
maxforce = 35
align = True

# Indicate if the alignment to the reference worm like chain is to the end state
# (fully unfolded arm for example)
end_alignment = True

# MODIFY THIS NUMBER TO MATCH THE WLCp TUPLE INDEX OF THE WLC THAT WILL CHANGE.
# For example, say WLCp = ([Lp_DNA, Lc_DNA, St_DNA],[Lp_protein, Lc_protein])
# and you are unfolding protein. Then you should select var_component = 1.
var_component = 0

# use corrected file
use_drift_correction_file = False

#%% CREATE NECESSARY SIUBFOLDERS

try:
    os.makedirs("postprocess/postprocess_pics")
except:
    pass

#%% LOADING PREPROCESSED DATA

if use_drift_correction_file:
    df = pd.read_csv("preprocess/" + folder + "_preprocessed_corrected.csv")
else:
    df = pd.read_csv("preprocess/" + folder + "_preprocessed.csv")
    
df_preprocessed_discarded = pd.read_csv("preprocess/" + folder + "_preprocessed_discarded.csv")
fecs = list(set(df.fec_id))
fecs_dict = {}

for i in fecs:
    fec_dict = get_fd_from_df(i, df, downfact)
    fecs_dict[i] = fec_dict

traces_base = list(set([get_trace_base (x) for x in fecs]))

#%% RAW PLOT

for i in traces_base:
    i = i+"_fecS03"
    dlf,flf = fecs_dict[i]["dlf"],fecs_dict[i]["flf"]
    plt.plot(dlf,flf)


#%% ALIGNMENT TO THE END

rough_bounds = [[20, 21.5], [8, 25]]
    
for i in traces_base:
    i = i+"_fecS03"
    dlf,flf = fecs_dict[i]["dlf"],fecs_dict[i]["flf"]
    rough_aln_fec = post.AlignFEC(dlf, flf, rough_bounds,
                                  [], *WLCp, custom_function = False)
    rough_aln_fec.dist_aln()
    doff = (rough_aln_fec.x_aln-dlf)[0]

    fecs_dict[i]["doff"] = doff
    plt.plot(rough_aln_fec.x_aln, flf)

#%% FINE DISTANCE ALIGNMENT

dist_aln_bounds = [[0.75, 0.9], [12, 20]]

for i in traces_base:
    i = i+"_fecS03"
    dlf,flf = fecs_dict[i]["dlf"],fecs_dict[i]["flf"]

    doff = fecs_dict[i]["doff"]

    # Now the iteration
    iter_aln_fec = post.AlignFEC(dlf+doff, flf, dist_aln_bounds,
                             [], *WLCp,
                             custom_function = False)
    iter_aln_fec.iterative_dist_aln()

    fecs_dict[i]["doff"] = (iter_aln_fec.x_aln-dlf)[0]
    fecs_dict[i]["foff"] = (iter_aln_fec.F_aln-flf)[0]

    # aligning full bandwidth data
    fecs_dict[i]["d_aln"] = fecs_dict[i]["d"] + fecs_dict[i]["doff"]
    fecs_dict[i]["f_aln"] = fecs_dict[i]["f"] + fecs_dict[i]["foff"]

    # storing aligned downsampled data
    fecs_dict[i]["dlf_aln"] = iter_aln_fec.x_aln
    fecs_dict[i]["flf_aln"] = iter_aln_fec.F_aln

    fplt.plot_individual_fec(iter_aln_fec.x_aln,
                             iter_aln_fec.F_aln,
                             i,
                             "black",
                             [0.55,1.2],
                             [-1,35],
                             "preprocess/preprocessed_pics/" + i + ".png",
                             *WLCp,
                             lcstates = [WLCp[0][1]])

#%%

for i in traces_base:
    i = i+"_fecS03"
    d_aln,f_aln,curve_type = fecs_dict[i]["d_aln"], fecs_dict[i]["f_aln"], fecs_dict[i]["curve_type"]
    
    curve_type= "S"
    if curve_type == "S":
        fecs_dict[i]["rupture_index"] = post.find_tether_rupture(f_aln, 2500, 1, 10, "stretch",
                                                                 maxforce = maxforce)
    else:
        fecs_dict[i]["rupture_index"] = post.find_tether_rupture(f_aln, 2500, 1, 10, "relax",
                                                                 maxforce = maxforce)

#%% CURATION

i = 0
while i < len(traces_base):
    fec = traces_base[i]+"_fecS03"
    dlf,flf = fecs_dict[fec]["dlf"],fecs_dict[fec]["flf"]
    d_aln,f_aln = fecs_dict[fec]["d_aln"],fecs_dict[fec]["f_aln"]
    dlf_aln,flf_aln = fecs_dict[fec]["dlf_aln"],fecs_dict[fec]["flf_aln"]
    ri = fecs_dict[fec]["rupture_index"]
    curve_type = fecs_dict[fec]["curve_type"]
    discard = fecs_dict[fec].get("discard",0)

    curation = post.Curate_FECPostprocessing(fec, 
                                             d_aln,
                                             f_aln,
                                             dlf_aln,
                                             flf_aln,ri, 
                                             discard,
                                             [[0.4,1.2],[-2,50]],
                                             maxforce, # Max force allowed in fec
                                             curve_type = "S",
                                             *WLCp,
                                             var_component=var_component)
    continue_decision = curation.run()
    fecs_dict[fec]["d_aln"] = curation.d_aln
    fecs_dict[fec]["f_aln"] = curation.f_aln
    fecs_dict[fec]["dlf_aln"] = curation.dlf_aln
    fecs_dict[fec]["flf_aln"] = curation.flf_aln
    fecs_dict[fec]["doff"] = (fecs_dict[fec]["dlf_aln"]-dlf)[0]
    fecs_dict[fec]["foff"] = (fecs_dict[fec]["flf_aln"]-flf)[0]
        
    fecs_dict[fec]["discard"] = curation.discard
    fecs_dict[fec]["rupture_index"] = curation.rupture_index

    if continue_decision == "c":
        i += 1
    elif continue_decision == "r":
        if i ==0:
            print("No previous fec to return to. Working on the same FEC.")
        else:
            i -= 1
    else:
        pass

#%% PROPAGATING TO THE SET

for i in traces_base:
    final_fec = i +"_fecS03"
    ri = fecs_dict[final_fec]["rupture_index"]
    set_of_traces = [x for x in fecs if i in x and "fecS03" not in x]
    discard = fecs_dict[final_fec]["discard"]
    
    for c in set_of_traces:
        fecs_dict[c]["doff"] = fecs_dict[final_fec]["doff"]
        fecs_dict[c]["foff"] = fecs_dict[final_fec]["foff"]
        fecs_dict[c]["dlf_aln"] = fecs_dict[c]["dlf"] + fecs_dict[c]["doff"]
        fecs_dict[c]["flf_aln"] = fecs_dict[c]["flf"] + fecs_dict[c]["foff"]
        
        
        fecs_dict[c]["d_aln"] = fecs_dict[c]["d"] + fecs_dict[c]["doff"]
        fecs_dict[c]["f_aln"] = fecs_dict[c]["f"] + fecs_dict[c]["foff"]
        fecs_dict[c]["rupture_index"] = len(fecs_dict[c]["d_aln"]) - 1
        fecs_dict[c]["discard"] = discard
       
#%% PLOTTIG NOT DISCARDED
   
for i in traces_base:
    set_of_traces = [x for x in fecs if i in x]
    for c in set_of_traces:
        discard = fecs_dict[c]["discard"]
        if discard == 0:
            plt.plot(fecs_dict[c]["dlf_aln"],fecs_dict[c]["flf_aln"], label = c)
            plt.legend()
    plt.savefig(f"postprocess/postprocess_pics/{i}.png")
    plt.show()

#%% PANDAS DF FOR KEPT TRACES

# Organizing columns
sorted_cols = ['relative_time', 'trappos1x', 'cal_trappos1x', 'molext',"molext_aln",
               'cal_molext','bead1_disp',
               'bead2_disp', 'diff_bead_disp', 
               'force1x','force2x','diffF',"diffF_aln",
               "rupture_index", 'bead_set', 'mol_id','curve_type', 'curve_number', 
               'date','error_code', 'fec_id']

kept_df = []

## Assembling kept df
for i in fecs:
    d_aln,f_aln = fecs_dict[i]["d_aln"],fecs_dict[i]["f_aln"]
    doff, foff = fecs_dict[i]["doff"],fecs_dict[i]["foff"]
    ri = fecs_dict[i]["rupture_index"]
    discard = fecs_dict[i]["discard"]

    if discard == 0:
        tmpdf = df[df["fec_id"] == i]
        tmpdf.loc[:,"rupture_index"] = int(ri)
        
        if align:
            tmpdf.loc[:,"molext_aln"] = d_aln
            tmpdf.loc[:,"diffF_aln"] = f_aln

            
        else:
            tmpdf.loc[:,"molext_aln"] = np.nan
            tmpdf.loc[:,"diffF_aln"] = np.nan
        kept_df.append(tmpdf)

kept_df = pd.concat(kept_df)

## Saving organized kept_df
kept_df = kept_df[sorted_cols]
kept_df.to_csv("postprocess/" + folder + "_postprocessed.csv", index = False)            
  
#%%          
discard_df = []

# Assembling discard df
for i in fecs:
    d_aln,f_aln = fecs_dict[i]["d_aln"],fecs_dict[i]["f_aln"]
    ri = fecs_dict[i]["rupture_index"]
    discard = fecs_dict[i]["discard"]

    if discard == 5:
        tmpdf = df[df["fec_id"] == i]
        tmpdf.loc[:,"rupture_index"] = ri
        
        if align:
            tmpdf.loc[:,"molext_aln"] = d_aln
            tmpdf.loc[:,"diffF_aln"] = f_aln

            
        else:
            tmpdf.loc[:,"molext_aln"] = np.nan
            tmpdf.loc[:,"diffF_aln"] = np.nan

            
        tmpdf.loc[:, "error_code"] = 5
        discard_df.append(tmpdf)

if len(discard_df) != 0:
    discard_df = pd.concat(discard_df)
else:
    discard_df = pd.DataFrame(columns=sorted_cols)

## Saving organized discard_df
discard_df = discard_df[sorted_cols]
discard_df.to_csv("postprocess/" + folder + "_postprocessed_discarded.csv", index = False)
        
#%% FILTERING STATISTINCS

kept = len(np.unique(kept_df["fec_id"]))
discarded = len(np.unique(discard_df["fec_id"])) + len(np.unique(df_preprocessed_discarded["fec_id"]))

discarded_1 = len(np.unique(df_preprocessed_discarded[df_preprocessed_discarded["error_code"] == 1]["fec_id"]))
discarded_2 = len(np.unique(df_preprocessed_discarded[df_preprocessed_discarded["error_code"] == 2]["fec_id"]))
discarded_3 = len(np.unique(df_preprocessed_discarded[df_preprocessed_discarded["error_code"] == 3]["fec_id"]))
discarded_4 = len(np.unique(df_preprocessed_discarded[df_preprocessed_discarded["error_code"] == 4]["fec_id"]))
discarded_5 = len(np.unique(discard_df[discard_df["error_code"] ==5]["fec_id"]))

cat_size = [kept, discarded_1, discarded_2, discarded_3, discarded_4,discarded_5]
plt.figure(figsize = (12, 9))
plt.pie(cat_size, autopct = lambda p: '{:.2f}%'.format(p) if p > 0 else '',
        textprops={'fontsize': 20})
plt.legend(["Pass",
            "Baseline not found",
            "Fec not found",
            "Piezo calibration has bad quality",
            "Bad baseline quality",
            "Manually discarded"], fontsize = 15
           )

plt.savefig(folder + "_filt_piechart.png", format = "png",dpi = 300)

#%% SAVING METADATA

# Saving parameters used and date of postprocessing run
now = datetime.now()

with open ("postprocess/" + folder + "_postprocessing_metadata.txt", "w+") as f:
    f.write(folder + " was postprocessed on: " + str(now) + "\n\n\n")

    if align:
        f.write("Alignment was on")
        if end_alignment:
            f.write("Alignment was done to the end state")
        else:
            f.write("Alignment was done to the initial state")
        f.write("Fec was aligned to a eWLC with the following parameters:\n")
        f.write(str(WLCp) + "\n")
        f.write(f"Rough aln was performed using the following slice: {rough_bounds}\n\n")
        f.write("Fine aln in the distance and force axis were done with the following bounds: " + "\n")
        f.write("distance aln: " + str(dist_aln_bounds) + "\n")

#%%

dill.dump_session("postprocess/" + folder + "_post.pkl");
