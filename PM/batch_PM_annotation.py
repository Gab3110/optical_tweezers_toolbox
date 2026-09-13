
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

#%% DEFINE MODULES

import numpy as np
import pandas as pd
import sys, os
from datetime import datetime
import dill,re
import matplotlib.pyplot as plt
import copy
import seaborn as sns

#%% DEFINE CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
from tweezers_toolbox_modules import analysis as ana
from tweezers_toolbox_modules.models import ini_eWLC

#%% DEFINE FUNCTIONS

def get_trace_base (trace):
    underscore_indexes = [x.start() for x in re.finditer("_", trace)]
    trace_base = trace[:underscore_indexes[-1]]
    return(trace_base)

def extract_from_pandasdf (df, rupture_force_detection, ds = 25, df_lctraces = None):
    fec_dict = {}
    fecs = list(set(df.fec_id))

    for i in fecs:
        tmpdf_post = df[df["fec_id"] == i]
        ri = tmpdf_post.rupture_index.to_list()[0]
        curve_type = tmpdf_post.curve_type.to_list()[0]
        error_code = tmpdf_post.error_code.to_list()[0]
        
        trappos = tmpdf_post.trappos1x.to_numpy()[0:ri+1]
        f = tmpdf_post.diffF.to_numpy()[0:ri+1]
        molext = tmpdf_post.molext.to_numpy()[0:ri+1]
        molext_aln =  tmpdf_post.molext_aln.to_numpy()[0:ri+1]
        f_aln = tmpdf_post.diffF_aln.to_numpy()[0:ri+1]

        trapposlf = CK_filtfilt(trappos, ds)
        flf = CK_filtfilt(f, ds)
        molextlf = CK_filtfilt(molext, ds)
        molext_alnlf = CK_filtfilt(molext_aln, ds)
        f_alnlf = CK_filtfilt(f_aln, ds)

        fec_data = {}
        fec_data["error_code"] = error_code
        fec_data["curve_type"] = curve_type
        fec_data["ri"] = ri
        fec_data["trappos"] = trappos
        fec_data["f"] = f
        fec_data["molext"] = molext
        fec_data["molext_aln"] = molext_aln
        fec_data["trapposlf"] = trapposlf
        fec_data["flf"] =flf
        fec_data["molextlf"] =molextlf
        fec_data["molext_alnlf"] =molext_alnlf
        fec_data["f_aln"] = f_aln
        fec_data["f_alnlf"] = f_alnlf

        if df_lctraces:
            tmpdf_lct = df_lctraces[df_lctraces["fec_id"] == i]
            lctrace = tmpdf_lct.Lc_trace.to_numpy()
            fec_data["lctrace"] = lctrace

        fec_dict[i] = fec_data

    return(fec_dict)

def unpack_trans_obj(trans_obj):

    total_path = trans_obj.total_direct_path
    tdExt_direct = trans_obj.all_trans_info[total_path]["dExt"]
    tdLc_direct = trans_obj.all_trans_info[total_path]["dLc"]

    if len(trans_obj.all_trans_info.keys()) != 1:
        dExts = [trans_obj.all_trans_info[x]["dExt"] for x in trans_obj.all_trans_info.keys() if x != total_path]
        dLcs = [trans_obj.all_trans_info[x]["dLc"] for x in trans_obj.all_trans_info.keys() if x != total_path]
    else:
        dExts = [trans_obj.all_trans_info[x]["dExt"] for x in trans_obj.all_trans_info.keys()]
        dLcs = [trans_obj.all_trans_info[x]["dLc"] for x in trans_obj.all_trans_info.keys()]

    # Append 0 to match old behavior
    dExts.append(0)
    dLcs.append(0)
    
    dExts_sum = np.cumsum([0] + dExts[:-1]).tolist()
    dLcs_sum = np.cumsum([0] + dLcs[:-1]).tolist()
    dLcs_sum_complement = np.cumsum([0] + list(np.flip(dLcs[:-1]))).tolist()

    return dExts, dLcs, dExts_sum, dLcs_sum,dLcs_sum_complement, tdExt_direct, tdLc_direct

def restore_from_csv(fecs, anndf,fec_dict):
    for i in fecs:
        tmpdfann = anndf[anndf.fec_id == i]
        fec_dict[i]["trans_idxs"] = tmpdfann.trans_idxs.to_numpy()
        fec_dict[i]["trans_idxs"][-1] = len(fec_dict[i]["molext_alnlf"]) - 1
        
        fec_dict[i]["dExts"] = tmpdfann.dExts.to_numpy()
        fec_dict[i]["dExts_sum"] = tmpdfann.dExts_sum.to_numpy()
        fec_dict[i]["dLcs"] = tmpdfann.dLcs.to_numpy()
        fec_dict[i]["dLcs_sum"] = tmpdfann.dLcs_sum.to_numpy()
        fec_dict[i]["states"] = tmpdfann.states.to_numpy()
        fec_dict[i]["tdExt_direct"] = tmpdfann.tdExt_direct.to_numpy()
        fec_dict[i]["tdLc_direct"] = tmpdfann.tdLc_direct.to_numpy()
        
#%%

# Define folder variable as the name of the folder containing the data (expt name)
folder = "260430_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_20nMChd1_1mMATP_1umBeads_b84_drift"

# Define which method is used to detect rupture forces (two options: peaks or hmm
# are available. If hmm is picked, make sure you have run batch_fec_perpointLc_calculation.py
# to generate the df_lctraces dataframe. Peaks is recommended if data only displays
# cooperative transitions.
rupture_force_detection = "peaks"

# Define dictionary of options. Modify accordingly to your methods and needs
if rupture_force_detection == "peaks":
    parameters = {"rupture_force_detection" : "peaks",
                "force_cutoff" : 1,
                "threshold" : 0.5,
                "distance" : 5,
                "window" : 5}

elif rupture_force_detection == "hmm":
    parameters = {"rupture_force_detection" : "hmm",
                "force_cutoff" : 2,
                "max_states" : 10}
else:
    print("Invalid method, please pick peaks or hmm.")


# NOTE: SELECT THE REFERENCE WLC MODEL YOU USED IN POST PROCESSING 
WLCp = ([50,  0.78506],[0.65,0.01296])

# THIS SHOULD BE ALWAYS ON UNLESS DURING POSTPROCESSING, YOU DID NOT ALIGN FECS
aligned = True

# CHANGE IF YOUR REFERENCE WLC IS FOR THE UNFOLDED
# PART OF YOUR MOLECULE
end_alignment = True 

# MODIFY THIS NUMBER TO MATCH THE WLCp TUPLE INDEX OF THE WLC THAT WILL CHANGE.
# For example, say WLCp = ([Lp_DNA, Lc_DNA, St_DNA],[Lp_protein, Lc_protein])
# and you are unfolding protein. Then you should select var_component = 1.
var_component = 0

# SESSION RESTORAGE OPTIONS
restore_session = False # Change to true if want to restore

dill_session_path = None # Define dill path, script will try to restore from here
                         # by default 
csv_file_path = None # Define csv file, if dill fails somehow, will default to this.

# for visualization
ds_factor = 25

#%% LOADING DATAFRAME

dfpost = pd.read_csv("postprocess/" + folder + "_postprocessed.csv")

if rupture_force_detection == "hmm":
    df_lctraces = pd.read_csv("per_pointLc/" + folder + "_lctraces.csv")
else:
    df_lctraces = None

#%% CREATE DIR
try:
    os.makedirs("annotate/annotate_pics")
except:
    pass

#%% Order dataframe on a dict

fec_dict = extract_from_pandasdf (dfpost,rupture_force_detection,ds = ds_factor, df_lctraces=df_lctraces)     
traces_base = list(set([get_trace_base (x) for x in fec_dict.keys()]))

#%% 

not_passive = []

# find rips on S03
for i in fec_dict.keys():
    if "fecS03" in i:
        #i = "250722_10_01_fecS01"
        fec_data = fec_dict[i]
        flf = fec_data["flf"]
        ind = np.arange(0,len(flf),1)
            
        fec_dict[i]["trans_idxs"] = ana.find_transitions_peaks (ind,
                                                                flf,
                                                                force_cutoff = parameters["force_cutoff"],
                                                                threshold = parameters["threshold"],
                                                                distance = parameters["distance"],
                                                                window = parameters["window"])
        not_passive.append(i)

# find rips on rest
for i in fec_dict.keys():
    if ("fecS00" in i) or ("fecS01" in i) or ("fecS02" in i):
        #i = "250722_10_01_fecS01"
        fec_data = fec_dict[i]
        flf = fec_data["flf"]
        ind = np.arange(0,len(flf),1)
        fec_dict[i]["trans_idxs"] = np.array([ind[-1]])
        not_passive.append(i)

#%% TRANSITIONS MANUAL CURATION

# here curation will be done plotting molext as it is easier to see minor
# transitions this way (at least for me).

i = 0

while i < len(not_passive):
    fec_id = not_passive[i]
    fec_data = fec_dict[fec_id]

    d = fec_data["molext_aln"]
    dlf = fec_data["molext_alnlf"]

    f =  fec_data["f_aln"]
    flf = fec_data["f_alnlf"]
    trans_idxs = fec_data["trans_idxs"]
    curve_type = fec_data["curve_type"]

    o = ana.EditTransitions(fec_id,dlf, flf,d,f,  trans_idxs,[[0.3,1],[-0.5,50]],
                            parameters,var_component = 0,  lctrace = None, curve_type=curve_type)

    continue_decision = o.run()

    fec_dict[fec_id]["trans_idxs"] = o.trans_idxs

    if continue_decision == "c":
        i += 1
    elif continue_decision == "r":
        if i ==0:
            print("No previous fec to return to. Working on the same FEC.")
        else:
            i -= 1
    else:
        pass

# this will save your session until now in case something bad happens.
# important since th curation is tedious and takes time
dill.dump_session("annotate/annotation_checkpoint.pkl")

#%% FIND RIP EXTENSIONS

trans_params = {"nfit" : 20, "navg" : 3, "FAvgRange" : 0.05, "FFitRange" : 1,"padding": 3,
              "Lp": 50, "S": 1200}

fecsS03 = [x for x in not_passive if "fecS03" in x]
fecsS00_01_02 = [x for x in not_passive if "fecS03" not in x]

for i in fecsS03:
    fec = fec_dict[i]
    trans_idxs = fec["trans_idxs"]
    curve_type = fec["curve_type"]
    d = fec["trapposlf"]
    f = fec["flf"]

    if curve_type == "R":
        d = np.flip(d)
        f = np.flip(f)
        trans_idxs = len(f) - 1 - trans_idxs
        
    trans_idxs.sort()
    
    if len(trans_idxs) > 1:
        trans_obj = ana.FindTransExtensions(i,d, f, trans_idxs,
                                            trans_params = trans_params, debug=False)
        trans_obj.run()
        fec_dict[i]["trans_obj"] = trans_obj

#%% SAVE INFO INTO DICT

for i in fecsS03:
    lf_trans_idxs = fec_dict[i]["trans_idxs"] 
    if len(lf_trans_idxs) > 1:
        trans_obj = fec_dict[i]["trans_obj"]
        dExts, dLcs,dExts_sum, dLcs_sum,dLcs_sum_complement, tdExt_direct, tdLc_direct = unpack_trans_obj (trans_obj)
    else:
        dExts, dLcs,dExts_sum, dLcs_sum,dLcs_sum_complement, tdExt_direct, tdLc_direct = [0],[0],[0],[0],[0],[0],[0]
        
    fec_dict[i]["dExts"] = dExts
    fec_dict[i]["dLcs"] = dLcs
    fec_dict[i]["dLcs_sum"] = dLcs_sum
    fec_dict[i]["dLcs_sum_complement"] = dLcs_sum_complement
    fec_dict[i]["dExts_sum"] = dExts_sum
    fec_dict[i]["tdExt_direct"] = tdExt_direct
    fec_dict[i]["tdLc_direct"] = tdLc_direct
    fec_dict[i]["fec_id"] = i
    
    # NOTE: IF ALIGNED TO THE END STATE (UNFOLDED, FOR EXAMPLE), YOU NEED TO SUBSTRACT
    # THE TRANSITION SIZE FROM THE REFERENCE LC OF YOUR WORM LIKE CHAIN.
    # IF ALIGNED TO THE BEGINING THEN YOU NEED TO SUM THEM.
    if aligned:
        if end_alignment:            
            fec_dict[i]["states"] =  np.flip(WLCp[var_component][1] - np.array(dLcs_sum_complement))
        else:
            fec_dict[i]["states"] =  WLCp[var_component][1] + np.array(dLcs_sum)
    else:
        fec_dict[i]["states"] = np.nan

#%% STATES MANUAL CURATION

if aligned:
    i = 0
    while i < len(fecsS03):
        fec_id = fecsS03[i]
        fec_data = fec_dict[fec_id]
    
        d = fec_data["molext_aln"]
        dlf = fec_data["molext_alnlf"]
    
        f =  fec_data["f_aln"]
        flf = fec_data["f_alnlf"]
        trans_idxs = fec_data["trans_idxs"]
        stts = fec_data["states"]
        
        error_code = fec_data["error_code"]
        
        o = ana.EditStates(
            fec_id,dlf,flf,d,f,trans_idxs,
            trans_params,
            stts,
            WLCp[var_component][1],
            error_code,
            [[0.3,1],[-2,35]],
            *WLCp,
            var_lc_component=var_component
        )
        
        continue_decision = o.run()
    
        fec_dict[fec_id]["trans_idxs"] = o.trans_idxs
        fec_dict[fec_id]["states"] = o.states
        fec_dict[fec_id]["dLcs_sum"] = o.dLcs_sum
        fec_dict[fec_id]["dLcs"] = o.dLcs
    
        if continue_decision == "c":
            i += 1
        elif continue_decision == "r":
            if i ==0:
                print("No previous fec to return to. Working on the same FEC.")
            else:
                i -= 1
        else:
            pass
    
    # this will save your session until now in case something bad happens.
    # important since th curation is tedious and takes time
    dill.dump_session("annotate/annotation_checkpoint2.pkl")


#%% RIP EXTENSION BETWEEN S00/S01/S02 AND FECS03

trans_params = {"nfit" : 20, "navg" : 5, "FAvgRange" : 0.05, "FFitRange" : 1,"padding": 3,
              "Lp": 50, "S": 1200}

for i in traces_base:
    fecS00_name = f"{i}_fecS00"
    fecS01_name = f"{i}_fecS01"
    fecS02_name = f"{i}_fecS02"
    fecS03_name = f"{i}_fecS03"

    fecS00_data = fec_dict.get(fecS00_name,{})
    fecS01_data = fec_dict.get(fecS01_name,{})
    fecS02_data = fec_dict.get(fecS02_name,{})
    fecS03_data = fec_dict.get(fecS03_name,{})
    
    for x in [fecS00_data,fecS01_data,fecS02_data]:
        if x:
            ind_offset = len(x["flf"])
            trans_x = x["trans_idxs"]
            trans_fecS03 = ind_offset + fecS03_data["trans_idxs"]
            trans_x_fecS03 = np.concatenate([trans_x,trans_fecS03])
            #stts_x_fecS03 = np.concatenate([np.array([np.nan]),fecS03_data["states"]])
            
            molext_alnlf_merged = np.concatenate([x["molext_alnlf"],  fecS03_data["molext_alnlf"]])
            flf_merged = np.concatenate([x["flf"],  fecS03_data["flf"]])
            
            trans_obj = ana.FindTransExtensions("",molext_alnlf_merged, flf_merged, trans_x_fecS03,
                                                trans_params = trans_params, debug=False)
            
            ruptureF, x1, x2,dExt,dLc = trans_obj.find_path_extension((trans_x[0],trans_fecS03[0]), 
                                                                      forward = True, 
                                                                      sanity_test = False)
            if np.isnan(dLc):
                _, fp = trans_obj.find_fixing_paths((trans_x[0],trans_fecS03[0]))
                if len(fp) != 0:
                    _, fp = trans_obj.find_fixing_paths((trans_x[0],trans_fecS03[0]))
                    fixing_path = fp[0]
                    idx_map = np.where(np.isin(trans_fecS03,fixing_path))[0]
                    
                    dLcs_fecS03 = fecS03_data["dLcs"]
                    dLc1 = np.sum(np.array(dLcs_fecS03)[idx_map[0]:idx_map[-1]+1])
                    _,_,_,_,dLc2 = trans_obj.find_path_extension((trans_x[0],fixing_path[-1]), 
                                                                              forward = True, 
                                                                              sanity_test = False)
                    dLc = dLc2 - dLc1
                
            x["states"] = np.array([fecS03_data["states"][0] - dLc])
            x["dLcs_sum"] = np.array([0])
            x["dExts"] = np.array([dExt])
            x["dExts_sum"] = np.array([0])
            x["tdExt_direct"] = np.nan
            x["tdLc_direct"] = np.nan
            x["dLcs"] = np.array([dLc])
                
#%% CURATE STATES OF S00/S01/S02 AND FECS03

if aligned:
    i = 0
    while i < len(fecsS00_01_02):
        fec_id = fecsS00_01_02[i]
        fec_data = fec_dict[fec_id]
        trace_base = get_trace_base(fec_id)
    
        d = fec_data["molext_aln"]
        dlf = fec_data["molext_alnlf"]
    
        f =  fec_data["f_aln"]
        flf = fec_data["f_alnlf"]
        trans_idxs = fec_data["trans_idxs"]
        stts = fec_data["states"]
        
        error_code = fec_data["error_code"]
        
        o = ana.EditStates(
            fec_id,dlf,flf,d,f,trans_idxs,
            trans_params,
            stts,
            WLCp[var_component][1],
            error_code,
            [[0.5,1],[-2,35]],
            *WLCp,
            var_lc_component=var_component
        )
        
        continue_decision = o.run()
    
        fec_dict[fec_id]["states"] = o.states
        fec_dict[fec_id]["dLcs_sum"] = o.dLcs_sum
        fec_dict[fec_id]["dLcs"] = fec_dict[trace_base+"_fecS03"]["states"][0] - o.states
    
        if continue_decision == "c":
            i += 1
        elif continue_decision == "r":
            if i ==0:
                print("No previous fec to return to. Working on the same FEC.")
            else:
                i -= 1
        else:
            pass
    
    # this will save your session until now in case something bad happens.
    # important since th curation is tedious and takes time
    dill.dump_session("annotate/annotation_checkpoint3.pkl")

#%% RECALCULATE MIRROR POSITION EXTENSIONS WITH FINAL RIPS FOR FECSS03

trans_params = {"nfit" : 20, "navg" : 3, "FAvgRange" : 0.05, "FFitRange" : 1,"padding": 3,
              "Lp": 50, "S": 1200}

fecsS03 = [x for x in not_passive if "fecS03" in x]
fecsS00_01_02 = [x for x in not_passive if "fecS03" not in x]

for i in fecsS03:
    fec = fec_dict[i]
    trans_idxs = fec["trans_idxs"]
    curve_type = fec["curve_type"]
    d = fec["trapposlf"]
    f = fec["flf"]

    if curve_type == "R":
        d = np.flip(d)
        f = np.flip(f)
        trans_idxs = len(f) - 1 - trans_idxs
        
    trans_idxs.sort()
    
    if len(trans_idxs) > 1:
        trans_obj = ana.FindTransExtensions(i,d, f, trans_idxs,
                                            trans_params = trans_params, debug=False)
        trans_obj.run()
        fec_dict[i]["trans_obj"] = trans_obj

# SAVE IN DICTIONARY
for i in fecsS03:
    lf_trans_idxs = fec_dict[i]["trans_idxs"] 
    if len(lf_trans_idxs) > 1:
        trans_obj = fec_dict[i]["trans_obj"]
        dExts, _, dExts_sum, _, _, _, _ = unpack_trans_obj (trans_obj)
    else:
        dExts, _,dExts_sum, _,_, _, _ = 0,0,0,0,0,0,0
        
    fec_dict[i]["dExts"] = dExts
    fec_dict[i]["dExts_sum"] = dExts_sum
        
#%%        
          
# ---------- helper: parse group token ----------
def get_group(name):
    """
    Extract a run-type token used for coloring.
    Examples matched: fecS03, dct02, spm02
    """
    m = re.search(r"(fecS\d+|dct\d+|spm\d+)", str(name))
    return m.group(1) if m else str(name)  # fallback

# ---------- build stable group -> color map (DO THIS ONCE) ----------
all_groups = sorted({get_group(k) for k in fec_dict.keys()})

# use a "marked" palette; tab20 gives 20 distinct-ish colors
palette = sns.color_palette("tab10", n_colors=max(20, len(all_groups)))

group_color = {g: palette[i % len(palette)] for i, g in enumerate(all_groups)}

# ---------- plotting ----------
pf = np.arange(0.01, 30, 0.01)
WLCpars_mod = copy.deepcopy(WLCp)

for i in traces_base:
    set_of_traces = [x for x in fec_dict.keys() if i in x]

    plt.figure()
    plt.ylim(0,25)
    plt.xlim(0.4, 0.87)

    trace_handles = []
    state_items = []  # (st_value, handle)

    for c in set_of_traces:
        base_color = group_color[get_group(c)]

        # ---- plot trace ----
        h_trace, = plt.plot(
            fec_dict[c]["molext_alnlf"],
            fec_dict[c]["f_alnlf"],
            label=c,
            color=base_color,
            alpha = 0.7
        )
        trace_handles.append(h_trace)

        # ---- plot WLC states for this trace (sorted) ----
        states_c = sorted(fec_dict[c].get("states", []))
        for st in states_c:
            WLCpars_mod[var_component][1] = st
            pdist = ini_eWLC(1, pf, *WLCpars_mod)

            h_state, = plt.plot(
                pdist, pf, "--",
                color=base_color,
                alpha=1,       # "lighter" via alpha
                linewidth=2,
                label=f"{st:.3f} um"
            )
            state_items.append((st, h_state))

    # ---- Legend 1: traces ----
    leg1 = plt.legend(handles=trace_handles, loc="upper left", title="Traces")
    plt.gca().add_artist(leg1)

    # ---- Legend 2: states sorted low->high ----
    state_items.sort(key=lambda t: t[0])
    state_handles = [h for _, h in state_items]
    plt.xlabel("Extension (um)")
    plt.ylabel("Force (pN)")

    plt.legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        title="States"
    )
    plt.savefig(f"annotate/annotate_pics/{i}.png" , dpi = 300, bbox_inches = "tight")
    plt.show()

#%% STORE IN DF

annotation_df = []
for i in not_passive:
    tmpdf = pd.DataFrame()
    curve_type = fec_dict[i]["curve_type"]
    
    if curve_type == "R":
        tmpdf["trans_idxs"] = np.flip(fec_dict[i]["trans_idxs"])
    elif curve_type == "S":
        tmpdf["trans_idxs"] = fec_dict[i]["trans_idxs"]
    else:
        tmpdf["trans_idxs"] = fec_dict[i]["trans_idxs"]
     
    tmpdf["dExts"] = fec_dict[i]["dExts"]
    tmpdf["dExts_sum"] = fec_dict[i]["dExts_sum"]
    tmpdf["dLcs"] = fec_dict[i]["dLcs"]
    tmpdf["dLcs_sum"] = fec_dict[i]["dLcs_sum"]
    tmpdf["states"] = fec_dict[i]["states"]
    tmpdf["tdExt_direct"] = fec_dict[i]["tdExt_direct"]
    tmpdf["tdLc_direct"] = fec_dict[i]["tdLc_direct"]
    tmpdf["fec_id"] = i
    annotation_df.append(tmpdf)


#%%
annotation_df = pd.concat(annotation_df)
annotation_df.to_csv("annotate/" + folder + "_annotated_states.csv", index = False)

#%% SAVING METADATA

now = datetime.now()
with open (f"annotate/{folder}_annotation_metadata.txt", "w+") as f:
    f.write(folder + " annotation was performed on : " + str(now) + "\n\n")
    f.write("The following parameters were used:\n")
    f.write("Method for transition detection: " + str(parameters["rupture_force_detection"]) + "\n")
    f.write("force_cutoff " + str(parameters["force_cutoff"]) + "\n")

    if rupture_force_detection == "peaks":
        f.write("threshold: " + str(parameters["threshold"]) + "\n")
        f.write("distance: " + str(parameters["distance"]) + "\n")
        f.write("window: " + str(parameters["window"]) + "\n\n")
    else:
        f.write("max_states: " + str(parameters["max_states"]) + "\n\n")

    f.write("Parameters for finding Lc changes are: " + "\n")
    f.write("Lp: " + str(trans_params["Lp"]) + "\n")
    f.write("S: " + str(trans_params["S"]) + "\n")
    f.write("Min points for linear fit: " + str(trans_params["nfit"]) + "\n")
    f.write("Min points for averaging: " + str(trans_params["navg"]) + "\n")
    f.write("Force range for averaging: " + str(trans_params["FAvgRange"]) + "\n")
    f.write("Force range for linear fitting: " + str(trans_params["FFitRange"]) + "\n")
    f.write("Padding between transitions: " + str(trans_params["padding"]) + "\n")

#%% PICKLE SESSION WITH DILL MODULE
 
dill.dump_session("annotate/" + folder + "_session.pkl")

#%% REMOVE CHECKPOINTS

try:
    os.remove("annotate/annotation_checkpoint.pkl")
except:
    pass  

try:
    os.remove("annotate/annotation_checkpoint2.pkl")
except:
    pass

try:
    os.remove("annotate/annotation_checkpoint3.pkl")
except:
    pass




