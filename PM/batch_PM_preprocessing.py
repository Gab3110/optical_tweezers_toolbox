import os, glob, re, sys
import numpy as np
import pandas as pd
from datetime import datetime
import lumicks.pylake as lk

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules import preprocessing as pre
from tweezers_toolbox_modules import fec_plotting as fplt

#%% DEFINE FUNCTIONS

def rename_dict_keys(d, rename_map):
    # rename keys by name (order-independent)
    return {rename_map.get(k, k): v for k, v in d.items()}

def get_random_file_sampling_freq ():
    files = glob.glob("*.h5")
    if not files:
        files = glob.glob("rawdat/*.h5")
    first_file = lk.File(files[0])
    frequency = pre.get_sampling_freq(first_file)
    return(frequency)

def gen_baseline_name (trace_id):
    # Getting baseline name
    underscore_ids = [x.start() for x in re.finditer("_", trace_id)]
    dte =  trace_id[0:underscore_ids[0]]
    bset = trace_id[underscore_ids[0]+1:underscore_ids[1]]
    number = trace_id[-5:-3]
    
    if (number == "00") or (number == "01"):
        bsl_number = "01"
    elif (number == "02") or (number == "03"):
        bsl_number = "02"
    else:
        return("Error")
    
    bsl_pthname_current = f"{dte}_{bset}_fec{bsl_number}.h5"
    bsl_pthname_current_base = os.path.basename(bsl_pthname_current).replace(".h5","")

    return(bsl_pthname_current, bsl_pthname_current_base)

def transpose_dict_list (list_dict):
    transposed_list = []
    for item_dict in list_dict:
        time_key = next(k for k in item_dict if "timeds" in k)
        signal_len = len(item_dict[time_key])

        for i in range(signal_len):
            row = {}

            for key, value in item_dict.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    row[key] = value[i]
                else:
                    row[key] = value
            transposed_list.append(row)
    return(transposed_list)

def stream_dicts_to_csv(
    dict_iter,
    out_csv,
    *,
    rename_by_position=None,   # list of column names, applied as df.columns = ...
    sorted_columns=None               # final ordered columns to write
):

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    wrote_header = False

    for d in dict_iter:
        vec = {}
        scalars = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                vec[k] = np.asarray(v)
            else:
                scalars[k] = v

        df = pd.DataFrame(vec)
        for k, v in scalars.items():
            df[k] = v

        # ---- NEW: rename columns by POSITION (matches your old df.columns = [...]) ----
        if rename_by_position is not None:
            if len(df.columns) != len(rename_by_position):
                raise ValueError(
                    f"rename_by_position has {len(rename_by_position)} names "
                    f"but df has {len(df.columns)} columns. "
                    f"Keys are: {list(df.columns)}"
                )
            df.columns = list(rename_by_position)

        # ---- Then reorder/select final columns ----
        if sorted_columns is not None:
            df = df[sorted_columns]

        df.to_csv(out_csv, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        
def parse_metadata (fec_name):
    fec_id = re.sub(r'^.*?FD Curve ', '', fec_name)
    fec_id = fec_id.replace(".h5","")
    fec_id = fec_id.replace("rawdat/", "")

    underscore_indexes = [x.start() for x in re.finditer("_", fec_id)]

    # Parsing the date from the fec's name. Also adding 20 in front
    # to complete the year. Adding the date to the df.
    date = "20" + fec_id[0:underscore_indexes[0]]

    # Parsing the beadset number
    bead_set = int(fec_id[underscore_indexes[0]+1:underscore_indexes[1]])

    # Parsing the molid from the csv's
    mol_id = fec_id[underscore_indexes[1]+1:underscore_indexes[2]]

    if len(fec_id[underscore_indexes[-1]+1:]) == 6:
        curve_type = fec_id[underscore_indexes[-1]+1:underscore_indexes[-1]+5]
        # Parsing the curve number from the csv's.
        curve_number = fec_id[underscore_indexes[-1]+5:len(fec_id)+1]
    elif len(fec_id[underscore_indexes[-1]+1:]) == 5:
        curve_type = fec_id[underscore_indexes[-1]+1:underscore_indexes[-1]+4]
        # Parsing the curve number from the csv's.
        curve_number = fec_id[underscore_indexes[-1]+4:len(fec_id)+1]
    else:
        curve_type = np.nan
        print(f"Could not get curve type of {fec_name}")
        
    # Adding metadata to the fec dict
    meta_data_dict = {}
    meta_data_dict["fec_id"] = fec_id
    meta_data_dict["date"] = date
    meta_data_dict["bead_set"] = bead_set
    meta_data_dict["mol_id"] = mol_id
    meta_data_dict["curve_type"] = curve_type
    meta_data_dict["curve_number"] = curve_number

    return(meta_data_dict)

#%% DEFINING GENERAL VARIABLES

# Define folder variable as the name of the folder containing the data (expt name)
folder = "260430_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_20nMChd1_1mMATP_1umBeads_b84_drift"

# Print current directory
full_path = os.getcwd()
print("you are working in: " + full_path) ## IMPORTANT: Make sure you are in the
                                         ## in the folder containing the data.
if folder not in full_path:
    sys.exit("""You are not in the folder containing the data. Change your
                 working directory to the folder containing the data.""")

# Aquisition frequency of the instrument
frequency = round(get_random_file_sampling_freq ())
print ("instrument frequency is: " + str(frequency))

# Downsampling factor
target_downsampled_frequency = 3125
downfact = pre.get_downsampling_factor (frequency,target_downsampled_frequency)
smoothening_factor = downfact # For smoothing the corrected forces and decimating residues for visual clarity
print(f"A factor of {downfact} will be applied to reach a target sampling rate of 3125 Hz")


# Polynomial order used to convert trap position to surface-to-surface bead
# distance at F = 0 (no tether)
video_microscopy_pol_order = 1

# Minimum polinomial order for the baseline
bsl_pol_order = 7


# Parameters for preprocessing
opt_dict = {"bead1_diameter" : 0.81,
            "bead2_diameter" : 0.81,
            "correct_force_offset" : False,
            "apply_dist_offset" : False,
            "video_trap_regress_cal" : False,
            "filter_type" : "bessel"}

correct_f1x_instability = False
correct_spatial_drift = False
recalibrate = False

#%% SORTING AND RENAMING RAW FILES

# Getting a list of the raw files (baseline + molecule fecs)
files = glob.glob("*.h5")

# Try to make rawdat folder in case it does not exist
try:
    os.mkdir("rawdat")
except:
    pass

# Remove identifier and move it to rawdat folder
for i in files:
    new_name = re.sub(r'^.*?(?:Marker\s+|FD Curve\s+)', '', i, flags=re.IGNORECASE)
    os.rename(i, "rawdat/" + new_name)

# Get a list of new path of the files
files = glob.glob("rawdat/*.h5")

#%%

try:
    os.makedirs("preprocess/video_microscopy/")
except:
    pass

try:
    os.makedirs("preprocess/baselines/fits/")
    os.makedirs("preprocess/baselines/corrected_baseline_forces/")
    os.makedirs("preprocess/baselines/residuals/")
except:
    pass

try:
    os.makedirs("preprocess/preprocessed_pics")
except:
    pass

#%%

cal_ranges = {}

for i in files:
    if all(x not in i for x in ("fec01", "fec02","fec03")):
        bsl_base = gen_baseline_name(i)[1]
        print(i,bsl_base)
        lo, hi = pre.get_raw_trappos_domain(i)
    
        if bsl_base not in cal_ranges:
            cal_ranges[bsl_base] = [lo, hi]
        else:
            cal_lo, cal_hi = cal_ranges[bsl_base]
            cal_ranges[bsl_base] = [min(cal_lo, lo), max(cal_hi, hi)]

for bsl_base in list(cal_ranges.keys()):
    bsl_lo, bsl_hi = pre.get_raw_trappos_domain(f"rawdat/{bsl_base}.h5")
    cal_lo, cal_hi = cal_ranges[bsl_base]
    # trim envelope to baseline domain (intersection)
    cal_ranges[bsl_base] = [max(cal_lo, bsl_lo), min(cal_hi, bsl_hi)]

#%%
# Initialize dictionaries
baselines = {}
video_microscopy = {}
fecs = {}

error_handling_dict = {0: "Pass", 1: "Baseline not found!", 2: "Fec not found!",
                       3: "Piezo calibration fit has bad quality!", 4: "Weird baseline!"}

for i in files:
    if all(x not in i for x in ("fec01", "fec02","fec03")):
        print(i)
        # Getting fec basename
        fec_basename = os.path.basename(i)
        fec_basename = fec_basename.replace(".h5","")

        # print fec
        print("Working on " + fec_basename)

        # Getting baseline name
        bsl_pthname_current, bsl_pthname_current_base = gen_baseline_name(i)
        
        # getting cal_range
        crange = cal_ranges[bsl_pthname_current_base]
        
        if recalibrate:
            if all(x in i for x in ("spm02","dct02","fecS03")):
                try:
                    recal_file = lk.File(bsl_pthname_current.replace("02","03"))
                except:
                    recal_file = None
        else:
            recal_file = None
        
        if bsl_pthname_current_base not in baselines.keys():
            preprocessed_fec = pre.PiezoTracking(i,
                                                  bsl_pthname_current,
                                                  1,
                                                  downfact,
                                                  bsl_pol_order,
                                                  crange,
                                                  opt_dict,
                                                  video_trappos_regression_bsl_mdl_results = None)
            preprocessed_fec.preprocess()
            errcod = preprocessed_fec.error_code

            if errcod == 0:
                baselines[bsl_pthname_current_base] = preprocessed_fec.baselines
                video_microscopy[bsl_pthname_current_base] = preprocessed_fec.video_microscopy
            else:
                print(error_handling_dict[errcod])

        else:

            precomputed_results_dict = {"baselines": baselines[bsl_pthname_current_base],
                                        "video_microscopy": video_microscopy[bsl_pthname_current_base]}

            preprocessed_fec = pre.PiezoTracking(i,
                                                  bsl_pthname_current,
                                                  1,
                                                  downfact,
                                                  bsl_pol_order,
                                                  crange,
                                                  opt_dict,
                                                  video_trappos_regression_bsl_mdl_results = precomputed_results_dict
                                                  )
            preprocessed_fec.preprocess()

        # Wrap results into a dict
        fecs[fec_basename] = preprocessed_fec

#%% PLOTTING VIDEOMICROSCOPY RESULTS

if opt_dict["apply_dist_offset"] or opt_dict["video_trap_regress_cal"]:
    for i in video_microscopy.keys():
        vm_obj = video_microscopy[i]
        fplt.quick_plotter (vm_obj.ds_trappos1x_trim,
                            vm_obj.cdist1_trim,
                            vm_obj.pred_dist,
                            "Trap position 1x (um)",
                            "Camera distance (um)",
                            f"preprocess/video_microscopy/{i}_video_microscopy.png" ,
                            0.2,
                            annotations = [
                                              str(vm_obj.video_trap_regress_obj.params),
                                              "R2= " + str(vm_obj.video_trap_regress_R2),
                                              i
                                              ],
                             save = True
                                )
        
#%% PLOTTING BASELINE RESULTS

for i in baselines.keys():
    bsl_obj = baselines[i]

    #baseline fit
    fplt.quick_plotter ([bsl_obj.baseline_trappos1xds,bsl_obj.baseline_trappos1xds],
                        [bsl_obj.baseline_force1xds,bsl_obj.baseline_force2xds],
                        [bsl_obj.pred_bsl1ds,bsl_obj.pred_bsl2ds],
                        "Trap position 1x (um)",
                        "Baseline Force (pN)",
                        f"preprocess/baselines/fits/{i}_baseline_fit.png" ,
                        0.04,
                        annotations = ["Baseline pol order = " + str(bsl_obj.bsl_pol_degree),
                                       i],
                        save = True)

    #baseline residuals
    fplt.quick_plotter (bsl_obj.baseline_trappos1xds[::smoothening_factor],
                        bsl_obj.res_bsl_f1xds[::smoothening_factor],
                        None,
                       "Predicted baseline force (pN)",
                       "Residuals (pN)",
                       f"preprocess/baselines/residuals/{i}_baseline1_residuals.png",
                       0.4,
                       annotations = ["Baseline pol order = " + str(bsl_obj.bsl_pol_degree),
                                      i,
                                      "Baseline 1 residuals"],
                       guideline = True,
                       save = True)

    fplt.quick_plotter (bsl_obj.baseline_trappos1xds[::smoothening_factor],
                        bsl_obj.res_bsl_f2xds[::smoothening_factor],
                        None,
                       "Predicted baseline force (pN)",
                       "Residuals (pN)",
                       f"preprocess/baselines/residuals/{i}_baseline2_residuals.png",
                       0.4,
                       annotations = ["Baseline pol order = " + str(bsl_obj.bsl_pol_degree),
                                      i,
                                      "Baseline 2 residuals"],
                       guideline = True,
                       save = True)

    # corrected baseline forces
    smoothed_baseline_corrected_force_1xds = pre.CK_filtfilt(bsl_obj.baseline_corrected_force_1xds, smoothening_factor, decimate = False)
    smoothed_baseline_corrected_force_2xds = pre.CK_filtfilt(bsl_obj.baseline_corrected_force_2xds, smoothening_factor, decimate = False)

    fplt.quick_plotter ([bsl_obj.baseline_trappos1xds[smoothening_factor:-smoothening_factor],bsl_obj.baseline_trappos1xds[smoothening_factor:-smoothening_factor]],
                        [bsl_obj.baseline_corrected_force_1xds[smoothening_factor:-smoothening_factor],bsl_obj.baseline_corrected_force_2xds[smoothening_factor:-smoothening_factor]],
                        [smoothed_baseline_corrected_force_1xds[smoothening_factor:-smoothening_factor],smoothed_baseline_corrected_force_2xds[smoothening_factor:-smoothening_factor]],
                       "Trap position 1x (um)",
                       "Corrected Baseline Force (pN)",
                       f"preprocess/baselines/corrected_baseline_forces/{i}_baseline_corrected_forces.png",
                       0.04,
                       annotations = ["Baseline pol order = " + str(bsl_obj.bsl_pol_degree),
                                      i],
                       guideline = True,
                       save = True)

#%%
for i in fecs.keys():
    fec_obj = fecs[i]
    errcod = fec_obj.error_code
    if errcod == 0:
        d_3125 = fec_obj.fec_molextds
        f_3125 = fec_obj.fec_diffFds
        d_125 = pre.CK_filtfilt(d_3125, smoothening_factor)
        f_125 = pre.CK_filtfilt(f_3125, smoothening_factor)

        fplt.plot_individual_fec(d_125,
                                 f_125,
                                 i,
                                 "black",
                                 [19,21],
                                 [-2, 30],
                                 "preprocess/preprocessed_pics/" + i + ".png",
                                 [],
                                 distances_hf=d_3125,
                                 forces_hf=f_3125,
                                 scatter=False,
                                 annot=False,
                                 savefig=True,
                                 rupture_index=False,
                                 lcstates=False,
                                 states_end_indexes=False,
                                 lc_trajectory=False)

#%%
if opt_dict["apply_dist_offset"] or opt_dict["video_trap_regress_cal"]:
    video_microscopy_df = []
    analyzed_baselines = []

    for i in fecs.keys():
        # Video microscopy results are stored per fec object.
        # This is redundant as there is only one video microscopy result
        # per baseline (fecs from same baseline will have the same video_miscroscopu
        # results/object. To fix this (quick and dirty) we will parse the baseline
        # name per fec and stored in a list. If a fec has a baseline that is already
        # on a list (meaning another fec with the same baseline was already analyzed_
        # we will skip storing the results.
        _, baseline_id = gen_baseline_name (i)

        if baseline_id not in analyzed_baselines:
            fec_obj = fecs[i]
            video_microscopy_obj = fec_obj.video_microscopy

            tmpdf = pd.DataFrame()
            # populating columns
            tmpdf["trappos1x"] = video_microscopy_obj.ds_trappos1x_trim
            tmpdf["bead_position_1"] = video_microscopy_obj.bead_position_1_trim
            tmpdf["bead_position_2"] = video_microscopy_obj.bead_position_2_trim
            tmpdf["bead1_diameter"] = video_microscopy_obj.bead1_diameter
            tmpdf["bead2_diameter"] = video_microscopy_obj.bead2_diameter
            tmpdf["bead_surface-to_surface_distance"] = video_microscopy_obj.cdist1_trim
            tmpdf["predicted_distance"] = video_microscopy_obj.pred_dist
            
            coeffs = video_microscopy_obj.video_trap_regress_obj.params
            for idx, val in enumerate(coeffs):
                tmpdf[f"coeff_{idx}"] = val

            tmpdf["R2"] = video_microscopy_obj.video_trap_regress_R2
            tmpdf["baseline_id"] = baseline_id
            video_microscopy_df.append(tmpdf)
            analyzed_baselines.append(baseline_id)

    video_microscopy_df = pd.concat(video_microscopy_df)
    video_microscopy_df.to_csv("preprocess/video_microscopy/" + folder.replace("/","") + "video_microscopy.csv", index = False)


#%% ORGANIZE BASELINE DATA IN DF

baseline_rename_map = {
    'bsl_pol_degree': "pol_degree",
    'coeff_bsl1': "coeff_bsl1",
    'coeff_bsl2': "coeff_bsl2",
    'baseline_trappos1xds': "trappos1x",
    'baseline_force1xds': "force1x",
    'pred_bsl1ds': "predicted_force1x",
    'baseline_corrected_force_1xds': "baseline_corrected_force1x",
    'res_bsl_f1xds': "baseline1_residuals",
    'baseline_force2xds': "force2x",
    'pred_bsl2ds': "predicted_force2x",
    'baseline_corrected_force_2xds': "baseline_corrected_force2x",
    'res_bsl_f2xds': "baseline2_residuals",
    'baseline_timeds': "relative_time",
    'baseline_id': "baseline_id"
}


baselines_df = []
for i in baselines.keys():
    bsl_object = baselines[i]
    bsl_dict = bsl_object.baselineds_dict
    bsl_dict = {k: v for k, v in bsl_dict.items() if k not in ["f1_offset", "f2_offset"]}
    bsl_dict["baseline_id"] = i
    
    # --- NEW: rename keys BEFORE writing ---
    bsl_dict = rename_dict_keys(bsl_dict, baseline_rename_map)
    baselines_df.append(bsl_dict)

# your old sorted_cols
baseline_sorted_cols = ['relative_time','trappos1x','force1x','predicted_force1x',
                        'baseline1_residuals','baseline_corrected_force1x',
                        'coeff_bsl1','force2x', 'predicted_force2x','baseline2_residuals',
                        'baseline_corrected_force2x','coeff_bsl2', 'pol_degree',
                        'baseline_id']

stream_dicts_to_csv(
    baselines_df,
    "preprocess/baselines/" + folder.replace("/","") + "_baselines.csv",
    sorted_columns=baseline_sorted_cols)

#%% ORGANIZE PARAMETERS DATA

calpars_df = []

for i in baselines.keys():
    bsl_object = baselines[i]
    calpars = bsl_object.calpars["bead1"].__dict__["data"] # for pylake 1.6.2
    calpars = pd.DataFrame(calpars,index=[0])
    calpars["trap"] = "trap1"
    calpars["baseline_id"] = i
    calpars_df.append(calpars)

    calpars = bsl_object.calpars["bead2"].__dict__["data"] # for pylake 1.6.2
    calpars = pd.DataFrame(calpars,index=[0])
    calpars["trap"] = "trap2"
    calpars["baseline_id"] = i
    calpars_df.append(calpars)

fecs03 = glob.glob("rawdat/*fec03.h5")

for i in fecs03:
    f = lk.File(i)
    
    calpars = f.force1x.calibration[0].__dict__["data"]
    calpars = pd.DataFrame(calpars,index=[0])
    calpars["trap"] = "trap1"
    calpars["baseline_id"] = os.path.basename(i).replace(".h5","")
    calpars_df.append(calpars)

    calpars = f.force2x.calibration[0].__dict__["data"]
    calpars = pd.DataFrame(calpars,index=[0])
    calpars["trap"] = "trap2"
    calpars["baseline_id"] = os.path.basename(i).replace(".h5","")
    calpars_df.append(calpars)


calpars_df = pd.concat(calpars_df)
calpars_df.to_csv("preprocess/" + folder.replace("/","") + "_calpars.csv", index = False)

#%% ORGANIZE HF KEPT KEPT IN DF..

fec_rename_map = {
    'fec_timeds': "relative_time",
    'fec_trappos1xds': "trappos1x",
    'fec_corrected_force_1xds': "force1x",
    'fec_corrected_force_2xds': "force2x",
    'fec_molextds': "molext",
    'fec_bead_disp1ds': "bead1_disp",
    'fec_bead_disp2ds': "bead2_disp",
    'fec_diff_bead_dispds': "diff_bead_disp",
    'fec_diffFds': "diffF",
    'fec_cal_trappos1xds': "cal_trappos1x",
    'fec_cal_molextds': "cal_molext",
    'fec_id': "fec_id",
    'date': "date",
    'bead_set': "bead_set",
    'mol_id': "mol_id",
    'curve_type': "curve_type",
    'curve_number': "curve_number",
    'error_code': "error_code",
}

fecs_hf_df = []
for i in fecs.keys():
    fec_obj = fecs[i]
    errcod = fec_obj.error_code

    if errcod == 0:
        fec_dict = fec_obj.fecds_dict
        fec_dict = {k: v for k, v in fec_dict.items() if k not in ["fec_force1xds", "fec_force2xds"]}
        fec_meta = parse_metadata(i)
        fec_final = fec_dict | fec_meta
        fec_final["error_code"] = errcod
        fec_final["fec_id"] = i
        # --- NEW: rename keys BEFORE writing ---
        fec_final = rename_dict_keys(fec_final, fec_rename_map)
        fecs_hf_df.append(fec_final)

fec_sorted_cols = ['relative_time', 'trappos1x', 'cal_trappos1x', 'molext', 'cal_molext',
            'bead1_disp','bead2_disp', 'diff_bead_disp','force1x', 'force2x',
            'diffF','bead_set', 'mol_id','curve_type', 'curve_number', 'date',
            "error_code",'fec_id']

stream_dicts_to_csv(
    fecs_hf_df,
    "preprocess/" + folder.replace("/","") + "_preprocessed.csv",
    sorted_columns=fec_sorted_cols
)

#%% ORGANIZE DISCARDED fecs IN DF

# Create df for discarded basedon the lists of failures
df_discarded = pd.DataFrame(columns = fec_sorted_cols)
df_discarded["error_code"] = None

for i in fecs.keys():
    fec_obj = fecs[i]
    errcod = fec_obj.error_code

    if errcod != 0:
        tmplist = [0 for x in range(len(df_discarded.columns)-2)] + [i] + [errcod]
        df_discarded.loc[len(df_discarded)] = tmplist

df_discarded.to_csv ("preprocess/" + folder.replace("/","") + "_preprocessed_discarded.csv", index = False)

#%% SAVING PARAMETERS USED AND DATE OF PREPROCESSING RUN (METADATA)
now = datetime.now()

with open ("preprocess/" + folder.replace("/","") + "_preprocessing_metadata.txt", "w+") as f:
    f.write (folder.replace("/","") + " was collected at an original frequency of " + str(frequency) + " Hz \n" )
    f.write(folder.replace("/","") + " was preprocessed on: " + str(now) + "\n")
    f.write("Bead 1 size was: " + str(opt_dict["bead1_diameter"]) + " um\n")
    f.write("Bead 2 size was: " + str(opt_dict["bead2_diameter"]) + " um\n")
    f.write("The following parameters were used:\n")
    f.write("Force 1x corrections were : " + str(correct_f1x_instability) + "\n")
    f.write("Spatial drift corrections were : " + str(correct_spatial_drift) + "\n")
    f.write("Video microscopy polynomial order : " + str(video_microscopy_pol_order) + "\n")
    f.write("baseline pol order = " + str(bsl_pol_order) + "\n")
    f.write("Downsamping factor = " + str(downfact) + "\n")
    f.write("Fec frequency = " + str(target_downsampled_frequency) + " Hz \n")
    f.write("The following options were on (True):\n")
    f.write("correct_force_offset: " + str(opt_dict["correct_force_offset"]) + "\n")
    f.write("apply_dist_offset: " + str(opt_dict["apply_dist_offset"]) + "\n")
    f.write("video_trap_regress_cal: " + str(opt_dict["video_trap_regress_cal"]) + "\n")
    f.write("filter type: " + str(opt_dict["filter_type"]) + "\n\n")




