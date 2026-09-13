import lumicks.pylake as lk
import pandas as pd
import sys, re, os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
#%%
sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.corrections import CorrectSpatialDrift

#%%

def get_fd_from_df(fec_id,df):
    tmpdf = df[df["fec_id"] == fec_id]
    fec_dict = {}
    fec_dict["t"] = tmpdf.relative_time.to_numpy()
    fec_dict["d"] = tmpdf.molext.to_numpy()
    fec_dict["f"] = tmpdf.diffF.to_numpy()
    fec_dict["curve_type"] = tmpdf.curve_type.to_list()[0]
    return(fec_dict)

def get_trace_base(trace):
    underscore_indexes = [x.start() for x in re.finditer("_",trace)]
    return(trace[:underscore_indexes[-1]])

def open_passive_trace(trace_base,passive_trace):
    for mode in ("spm","dct"):
        trace_name = f"{trace_base}_{mode}{passive_trace}"
        path = f"rawdat/{trace_name}.h5"

        if os.path.exists(path):
            return(lk.File(path),trace_name)

    return(None,None)

def get_camera_trappos_slope_factor(baseline_h5_file,channel="1x"):
    baseline = lk.File(baseline_h5_file)

    ds_trappos1x,bead_position1 = baseline["Trap position"]["1X"].downsampled_like(
        baseline["Bead position"]["Bead 2 X"]
    )
    
    ds_trappos1x = ds_trappos1x.data
    bead_position1 = bead_position1.data
    
    ds_trappos1x,bead_position2 = baseline["Trap position"]["1X"].downsampled_like(
        baseline["Bead position"]["Bead 1 X"]
    )
    
    bead_position2 = bead_position2.data
    
    mask = (bead_position2 != 0) & (bead_position1 != 0) & (ds_trappos1x != 0)
    ds_trappos1x = ds_trappos1x[mask]
    bead_position1 = bead_position1[mask]
    bead_position2 = bead_position2[mask]
    
    if channel == "1x":
        distance1 = bead_position1 - bead_position2[0]
    elif channel == "2x":
        distance1 = bead_position1[0] - bead_position2
    elif channel == "both":
        distance1 = bead_position1 - bead_position2
    else:
        raise ValueError("Select a valid channel (1x/2x/both)")
    
    fit = np.polyfit(ds_trappos1x,distance1,deg=1)
    pdistance1 = np.polyval(fit,ds_trappos1x)
    
    return(ds_trappos1x,distance1,pdistance1,fit)

def gen_baseline_name(trace_id):
    underscore_ids = [x.start() for x in re.finditer("_",trace_id)]
    dte = trace_id[:underscore_ids[0]]
    bset = trace_id[underscore_ids[0]+1:underscore_ids[1]]
    
    match = re.search(r"(?:spm|dct)(\d{2})$",trace_id)
    
    if match is None:
        return("Error")
    
    number = match.group(1)
    
    if number == "01":
        bsl_number = "01"
    elif number == "02":
        bsl_number = "02"
    else:
        return("Error")
    
    bsl_pthname_current = f"{dte}_{bset}_fec{bsl_number}.h5"
    bsl_pthname_current_base = os.path.basename(bsl_pthname_current).replace(".h5","")

    return(bsl_pthname_current,bsl_pthname_current_base)

#%% PARAMETERS

folder = "260430_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_20nMChd1_1mMATP_1umBeads_b84_drift"

drift_model_order = 3
pause_edge_trim = 1
only_positive_drift = True

channels_for_corr = "1x"

trappos_segment_threshold = 0.001
trappos_segment_min_len = 5
trappos_segment_min_vel = 9

excluded = ["260513_04_01_spm02"]

#%%

drift_plot_dir = "preprocess/preprocess_correction_pics/drift_global_fits"
corrected_plot_dir = "preprocess/preprocess_correction_pics/corrected_molext"

os.makedirs(drift_plot_dir,exist_ok=True)
os.makedirs(corrected_plot_dir,exist_ok=True)

#%% LOADING PREPROCESSED DATA

df = pd.read_csv("preprocess/" + folder + "_preprocessed.csv")
baselinesdf = pd.read_csv("preprocess/baselines/" + folder + "_baselines.csv")
fecs = sorted(set(df.fec_id))
traces_base = sorted(set(get_trace_base(x) for x in fecs))
baselines = sorted(set(baselinesdf.baseline_id))

#%% put data on dict

fecs_dict = {}

for i in fecs:
    fecs_dict[i] = get_fd_from_df(i,df)

#%% getting camera_trappos_slope_factor

# Camera distance = slope_factor * trap-position distance
# Leave as 1 to assume identical spatial calibration

baselines_dict = {}

for i in baselines:
    ds_trappos1x,distance1,pdistance1,fit = get_camera_trappos_slope_factor(
        f"rawdat/{i}.h5",
        channel=channels_for_corr
    )
    
    tmpdict = {}
    tmpdict["camera_trappos_slope_factor"] = fit[0]
    baselines_dict[i] = tmpdict
    
    plt.scatter(ds_trappos1x,distance1,s=5,color="gray")
    plt.plot(ds_trappos1x,pdistance1,linewidth=2,linestyle="--",color="black",label=f"m={fit[0]}")
    plt.ylabel("Camera distance (um)")
    plt.xlabel("Trap position (um)")
    plt.legend()
    plt.savefig(f"preprocess/video_microscopy/{i}_video_microscopy.png")
    plt.show()
    
#%%

corrected_traces = set()

def apply_constant_offset(trace_name,offset):
    if trace_name not in fecs_dict:
        return

    fecs_dict[trace_name]["corrected_d"] = fecs_dict[trace_name]["d"] + offset
    corrected_traces.add(trace_name)


for i in traces_base:
    for passive_trace in ("01","02"):

        h5file,translocation_trace_name = open_passive_trace(i,passive_trace)

        if h5file is None:
            print(f"passive {passive_trace} not found for basename {i}")
            continue
        if translocation_trace_name in excluded:
            continue

        t = fecs_dict[translocation_trace_name]["t"]
        raw_d = fecs_dict[translocation_trace_name]["d"]

        if translocation_trace_name in corrected_traces:
            d = fecs_dict[translocation_trace_name]["corrected_d"]
        else:
            d = raw_d

        _,bsl = gen_baseline_name(translocation_trace_name)
        camera_trappos_slope_factor = baselines_dict[bsl]["camera_trappos_slope_factor"]

        correct_obj = CorrectSpatialDrift(
            h5file,
            t,
            d,
            drift_model_order=drift_model_order,
            pause_edge_trim=pause_edge_trim,
            camera_trappos_slope_factor=camera_trappos_slope_factor,
            channels_for_drift_estimation=channels_for_corr,
            only_positive_drift=only_positive_drift,
            trappos_mode_threshold=trappos_segment_threshold,
            trappos_mode_min_run_len=trappos_segment_min_len,
            trappos_mode_min_velocity_threshold=trappos_segment_min_vel,
            min_drift_velocity=-1.5e-5
        )

        correct_obj.run()

        # Spatial drift fit
        fig,ax = plt.subplots()

        ax.plot(
            correct_obj.ds_time[correct_obj.passive],
            correct_obj.spatial_drift[correct_obj.passive],
            ".",
            markersize=4,
            label = translocation_trace_name
        )

        for j,offset in enumerate(correct_obj.segment_offsets):
            mask = correct_obj.passive_segment_ids == j

            ax.plot(
                correct_obj.ds_time[mask],
                correct_obj.spatial_drift_fit[mask] + offset,
                color="black",
                linewidth=2
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spatial drift (um)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(
            f"{drift_plot_dir}/{translocation_trace_name}_drift_fit.png",
            dpi=300
        )
        plt.show()
        plt.close(fig)

        # Corrected molecular extension
        fig,ax = plt.subplots()

        ax.plot(t,d,label="input_molext")
        ax.plot(t,correct_obj.corrected_molext,label="corrected_molext")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative molecular extension (um)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            f"{corrected_plot_dir}/{translocation_trace_name}_corrected_molext.png",
            dpi=300
        )
        plt.show()
        plt.close(fig)

        fecs_dict[translocation_trace_name]["corrected_d"] = correct_obj.corrected_molext
        corrected_traces.add(translocation_trace_name)

        # Total accumulated correction relative to the original trace
        cumulative_offset = correct_obj.corrected_molext[-1] - raw_d[-1]

        if passive_trace == "01":
            apply_constant_offset(f"{i}_fecS01",cumulative_offset)
            apply_constant_offset(f"{i}_fecS02",cumulative_offset)
            apply_constant_offset(f"{i}_fecS03",cumulative_offset)
            apply_constant_offset(f"{i}_spm02",cumulative_offset)
            apply_constant_offset(f"{i}_dct02",cumulative_offset)

        elif passive_trace == "02":
            apply_constant_offset(f"{i}_fecS03",cumulative_offset)

#%% BUILD CORRECTED DATAFRAME

corrected_df = []

for i in fecs:
    tmpdf = df[df.fec_id == i].copy()

    if i in corrected_traces:
        corrected_d = fecs_dict[i]["corrected_d"]
        drift = corrected_d - tmpdf["molext"].to_numpy()

        tmpdf["trappos1x"] = tmpdf["trappos1x"].to_numpy() + drift
        tmpdf["molext"] = corrected_d

    corrected_df.append(tmpdf)

corrected_df = pd.concat(corrected_df)

#%%

corrected_df.to_csv(
    f"preprocess/{folder}_preprocessed_corrected.csv",
    index=False
)

#%% METADATA

now = datetime.now()

with open("preprocess/" + folder.replace("/","") + "_correction_metadata.txt","w+") as f:
    f.write(f"Correction was performed on {now}\n")
    f.write("The following parameters were used\n")
    f.write(f"Drift model order = {drift_model_order}\n")
    f.write(f"Pause edge trim = {pause_edge_trim} s\n")
    f.write(f"Only positive drift = {only_positive_drift}\n")
    f.write(f"Channels for correction = {channels_for_corr}\n")
    f.write(f"Trap-position segmentation threshold = {trappos_segment_threshold}\n")
    f.write(f"Trap-position minimum run length = {trappos_segment_min_len}\n")
    f.write(f"Trap-position minimum velocity = {trappos_segment_min_vel}\n")
    f.write("Camera/trap slope factors:\n")

    for bsl,tmpdict in baselines_dict.items():
        f.write(f"{bsl}: {tmpdict['camera_trappos_slope_factor']}\n")