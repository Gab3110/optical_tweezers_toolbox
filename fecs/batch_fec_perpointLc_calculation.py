'''
Description  : This script works with the main dataframe generated as an
               output of the script batch_fec_postprocess. Using the aligned
               molext and diffF, it inverts a eWLC model to fit the contour
               lenght in a per-point basis.

Author       : Gabriel Jiménez-Avalos, PhD. student.
Affiliation  : T. C. Jenkins department of Biophysics. Johns Hopkins University.
Email        : gjimene5@jhu.edu
Version      : 3.1
Date         : 2025/04/06


'''


#%% DEFINE MODULES

import numpy as np
import pandas as pd
import os, sys
import concurrent.futures
import matplotlib.pyplot as plt
from datetime import datetime
import dill

#%% DEFINE CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
from tweezers_toolbox_modules.fitting import PerPointLc
from tweezers_toolbox_modules.analysis import quick_per_point_Lc

#%% MULTITHREADING FUNCTIONS

def calculator(fec_id, df_postprocess, WLCp, calculation_method = "perpoint_fit", var_component = 0):
    try:
        # Filter the dataframe to get the relevant data for this fec_id
        tmpdf = df_postprocess[df_postprocess["fec_id"] == fec_id]
        ri = tmpdf.rupture_index.to_numpy()[0]
        d = tmpdf.molext_aln.to_numpy()[0:ri+1]
        f = tmpdf.diffF_aln.to_numpy()[0:ri+1]

        # Apply filtering to the data
        dlf = CK_filtfilt(d, 25)
        flf = CK_filtfilt(f, 25)

        if calculation_method == "perpoint_fit":
            # Initialize the model inside the parallel function to avoid serialization issues
            calculated_trace = PerPointLc(dlf, flf, *WLCp, var_lc_component=0,
                                          lc_bounds = [0.001, 2], distance_offset = False,
                                          force_offset = False, approx = "marko")
            calculated_trace.run()
    
    
            # Prepare the result dictionary
            fec_data = {
                "perpointlctrace": calculated_trace.trace,
                "molext_aln": dlf,
                "diffF_aln": flf
            }
        elif calculation_method == "quick_lc":
            calculated_trace = quick_per_point_Lc(dlf, flf, *WLCp)
            # Prepare the result dictionary
            fec_data = {
                "perpointlctrace": calculated_trace,
                "molext_aln": dlf,
                "diffF_aln": flf
            }
        else:
            print("Invalid calculation method.")
            return fec_id, None

        return fec_id, fec_data
    except Exception as e:
        print(f"Error processing fec_id {fec_id}: {e}")
        return fec_id, None

# STUPID MAIN FUNCTION FOR COMPATIBILITY WITH MACOS-WINDOWS. OF COURSE
# LINUX DOESNT NEED IT
def main(fecs, df_postprocess, WLCp,num_processors="all"):
    """
    Main function to process the fec data in parallel.
    You can specify the number of processors to use.

    Parameters:
    - num_processors (int or None): Number of processors to use. If None, defaults to the number of available CPUs.
    """
    if num_processors == "all":
        num_processors = os.cpu_count()  # Use all available processors if none specified

    # Create the pool of workers and process the fecs in parallel
    fec_dict = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        # Map the annotator function directly without using a lambda
        results = executor.map(calculator, fecs, [df_postprocess] * len(fecs), [WLCp] * len(fecs), [calculation_method]*len(fecs), [var_component] * len(fecs) )

        # Collecting results into fec_dict
        for fec_id, fec_data in results:
            fec_dict[fec_id] = fec_data

    print("Processing complete.")
    return fec_dict

#%% DEFINING GENERAL VARIABLES

# Define folder variable as the name of the folder containing the data (expt name)
folder = "240908_1016-601-998H_STdimerTArich_90mMKCl_2500uMMgCl2_3xBiotin"

# Define a reference eWLC model. If quick_lc is used, put None in the contour lenght
# of the WLC that is going to vary. For example, if your system is described by
# two worm like chains (one for DNA handles and one for protein) and you are
# unfolding proteins, the variable should be ([Lp_DNA, Lc_DNA, St_DNA],[Lp_prot, None]).
# Note that your system could be described by more than two worm like chains, 
# and could be a mixture of extensible or inextensible models. You just need to 
# add more lists to the tuple (i. e. ([Lp,Lc,St],[Lp2,Lc2,St2],[Lp3,Lc3],[Lp4,Lc4,St4]))
WLCp = ([40, None, 500], [0.65,0.01908])

# select calculation method. Available: 
# - perpoint_fit: uses lumicks function to calculate per point Lc by fitting
#                 a wlc on a per-point basis.
# - quick_lc: (1) predicts extensions of all non variable wlc components using
#             the force points on the data. 
#             (2) Substracts these extensions from the data. 
#             (3) Calculates fractional extension (ε) for the remaining
#             wlc (with unknown lc) for each force point on the data. 
#             (4) divide the substracted extension data with the corresponding 
#             ε(F) value to get the per point Lc. 
# 
calculation_method = "quick_lc"

# if using a perpoint fit method, specify the index of the WLC component 
# (as it appears on the tuple) that is going to vary. For instance, for protein
# unfolding with DNA handles: var_component = 1.
if calculation_method == "perpoint_fit":
    var_component = 0
else:
    var_component = None

fec_dict = {}

#%% MAKE NECESSARY FOLDERS

try:
    os.makedirs("per_pointLc/per_pointLc_traces_pics/")
except:
    pass

#%% LOADING DATAFRAME

df_postprocess = pd.read_csv(f"postprocess/{folder}_postprocessed.csv")
fecs = list(set(df_postprocess.fec_id))

#%% RUN THE ANNOTATOR

if __name__ == '__main__':
    # Calculate Lc traces for all fecs
    fec_dict = main(fecs, df_postprocess, WLCp, num_processors="all")

    # Saving results in pandas DF
    lf_lctrace_df = []

    for i in fecs:
        tmpdf = pd.DataFrame()
        tmpdf["relative_time"] = np.arange(0, len(fec_dict[i]["molext_aln"])*1/125, 1/125)
        tmpdf["molext_aln"] = fec_dict[i]["molext_aln"]
        tmpdf["diffF_aln"] = fec_dict[i]["diffF_aln"]
        tmpdf["Lc_trace"] = fec_dict[i]["perpointlctrace"]
        tmpdf["fec_id"] = i
        lf_lctrace_df.append(tmpdf)
    lf_lctrace_df = pd.concat(lf_lctrace_df)
    lf_lctrace_df.to_csv("per_pointLc/" + folder + "_lctraces.csv", index = False)

    # Plotting results and save the plots
    for i in fecs:
        plt.ylim(0,2)
        t = np.arange(0, len(fec_dict[i]["molext_aln"])*1/125, 1/125)
        trace = fec_dict[i]["perpointlctrace"]
        plt.plot(t, trace)
        plt.savefig(f"per_pointLc/per_pointLc_traces_pics/{i}.png", dpi = 300)
        plt.show()

    # Saving metadata
    now = datetime.now()

    with open (f"per_pointLc/{folder}_postprocessing_metadata.txt", "w+") as f:
        f.write(folder + " per point Lc was calculated on: " + str(now) + "\n\n\n")
        f.write("The following eWLC parameters were used:\n")
        f.write(str(WLCp) + "\n")

    dill.dump_session("per_pointLc/" + folder + "_session.pkl")
