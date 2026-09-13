#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:19:21 2026

@author: einstein
"""

def gen_qual_control_df (passive_subdict, trace_id):
    segment_names = [x for x in passive_subdict.keys() if "seg" in x]
    segment_names = [x for x in segment_names if passive_subdict[x]["mode"] == "passive"]
    
    
    k1_stiffness_change_ratio = passive_subdict["k1_stiffness_change_ratio"]
    k2_stiffness_change_ratio = passive_subdict["k2_stiffness_change_ratio"]
    keq_stiffness_change_ratio = passive_subdict["keq_stiffness_change_ratio"]
    lctrace_diff_std_estimate = passive_subdict["lctrace_diff_std_estimate"]
    
    k1_stiffness_change_ratio_col = [k1_stiffness_change_ratio for x in segment_names]
    k2_stiffness_change_ratio_col = [k2_stiffness_change_ratio for x in segment_names]
    keq_stiffness_change_ratio_col = [keq_stiffness_change_ratio for x in segment_names]
    lctrace_diff_std_estimate_col = [lctrace_diff_std_estimate for x in segment_names]
    
    dLc_obs_from_diff_channel_col = [passive_subdict[x].get("dLc_obs_from_diff_channel", np.nan) for x in segment_names]
    dLc_pred_from_diff_channel_col = [passive_subdict[x].get("dLc_pred_from_diff_channel",np.nan) for x in segment_names]
    percent_error_from_diff_channel_col = [passive_subdict[x].get("percent_error_from_diff_channel", np.nan) for x in segment_names]
    correction_factor_from_diff_channel_col = [passive_subdict[x].get("correction_factor_from_diff_channel", np.nan) for x in segment_names]
    
    out = pd.DataFrame()
    out["percent_error_from_diff_channel"] = percent_error_from_diff_channel_col
    out["correction_factor_from_diff_channel"] = correction_factor_from_diff_channel_col
    out["dLc_pred_from_diff_channel"] = dLc_pred_from_diff_channel_col
    out["dLc_obs_from_diff_channel"] = dLc_obs_from_diff_channel_col
    out["segment_id"] = segment_names
    out["k1_stiffness_change_ratio"] = k1_stiffness_change_ratio_col
    out["k2_stiffness_change_ratio"] = k2_stiffness_change_ratio_col
    out["keq_stiffness_change_ratio"] = keq_stiffness_change_ratio_col
    out["lctrace_diff_std_estimate_col"] = lctrace_diff_std_estimate_col
    out["trace"] = trace_id
    
    return(out)

def store_dlc_metrics(segment, dLc_obs, dLc_pred, suffix="from_diff_channel"):
    segment[f"dLc_obs_{suffix}"] = dLc_obs
    segment[f"dLc_pred_{suffix}"] = dLc_pred
    segment[f"percent_error_{suffix}"] = 100 * (dLc_obs - dLc_pred) / dLc_pred
    segment[f"correction_factor_{suffix}"] = dLc_pred / dLc_obs


def compute_dlc_obs(d, f):
    lc = ana.quick_per_point_Lc(d, f, *WLCp)
    return lc[-1] - lc[0]


def find_closest_force_idx(f, mask, refF=7):
    """
    Returns the global index closest to refF within mask, plus the force value.
    """
    global_idxs = np.flatnonzero(mask)
    local_idx = np.argmin(np.abs(f[mask] - refF))
    global_idx = global_idxs[local_idx]
    force_val = f[global_idx]
    return global_idx, force_val


def compute_passive_to_passive_metrics(trace, dur1, dur2, refF=7):
    f = trace["flf_trim"]
    d = trace["dlf_trim"]
    trappos = trace["trappos1xlf_trim"]

    tlf = np.arange(len(f)) / FS

    mask1 = (tlf >= dur1[0]) & (tlf < dur1[1])
    mask2 = (tlf >= dur2[0]) & (tlf < dur2[1])

    idx1, f1 = find_closest_force_idx(f, mask1, refF)
    idx2, f2 = find_closest_force_idx(f, mask2, refF)

    if (abs(f1 - refF) > FORCE_TOL) or (abs(f2 - refF) > FORCE_TOL):
        return np.nan, np.nan, np.nan

    dExt = trappos[idx2] - trappos[idx1]

    eps = ana.get_normalized_extension(
        refF,
        trans_params["Lp"],
        S=trans_params["S"],
    )

    dLc_pred = dExt / eps

    d_slice = d[idx1:idx2]
    f_slice = f[idx1:idx2]
    dLc_obs = compute_dlc_obs(d_slice, f_slice)

    return dExt, dLc_pred, dLc_obs


def compute_base_to_fecS03_metrics(trace_name, base_trappos, base_d, base_f, fecS03_name):
    """
    Handles the repeated logic where a passive/base segment is merged into fecS03,
    then FindTransExtensions is used to get dLc_pred.
    """
    ind_offset = len(base_d)

    trans_fecS03 = ind_offset + np.asarray(fecs_dict[fecS03_name]["trans_idxs"])
    trans_passive = np.array([0, ind_offset - 1])
    trans_all = np.concatenate([trans_passive, trans_fecS03])

    s03 = fecs_dict[fecS03_name]

    ruptureF = np.nan
    dLc_pred = np.nan
    merged_d = None
    merged_f = None

    for tfs03 in trans_fecS03:
        merged_trappos = np.concatenate([
            base_trappos,
            s03["trappos1xlf"],
        ])[:tfs03 + 1]

        merged_d = np.concatenate([
            base_d,
            s03["dlf_trim"],
        ])[:tfs03 + 1]

        merged_f = np.concatenate([
            base_f,
            s03["flf_trim"],
        ])[:tfs03 + 1]

        trans_obj = ana.FindTransExtensions(
            trace_name,
            merged_trappos,
            merged_f,
            trans_all,
            trans_params=trans_params,
            debug=True,
        )

        ruptureF, x1, x2, dExt, dLc_pred = trans_obj.find_path_extension(
            (trans_passive[0], tfs03),
            forward=True,
            sanity_test=False,
            bypass_sl=True,
        )

        if not np.isnan(dLc_pred):
            break

    if np.isnan(dLc_pred):
        return np.nan, np.nan

    idx = np.argmin(np.abs(s03["flf_trim"] - ruptureF)) + ind_offset
    dLc_obs = compute_dlc_obs(merged_d[:idx + 1], merged_f[:idx + 1])

    return dLc_pred, dLc_obs

#%%
#%% Compute Errors on distances

trans_params = {
    "nfit": 20,
    "navg": 3,
    "FAvgRange": 0.05,
    "FFitRange": 1,
    "padding": 3,
    "Lp": 50,
    "S": 1200,
}

REF_F = 9
FORCE_TOL = 0.1
FS = 125

for i in traces_base:
    
    print(i)

    fecS03_name = f"{i}_fecS03"

    for c in passive:
        if i not in c:
            continue

        trace = fecs_dict[c]

        if "dct02" in c:
            dLc_pred, dLc_obs = compute_base_to_fecS03_metrics(
                trace_name=i,
                base_trappos=trace["trappos1xlf"],
                base_d=trace["dlf"],
                base_f=trace["flf"],
                fecS03_name=fecS03_name,
            )

            store_dlc_metrics(
                trace["segment_1"],
                dLc_obs=dLc_obs,
                dLc_pred=dLc_pred,
            )

        elif "spm02" in c:
            # Non-dct case: passive-to-passive corrections first
            seg_names = passive_segment_names(trace)
            seg_durations = [trace[name]["duration"] for name in seg_names]
    
            for seg_name, dur1, dur2 in zip(
                seg_names[:-1],
                seg_durations[:-1],
                seg_durations[1:],
            ):
                dExt, dLc_pred, dLc_obs = compute_passive_to_passive_metrics(
                    trace,
                    dur1,
                    dur2,
                    refF=REF_F,
                )
    
                if np.isnan(dLc_pred):
                    continue
    
                store_dlc_metrics(
                    trace[seg_name],
                    dLc_obs=dLc_obs,
                    dLc_pred=dLc_pred,
                )
    
            # Last passive segment goes to fecS03
            last_seg_name = seg_names[-1]
            last_duration = seg_durations[-1]
    
            tlf = np.arange(len(trace["flf_trim"])) / FS
            last_mask = (tlf >= last_duration[0]) & (tlf < last_duration[1])
    
            dLc_pred, dLc_obs = compute_base_to_fecS03_metrics(
                trace_name=i,
                base_trappos=trace["trappos1xlf"][last_mask],
                base_d=trace["dlf"][last_mask],
                base_f=trace["flf"][last_mask],
                fecS03_name=fecS03_name,
            )
    
            store_dlc_metrics(
                trace[last_seg_name],
                dLc_obs=dLc_obs,
                dLc_pred=dLc_pred,
            )
        else:
            pass

#%% get noise estimation

for i in passive:
    diffF = fecs_dict[i]["f"]
    lcdiff = fecs_dict[i]["lctrace_diff"][diffF>4]
    dlcdiff = np.diff(lcdiff)
    thr = np.quantile(np.abs(dlcdiff), 0.995)
    dlcdiff = np.clip(dlcdiff, -thr, thr)
    mad = np.median(np.abs(dlcdiff - np.median(dlcdiff)))
    noise = mad / 0.6745 / np.sqrt(2)
    fecs_dict[i]["lctrace_diff_std_estimate"] = noise
    
#%% compute stiffness changes

for i in passive:
    bsls = gen_baseline_name(i)
    k1_before = dfcalpars[ (dfcalpars.baseline_id == bsls[1]) & (dfcalpars["trap"] == "trap1") ]["kappa (pN/nm)"].to_numpy()[0]
    k2_before = dfcalpars[ (dfcalpars.baseline_id == bsls[1]) & (dfcalpars["trap"] == "trap2") ]["kappa (pN/nm)"].to_numpy()[0]
    keq_before = (1/k1_before + 1/k2_before)**-1
    
    try:
        k1_after = dfcalpars[ (dfcalpars.baseline_id == bsls[2]) & (dfcalpars["trap"] == "trap1") ]["kappa (pN/nm)"].to_numpy()[0]
        k2_after = dfcalpars[ (dfcalpars.baseline_id == bsls[2]) & (dfcalpars["trap"] == "trap2") ]["kappa (pN/nm)"].to_numpy()[0]
        keq_after = (1/k1_after + 1/k2_after)**-1
    except:
        k1_after = np.nan
        k2_after = np.nan
        keq_after = np.nan

    fecs_dict[i]["k1_stiffness_change_ratio"] = k1_after/k1_before
    fecs_dict[i]["k2_stiffness_change_ratio"] = k2_after/k2_before
    fecs_dict[i]["keq_stiffness_change_ratio"] = keq_after/keq_before
#%%

#%% assemble and store 
        
df_lc_quality_control = []
for i in passive:
    if ("dct02" in i) or ("spm02" in i):
        print(i)
        subdict = fecs_dict[i]
        tmpdf = gen_qual_control_df (subdict,i)
        df_lc_quality_control.append(tmpdf)
    
df_lc_quality_control = pd.concat(df_lc_quality_control, ignore_index=True)
df_lc_quality_control.to_csv(f"data_consolidation/{csv_basename}_lctraces_qual_control.csv", index = False)
