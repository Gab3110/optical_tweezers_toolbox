#%% CLEAN SESSION BEFORE START

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

#%%
import numpy as np
import pandas as pd
import sys, os, re, gc
from datetime import datetime
import copy
import matplotlib.pyplot as plt 
import ast

#%%

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt, bessel_filtfilt, butter_filtfilt
import tweezers_toolbox_modules.postprocessing as post
import tweezers_toolbox_modules.analysis as ana
from tweezers_toolbox_modules.general_utils import get_complement_intervals
from tweezers_toolbox_modules import plotting as fplt
from tweezers_toolbox_modules.general_utils import intervals_from_binary_labels

#%%
def get_pause_levels(pause_intervals, lc):
    pause_levels = []

    for start, end in pause_intervals:
        lc_pause_seg = lc[start:end+1]
        pause_levels.append(np.mean(lc_pause_seg))

    return np.asarray(pause_levels)


def get_pause_segments(pause_intervals, pause_levels):
    pause_segments = []

    for (start, end), pause_level in zip(pause_intervals, pause_levels):
        n = end - start + 1
        pause_segments.append(np.full(n, pause_level))

    return pause_segments


def rebuild_fit_regions_from_preserved_pauses(trace_subdict):
    """Build fitting inputs on a recalculated Lc while retaining curated pauses."""
    # intervals_from_binary_labels returns a list, while
    # get_complement_intervals expects an (N, 2) NumPy array. Reshape also
    # guarantees that an empty interval list has shape (0, 2).
    pause_intervals = np.asarray(
        trace_subdict["putative_LLP_intervals_idx"],
        dtype=int
    ).reshape(-1, 2)
    trace_subdict["putative_LLP_intervals_idx"] = pause_intervals

    lc = trace_subdict["lctrace_difflf"]

    pause_levels = get_pause_levels(pause_intervals, lc)
    pause_segments = get_pause_segments(pause_intervals, pause_levels)

    trace_subdict["putative_LLP_levels"] = pause_levels
    trace_subdict["putative_LLP_segments"] = pause_segments
    trace_subdict["KV_intervals"] = get_complement_intervals(
        pause_intervals,
        len(lc),
        segment_inclusive=True
    )


def fit_kv_segment_or_single_dwell(
    lcseg,
    PF,
    min_step_size,
    min_dwell_samples
):
    """Fit one KV segment, using its mean when no valid split exists."""
    lcseg = np.asarray(lcseg, dtype=float)

    if lcseg.ndim != 1 or len(lcseg) == 0:
        raise ValueError("A KV fitting interval must contain at least one sample.")

    if not np.all(np.isfinite(lcseg)):
        raise ValueError("A KV fitting interval contains non-finite Lc values.")

    single_dwell = np.full(len(lcseg), np.mean(lcseg))

    # Two retained dwells cannot exist unless each can contain the required
    # minimum number of samples. The correct fit is therefore one dwell.
    if len(lcseg) < 2 * min_dwell_samples:
        return single_dwell, (
            f"only {len(lcseg)} samples; fewer than the "
            f"{2 * min_dwell_samples} required for two dwells"
        )

    if np.ptp(lcseg) == 0:
        return single_dwell, "the Lc segment is constant"

    KVobj = ana.FindKVTrace(
        lcseg,
        PF=PF,
        min_step_size=min_step_size,
        min_dwell_samples=min_dwell_samples
    )

    try:
        KVobj.run()
    except IndexError as exc:
        # This exact error is raised by sfHMM when its candidate-split heap is
        # empty. It means that no admissible step was found, so the appropriate
        # result is the no-step, single-mean-dwell model.
        if str(exc) != "index out of range":
            raise
        return single_dwell, "sfHMM found no admissible split"

    return KVobj.KV_trace, None


def get_putative_LLPS(
    trace_subdict,
    sampling_rate,
    channel="diff",
    target_ds_factor_4_LLPs_det=625,
    pause_loc_tol_um=0.34/1000,
    mad_threshold=5,
    split_pause_window_size_idx=313,
    min_pause_samples=1250,
    filter_type="bessel"
):
    lc = trace_subdict["lctrace_difflf"]

    if channel == "diff":
        d = trace_subdict["dlf"]
        f = trace_subdict["flf"]

    elif channel == "1x":
        d = trace_subdict["d1xlf"]
        f = trace_subdict["f1xlf"]

    elif channel == "2x":
        d = trace_subdict["d2xlf"]
        f = trace_subdict["f2xlf"]

    else:
        raise ValueError("Select valid channels (1x/2x/diff)")


    LLPS = ana.PutativeLLPsDetector(
        d,
        f,
        sampling_rate,
        target_ds_factor_4_LLPs_det,
        pause_loc_tol_um,
        WLCp,
        mad_thresh=mad_threshold,
        split_pause_window_size_idx=split_pause_window_size_idx,
        min_pause_samples=min_pause_samples,
        filter_type=filter_type
    )

    LLPS.run()


    pause_levels = get_pause_levels(
        LLPS.pause_intervals_idx,
        lc
    )

    pause_segments = get_pause_segments(
        LLPS.pause_intervals_idx,
        pause_levels
    )


    return (
        LLPS.pause_intervals_idx,
        pause_levels,
        pause_segments
    )

def load_postprocessed_traces(csv_path, include_alignment_fecs=True, chunksize=250_000):
    """Load only required columns without retaining the complete CSV DataFrame.

    Only passive traces and the fecS02 traces required for single-channel
    alignment are retained. Arrays are accumulated in bounded CSV chunks and
    consolidated one trace at a time.
    """
    usecols = [
        "fec_id",
        "rupture_index",
        "relative_time",
        "trappos1x",
        "molext_aln",
        "diffF_aln",
        "curve_type",
        "force1x",
        "force2x",
    ]
    array_columns = {
        "t": "relative_time",
        "trappos1x": "trappos1x",
        "d": "molext_aln",
        "f": "diffF_aln",
        "f1x": "force1x",
        "f2x": "force2x",
    }

    trace_parts = {}
    trace_order = []

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"fec_id": "category", "curve_type": "category"},
        chunksize=chunksize,
    )

    for chunk in reader:
        for trace_id, trace_chunk in chunk.groupby(
            "fec_id", sort=False, observed=True
        ):
            trace_id = str(trace_id)

            is_fec = "fec" in trace_id
            if is_fec and not (
                include_alignment_fecs and "fecS02" in trace_id
            ):
                continue

            if trace_id not in trace_parts:
                trace_order.append(trace_id)
                trace_parts[trace_id] = {
                    "ri": trace_chunk.rupture_index.iloc[0],
                    "curve_type": trace_chunk.curve_type.iloc[0],
                    "arrays": {key: [] for key in array_columns},
                }

            arrays = trace_parts[trace_id]["arrays"]
            for key, column in array_columns.items():
                arrays[key].append(trace_chunk[column].to_numpy(copy=True))

        del chunk

    traces = {}
    for trace_id in trace_order:
        parts = trace_parts.pop(trace_id)
        trace = {
            "ri": parts["ri"],
            "curve_type": parts["curve_type"],
        }

        for key, arrays in parts["arrays"].items():
            trace[key] = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

        traces[trace_id] = trace

    return traces, trace_order

def downsample_signals (x,downsampling_factor, fs_original, filter_type = "bessel"):
    if filter_type == "bessel":
        xdf = bessel_filtfilt(x, downsampling_factor, fs_original)
    elif filter_type == "butter":
        xdf = butter_filtfilt(x, downsampling_factor, fs_original)
    elif filter_type == "boxcar":
        xdf = CK_filtfilt(x, downsampling_factor)
    else:
         raise ValueError("Enter a valid filter type (bessel/boxcar/butter)")
    return(xdf)

def get_trace_base (trace):
    underscore_indexes = [x.start() for x in re.finditer("_", trace)]
    trace_base = trace[:underscore_indexes[-1]]
    return(trace_base)

def gen_baseline_name (fec_id):
    # Getting baseline name
    underscore_ids = [x.start() for x in re.finditer("_", fec_id)]
    dte =  fec_id[0:underscore_ids[0]]
    bset = fec_id[underscore_ids[0]+1:underscore_ids[1]]
    bsls = [dte + "_" + bset + "_"+ "fec01", dte + "_" + bset + "_"+ "fec02",dte + "_" + bset + "_"+ "fec03"]
    return(bsls)

def get_alignment_offsets(dlf, flf, dist_bounds, force_bounds, WLCpars, distance_aln_indexes=None):
    iter_aln_fec = post.AlignFEC(
        dlf, flf,
        dist_bounds,
        force_bounds,
        *WLCpars,
        custom_function=False,
        distance_aln_indexes=distance_aln_indexes
    )
    iter_aln_fec.iterative_fd_aln()

    doff = (iter_aln_fec.x_aln - dlf)[0]
    foff = (iter_aln_fec.F_aln - flf)[0]
    return doff, foff

def get_trace_group(base, passive, alltraces):
    fecs = [f"{base}_fecS00", f"{base}_fecS01", f"{base}_fecS02", f"{base}_fecS03"]
    dct_smp = [x for x in passive if base in x]
    traces = fecs + dct_smp

    return [x for x in traces if x in alltraces]

def gen_lc_trace_df (passive_subdict, trace_id, single_channel = False):
    tlf = passive_subdict["tlf"]
    
    
    segment_names = [x for x in passive_subdict.keys() if x.startswith("segment_")]
    
    interval_edges = [passive_subdict[x]["duration"] for x in segment_names]
    segment_modes = [passive_subdict[x]["mode"] for x in segment_names]
    
    # Preallocate instead of repeatedly growing Python lists with one entry per
    # sample. Categories keep the repeated string columns compact in memory.
    segment_name_col = np.empty(len(tlf), dtype=object)
    segment_mode_col = np.empty(len(tlf), dtype=object)
    covered = np.zeros(len(tlf), dtype=bool)
    
    for c, (sn, ie, sm) in enumerate(zip(segment_names, interval_edges, segment_modes)):
        if c == len(segment_names)-1:
            mask = (tlf >= ie[0]) & (tlf <= ie[-1])
        else:
            mask = (tlf >= ie[0]) & (tlf < ie[-1])
        
        segment_name_col[mask] = sn
        segment_mode_col[mask] = sm
        covered[mask] = True

    if not np.all(covered):
        raise ValueError(f"Segment definitions do not cover every sample in {trace_id}.")

    data = {
        "relative_time": tlf,
        "molext_diff": passive_subdict["dlf"],
        "diffF": passive_subdict["flf"],
        "lcdiff": passive_subdict["lctrace_difflf"],
        "ideal_trace": passive_subdict["ideal_trace"],
        "LLPs_label": passive_subdict["ideal_LLPs_label"],
        "segment_id": pd.Categorical(segment_name_col),
        "segment_mode": pd.Categorical(segment_mode_col),
        "trace": pd.Categorical.from_codes(
            np.zeros(len(tlf), dtype=np.int8), categories=[trace_id]
        ),
    }

    for ch in ("1x", "2x"):
        if single_channel:
            data[f"molext_{ch}"] = passive_subdict[f"d{ch}_shiftedlf"]
            data[f"f{ch}"] = passive_subdict[f"f{ch}_shiftedlf"]
            data[f"lc{ch}"] = passive_subdict[f"lctrace_{ch}lf"]
            
        else:
            data[f"molext_{ch}"] = np.nan
            data[f"f{ch}"] = np.nan
            data[f"lc{ch}"] = np.nan

    out = pd.DataFrame(data, copy=False)

    extra_columns = [
        "molext_1x", "f1x", "lc1x",
        "molext_2x", "f2x", "lc2x",
    ]
    
    sorted_columns = ["relative_time"] + extra_columns + ["molext_diff",
                                                          "diffF",
                                                          "lcdiff",
                                                          "ideal_trace",
                                                          "LLPs_label",
                                                          "segment_id",
                                                          "segment_mode",
                                                          "trace"]
    out = out[sorted_columns]
           
    return(out)
            
def get_nominal_sampling_rate(t):
    """Infer the original regular sampling rate even if time regions were trimmed."""
    t = np.asarray(t, dtype=float)

    if len(t) < 2:
        raise ValueError("At least two saved time points are required to infer sampling rate.")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        raise ValueError("Saved time points must be finite and strictly increasing.")

    # Manual trimming may leave large gaps. The shortest increments represent
    # the original regular grid; use their median to suppress floating error.
    smallest_dt = np.min(dt)
    regular_dt = dt[dt <= 1.5 * smallest_dt]

    return 1 / np.median(regular_dt)


def nearest_time_indexes(source_t, target_t):
    """Map target times to their nearest samples on a source time grid."""
    source_t = np.asarray(source_t, dtype=float)
    target_t = np.asarray(target_t, dtype=float)

    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 0, len(source_t) - 1)
    left = np.clip(right - 1, 0, len(source_t) - 1)

    use_right = np.abs(source_t[right] - target_t) < np.abs(target_t - source_t[left])
    return np.where(use_right, right, left)


def exact_subset_indexes(source_t, retained_t):
    """Return indexes of retained sorted times without np.isin's large temporaries."""
    source_t = np.asarray(source_t, dtype=float)
    retained_t = np.asarray(retained_t, dtype=float)

    if len(retained_t) == 0:
        return np.empty(0, dtype=np.intp)

    indexes = np.searchsorted(source_t, retained_t, side="left")
    if np.any(indexes >= len(source_t)):
        raise ValueError("A retained time lies beyond the source time grid.")

    if len(source_t) > 1:
        atol = max(1e-12, np.median(np.diff(source_t)) * 1e-6)
    else:
        atol = 1e-12

    if not np.allclose(source_t[indexes], retained_t, rtol=0, atol=atol):
        raise ValueError("Retained times do not match the source sampling grid.")

    return indexes


LOW_FREQUENCY_SIGNAL_KEYS = (
    "dlf",
    "flf",
    "d1xlf",
    "f1xlf",
    "lctrace_1xlf",
    "d1x_shiftedlf",
    "f1x_shiftedlf",
    "d2xlf",
    "f2xlf",
    "lctrace_2xlf",
    "d2x_shiftedlf",
    "f2x_shiftedlf",
)

def remove_trace_keys(trace_subdict, keys):
    """Remove optional large arrays without failing when a key is absent."""
    for key in keys:
        trace_subdict.pop(key, None)


def manual_checkpoint_path(checkpoint_dir, trace_id):
    if re.fullmatch(r"[A-Za-z0-9_.-]+", trace_id) is None:
        raise ValueError(f"Unsafe trace ID for checkpoint filename: {trace_id!r}")
    return os.path.join(checkpoint_dir, f"{trace_id}.npz")


def load_discarded_trace_ids(discarded_trace_path):
    """Load trace IDs that were manually discarded in an earlier run."""
    if not os.path.exists(discarded_trace_path):
        return set()

    with open(discarded_trace_path, "r", encoding="utf-8") as discarded_file:
        return {
            line.strip()
            for line in discarded_file
            if line.strip()
        }


def save_discarded_trace_ids(discarded_trace_path, discarded_trace_ids):
    """Atomically persist the complete set of manually discarded trace IDs."""
    os.makedirs(os.path.dirname(discarded_trace_path), exist_ok=True)
    temporary_path = discarded_trace_path + ".tmp"

    try:
        with open(temporary_path, "w", encoding="utf-8") as discarded_file:
            for trace_id in sorted(discarded_trace_ids):
                discarded_file.write(f"{trace_id}\n")
            discarded_file.flush()
            os.fsync(discarded_file.fileno())

        os.replace(temporary_path, discarded_trace_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def save_manual_checkpoint(
    checkpoint_dir,
    trace_id,
    trace_subdict,
    configuration_signature,
):
    """Atomically save only the state needed to recover one curated trace."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = manual_checkpoint_path(checkpoint_dir, trace_id)
    temporary_path = checkpoint_path + ".tmp"

    pause_intervals = np.asarray(
        trace_subdict["putative_LLP_intervals_idx"], dtype=np.int64
    ).reshape(-1, 2)

    try:
        with open(temporary_path, "wb") as checkpoint_file:
            np.savez(
                checkpoint_file,
                format_version=np.asarray(1, dtype=np.int64),
                trace_id=np.asarray(trace_id),
                configuration_signature=np.asarray(configuration_signature),
                tlf=np.asarray(trace_subdict["tlf"]),
                lctrace_difflf=np.asarray(trace_subdict["lctrace_difflf"]),
                ideal_trace=np.asarray(trace_subdict["ideal_trace"]),
                putative_LLP_intervals_idx=pause_intervals,
            )
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

        os.replace(temporary_path, checkpoint_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_manual_checkpoint(
    checkpoint_dir,
    trace_id,
    configuration_signature,
):
    """Load and validate one completed manual-curation checkpoint."""
    checkpoint_path = manual_checkpoint_path(checkpoint_dir, trace_id)
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            format_version = int(checkpoint["format_version"])
            saved_trace_id = str(checkpoint["trace_id"])
            saved_signature = str(checkpoint["configuration_signature"])

            if format_version != 1:
                raise ValueError(f"unsupported format version {format_version}")
            if saved_trace_id != trace_id:
                raise ValueError(
                    f"contains trace {saved_trace_id!r}, expected {trace_id!r}"
                )
            if saved_signature != configuration_signature:
                raise ValueError("was created with different analysis parameters")

            payload = {
                "tlf": checkpoint["tlf"].copy(),
                "lctrace_difflf": checkpoint["lctrace_difflf"].copy(),
                "ideal_trace": checkpoint["ideal_trace"].copy(),
                "putative_LLP_intervals_idx": checkpoint[
                    "putative_LLP_intervals_idx"
                ].copy(),
            }
    except Exception as exc:
        raise RuntimeError(
            f"Cannot recover manual checkpoint {checkpoint_path}: {exc}"
        ) from exc

    lengths = {
        len(payload["tlf"]),
        len(payload["lctrace_difflf"]),
        len(payload["ideal_trace"]),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has inconsistent lengths.")

    return payload


def restore_manual_checkpoint(trace_subdict, payload):
    """Apply a saved curation to signals recalculated from the source files."""
    source_t = np.asarray(trace_subdict["tlf"])
    retained_t = payload["tlf"]
    retained_indexes = exact_subset_indexes(source_t, retained_t)

    for key in LOW_FREQUENCY_SIGNAL_KEYS:
        if key in trace_subdict:
            trace_subdict[key] = np.asarray(trace_subdict[key])[retained_indexes]

    trace_subdict["tlf"] = retained_t
    trace_subdict["lctrace_difflf"] = payload["lctrace_difflf"]
    trace_subdict["ideal_trace"] = payload["ideal_trace"]
    trace_subdict["putative_LLP_intervals_idx"] = payload[
        "putative_LLP_intervals_idx"
    ]


def restore_segments_from_lctrace_df(trace_subdict, tmpdf):
    """Rebuild segment_* dictionaries from the saved per-sample segment columns."""
    t = tmpdf.relative_time.to_numpy()
    segment_ids = tmpdf.segment_id.to_numpy()
    segment_modes = tmpdf.segment_mode.to_numpy()

    # Remove any segment definitions already present in the dictionary.
    for key in list(trace_subdict.keys()):
        if key.startswith("segment_"):
            del trace_subdict[key]

    if len(t) == 0:
        return

    change_idx = np.r_[0, np.where(segment_ids[1:] != segment_ids[:-1])[0] + 1]

    for c, start in enumerate(change_idx):
        next_start = change_idx[c + 1] if c + 1 < len(change_idx) else len(t)

        segment_name = segment_ids[start]
        if pd.isna(segment_name):
            segment_name = f"segment_{c + 1}"
        else:
            segment_name = str(segment_name)

        start_time = t[start]
        if next_start < len(t):
            end_time = t[next_start]
        else:
            end_time = t[-1]

        trace_subdict[segment_name] = {
            "duration": [start_time, end_time],
            "mode": segment_modes[start],
        }


def restore_existing_annotation(
    trace_subdict,
    tmpdf,
    target_t=None,
    restore_ideal_trace=True
):
    """Restore saved annotations, optionally mapping them to a new time grid."""
    if target_t is None:
        mapped_df = tmpdf.copy()
    else:
        target_t = np.asarray(target_t, dtype=float)
        saved_t = tmpdf.relative_time.to_numpy(dtype=float)
        saved_idx = nearest_time_indexes(saved_t, target_t)
        mapped_df = tmpdf.iloc[saved_idx].copy().reset_index(drop=True)
        mapped_df["relative_time"] = target_t

    if restore_ideal_trace:
        trace_subdict["ideal_trace"] = mapped_df.ideal_trace.to_numpy()

    labels = mapped_df.LLPs_label.to_numpy()
    trace_subdict["ideal_LLPs_label"] = labels
    trace_subdict["putative_LLP_intervals_idx"] = intervals_from_binary_labels(labels)

    restore_segments_from_lctrace_df(trace_subdict, mapped_df)


def restore_existing_signals(trace_subdict, tmpdf, single_channel=True):
    """Restore the already-downsampled signals and Lc traces from lctraces.csv."""
    trace_subdict["tlf"] = tmpdf.relative_time.to_numpy()
    trace_subdict["dlf"] = tmpdf.molext_diff.to_numpy()
    trace_subdict["flf"] = tmpdf.diffF.to_numpy()
    trace_subdict["lctrace_difflf"] = tmpdf.lcdiff.to_numpy()

    if single_channel:
        for ch in ("1x", "2x"):
            d = tmpdf[f"molext_{ch}"].to_numpy()
            f = tmpdf[f"f{ch}"].to_numpy()

            # The CSV stores the aligned single-channel signals. For an old trace
            # these are also sufficient for the manual-editor bookkeeping. Use
            # shared read-only references here; later trimming rebinds rather
            # than modifying either array in place.
            trace_subdict[f"d{ch}lf"] = d
            trace_subdict[f"f{ch}lf"] = f
            trace_subdict[f"d{ch}_shiftedlf"] = d
            trace_subdict[f"f{ch}_shiftedlf"] = f
            trace_subdict[f"lctrace_{ch}lf"] = tmpdf[f"lc{ch}"].to_numpy()

    restore_existing_annotation(trace_subdict, tmpdf)


def trim_recalculated_trace_to_saved_times(trace_subdict, tmpdf):
    """Apply saved retained-time regions to a recalculated trace on any time grid."""
    new_t = np.asarray(trace_subdict["tlf"], dtype=float)
    saved_t = tmpdf.relative_time.to_numpy(dtype=float)

    saved_sampling_rate = get_nominal_sampling_rate(saved_t)
    saved_dt = 1 / saved_sampling_rate

    # Divide the saved samples into contiguous regions. Gaps correspond to
    # regions removed during previous manual curation.
    gap_idx = np.where(np.diff(saved_t) > 1.5 * saved_dt)[0]
    starts = np.r_[0, gap_idx + 1]
    ends = np.r_[gap_idx, len(saved_t) - 1]

    # Treat every saved timestamp as the center of one old sampling interval.
    # This preserves the same retained time support when the new grid is either
    # finer or coarser than the saved grid.
    trim_mask = np.zeros(len(new_t), dtype=bool)
    half_saved_dt = saved_dt / 2
    for start, end in zip(starts, ends):
        trim_mask |= (
            (new_t >= saved_t[start] - half_saved_dt)
            & (new_t <= saved_t[end] + half_saved_dt)
        )

    if not np.any(trim_mask):
        raise ValueError("Saved retained-time regions do not overlap the recalculated trace.")

    keys_to_trim = [
        "tlf",
        "trappos1xlf",
        "dlf",
        "flf",
        "d1xlf",
        "f1xlf",
        "d2xlf",
        "f2xlf",
        "d1x_shiftedlf",
        "f1x_shiftedlf",
        "d2x_shiftedlf",
        "f2x_shiftedlf",
        "lctrace_difflf",
        "lctrace_1xlf",
        "lctrace_2xlf",
    ]

    for key in keys_to_trim:
        if key in trace_subdict and len(trace_subdict[key]) == len(new_t):
            trace_subdict[key] = trace_subdict[key][trim_mask]

def get_WLCp_from_metadata(csv_basename, default_WLCp):
    metadata_path = (
        f"data_consolidation/"
        f"{csv_basename}_lctrace_calculation_metadata.txt"
    )

    if not os.path.exists(metadata_path):
        print(f"No metadata found. Using default WLC parameters: {default_WLCp}")
        return default_WLCp

    with open(metadata_path, "r") as metadata_file:
        for line in metadata_file:
            if line.startswith("WLC model:"):
                try:
                    stored_WLCp = ast.literal_eval(
                        line.split(":", 1)[1].strip()
                    )
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Invalid WLC model in {metadata_path}"
                    ) from exc

                print(f"Using WLC parameters from metadata: {stored_WLCp}")
                return stored_WLCp

    print(f"No WLC model found in metadata. Using default: {default_WLCp}")
    return default_WLCp

def finalize_ideal_trace_and_LLPs(trace_subdict):
    """Apply final KV cleanup and store final inclusive LLP labels."""
    final_fix_obj = ana.FindKVTrace(
        None,
        PF=None,
        min_step_size=step_size_tol,
        min_dwell_samples=min_dwell_samples,
    )
    final_fix_obj.KV_trace = trace_subdict["ideal_trace"]
    final_fix_obj.remove_short_kv_dwells()
    final_fix_obj.remove_small_kv_steps()

    ideal_trace = final_fix_obj.KV_trace
    ideal_LLPs_intervals, _, _ = ana.get_ideal_putative_LLPS(
        ideal_trace,
        mad_thresh=5
    )

    ideal_LLPs_label_array = np.zeros(len(ideal_trace), dtype=np.int8)
    for start, end in ideal_LLPs_intervals:
        ideal_LLPs_label_array[start:end + 1] = 1

    trace_subdict["ideal_trace"] = ideal_trace
    trace_subdict["ideal_LLPs_pause_intervals"] = ideal_LLPs_intervals
    trace_subdict["ideal_LLPs_label"] = ideal_LLPs_label_array

    return ideal_LLPs_intervals


def save_final_trace_figure(trace_id, trace_subdict, ideal_LLPs_intervals):
    """Create one diagnostic figure and release its temporary arrays on return."""
    trace = trace_subdict["lctrace_difflf"]
    ideal_trace = trace_subdict["ideal_trace"]
    t = trace_subdict["tlf"]
    d = trace_subdict["dlf"]
    f = trace_subdict["flf"]

    trace_avg = ana.downsample_lc_trace(
        d,
        f,
        target_ds_factor_4_vis,
        sampling_rate,
        WLCp,
        filter_type="boxcar",
        decimate=False,
        cutoff_factor=1,
    )

    trace_bp = trace * 1000 / 0.34
    ideal_trace_bp = ideal_trace * 1000 / 0.34
    trace_avg_bp = trace_avg * 1000 / 0.34

    try:
        fplt.plot_individual_motor_trace(
            t,
            trace_bp,
            ideal_trace_bp,
            xlim=(t[0], t[-1]),
            ylim=(
                np.nanmin(trace_bp) - 50,
                np.nanmax(trace_bp) + 50,
            ),
            xtickspace=20,
            ytickspace=50,
            trace_id=trace_id,
            lctrace_averaged=trace_avg_bp,
            pause_intervals=ideal_LLPs_intervals,
            savefig=True,
            show=True,
            path="data_consolidation/trace_figures",
        )
    finally:
        plt.close("all")


def append_trace_to_lctrace_csv(
    trace_id,
    trace_subdict,
    output_path,
    write_header,
):
    """Build and append one trace DataFrame without retaining it."""
    tmpdf = gen_lc_trace_df(
        trace_subdict,
        trace_id,
        single_channel=single_channel,
    )
    tmpdf.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


#%%
csv_basename = "260820_BiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_50nMChd1_1mMATP_1umBeads"
single_channel = True
var_component = 0
#ds_factor = 5
filter_type = "bessel"

# If True, old passive traces are re-downsampled/re-aligned and their Lc and KV
# steps are recalculated from the consolidate files. Their manual trimming,
# segment definitions, and curated LLP intervals are preserved, but their old
# ideal trace is not reused. Existing traces saved at a different rate from
# target_frequency always undergo this complete recalculation, regardless of
# this flag and regardless of whether target_frequency is higher or lower.
recalculate_existing_lctraces = False

#sampling variables
target_frequency = sampling_ratelf = 1562.5

#WLC
WLCp  = get_WLCp_from_metadata(csv_basename,
                               default_WLCp=([50, None, 1200], [0.65, 0.01296]),)

#single channel aln pars
if single_channel:
    alignment_target_freq = 125 #Hz

# pause det parameters
pause_det_channel = "diff" # diff is for normal cases, but B10 has issues in 1x...
pause_position_tolerance = 0.34/1000 # len of 1 bp in um
step_size_tol = 0.4*0.34/1000 # len of 0.4 bp in 1 um
fc_for_pause_det = 0.5 # Hz
min_pause_duration_s = 1

#%%
# Read only columns used by this script. The postprocessed file is loaded in
# bounded chunks directly into per-trace arrays, so a second full copy never
# remains alive as dfpost.
calpars_path = f"data_consolidation/{os.path.basename(csv_basename)}_calpars_consolidate.csv"
annotation_path = f"data_consolidation/{os.path.basename(csv_basename)}_annotated_states_consolidate.csv"
postprocessed_path = f"data_consolidation/{os.path.basename(csv_basename)}_postprocessed_consolidate.csv"

dfcalpars = pd.read_csv(
    calpars_path,
    usecols=["baseline_id", "trap", "kappa (pN/nm)"],
    dtype={"baseline_id": "category", "trap": "category"},
)

dfann = pd.read_csv(
    annotation_path,
    usecols=["fec_id", "states"],
    dtype={"fec_id": "category"},
)

fecs_dict, alltraces = load_postprocessed_traces(
    postprocessed_path,
    include_alignment_fecs=single_channel,
)

traces_base = list(dict.fromkeys(get_trace_base(x) for x in alltraces))
passive = [x for x in alltraces if "fec" not in x]
# The chunked loader retains only fecS02, the alignment reference actually used.
alignment_reference_fecs = [x for x in alltraces if "fecS02" in x]

# Existing Lc/annotation output, if this dataset has already been processed before.
lctrace_csv_path = f"data_consolidation/{csv_basename}_lctraces.csv"
discarded_lctraces_path = (
    f"data_consolidation/{csv_basename}_discarded_lctraces.txt"
)
discarded_lc_traces = load_discarded_trace_ids(discarded_lctraces_path)

discarded_traces_in_consolidate = [
    trace_id for trace_id in passive
    if trace_id in discarded_lc_traces
]
for trace_id in discarded_traces_in_consolidate:
    fecs_dict.pop(trace_id, None)
passive = [
    trace_id for trace_id in passive
    if trace_id not in discarded_lc_traces
]

if discarded_traces_in_consolidate:
    print(
        f"Excluded {len(discarded_traces_in_consolidate)} previously "
        "discarded Lc trace(s)."
    )

if os.path.exists(lctrace_csv_path):
    if recalculate_existing_lctraces:
        # Recalculation needs only retained times and manual annotations.
        lctrace_usecols = [
            "relative_time",
            "LLPs_label",
            "segment_id",
            "segment_mode",
            "trace",
        ]
    else:
        lctrace_usecols = None

    df_lc_existing = pd.read_csv(
        lctrace_csv_path,
        usecols=lctrace_usecols,
        dtype={
            "trace": "category",
            "segment_id": "category",
            "segment_mode": "category",
        },
    )
else:
    df_lc_existing = None

#%%

if df_lc_existing is None:
    existing_passive = []
else:
    saved_trace_ids = set(df_lc_existing.trace.unique())
    existing_passive = [x for x in passive if x in saved_trace_ids]

new_passive = [x for x in passive if x not in existing_passive]

existing_sampling_rates = {}
for i in existing_passive:
    saved_t = df_lc_existing.loc[
        df_lc_existing.trace == i,
        "relative_time"
    ].to_numpy()
    existing_sampling_rates[i] = get_nominal_sampling_rate(saved_t)

existing_with_target_rate = [
    i for i in existing_passive
    if np.isclose(existing_sampling_rates[i], target_frequency, rtol=1e-6, atol=1e-6)
]

existing_with_different_rate = [
    i for i in existing_passive
    if i not in existing_with_target_rate
]

if recalculate_existing_lctraces:
    existing_to_recalculate = existing_passive.copy()
else:
    # A saved trace is reusable only when it already obeys target_frequency.
    existing_to_recalculate = existing_with_different_rate.copy()

existing_to_restore = [
    i for i in existing_passive
    if i not in existing_to_recalculate
]

passive_to_recalculate = new_passive + existing_to_recalculate

print(f"Existing passive traces: {len(existing_passive)}")
print(f"New passive traces: {len(new_passive)}")
print(f"Existing traces already at {target_frequency:g} Hz: {len(existing_with_target_rate)}")
print(f"Existing traces requiring rate conversion: {len(existing_with_different_rate)}")
print(f"Existing traces being recalculated: {len(existing_to_recalculate)}")

#%% Extract data

if len(passive) == 0:
    raise ValueError("No passive traces were found in the postprocessed file.")

# Build single-channel molecular extensions only for passive traces and the
# fecS02 alignment references retained by the chunked loader.
for i in passive + alignment_reference_fecs:
    bsl = gen_baseline_name(i)[0]
    k1 = dfcalpars[ (dfcalpars.baseline_id == bsl) & (dfcalpars["trap"] == "trap1") ]["kappa (pN/nm)"].to_numpy()[0]
    k2 = dfcalpars[ (dfcalpars.baseline_id == bsl) & (dfcalpars["trap"] == "trap2") ]["kappa (pN/nm)"].to_numpy()[0]

    if i in alignment_reference_fecs:
        fecs_dict[i]["states"] = dfann.loc[dfann.fec_id == i, "states"].to_numpy()

    if single_channel:
        fecs_dict[i]["d1x"] = fecs_dict[i]["trappos1x"] + fecs_dict[i]["f1x"]/k1/1000 + fecs_dict[i]["f1x"]/k2/1000
        fecs_dict[i]["d2x"] = fecs_dict[i]["trappos1x"] - fecs_dict[i]["f2x"]/k1/1000 - fecs_dict[i]["f2x"]/k2/1000

# The calibration and annotation DataFrames have now been reduced to the
# scalars/arrays required later and can be released.
del dfcalpars, dfann
gc.collect()

original_sampling_rate = round(1/(fecs_dict[passive[0]]["t"][1]-fecs_dict[passive[0]]["t"][0]))

if original_sampling_rate <= target_frequency:
    raise ValueError(
        f"The detected original sampling rate ({original_sampling_rate:g} Hz) "
        f"must be higher than target_frequency ({target_frequency:g} Hz)."
    )

#%% Downsample relevant signals

### Important to recall that for a transformation of a variable x by process g,
### if the process is linear, then average(g(x)) = g(average(x)), where average
### could be a filter_decimate (downsample) process with any linear filter, like
### the three implemented here. However, if the process is non-linear, then
### average(g(x)) != g(average(x)). In this case, it is better to do g(average(x)),
### since here you average white noise on x. If you do average(g(x)), you include
### transform the white noise to something else and then averging with linear filters
### will be worse. How much worse, I do not know. 
### Conversion of F,d signals into Lc is a nonlinear process, hence, it is better to
### downsample the signal firsts before doing the transformation.
### As a countercase, conversion of trap position and F to d is a linear process, 
### hence, calculation of d(molecular extension) can be done before downsampling or after,
### it should be the same.
ds_factor = int(round(original_sampling_rate / target_frequency))
actual_target_frequency = original_sampling_rate / ds_factor

if not np.isclose(actual_target_frequency, target_frequency, rtol=1e-9, atol=1e-9):
    raise ValueError(
        f"target_frequency={target_frequency:g} Hz cannot be obtained from the "
        f"detected original_sampling_rate={original_sampling_rate:g} Hz by integer "
        "Bessel filtering/decimation."
    )

raw_signal_keys = (
    "t", "trappos1x", "d", "f", "f1x", "f2x", "d1x", "d2x"
)

# Traces restored directly from the existing Lc CSV do not need any raw arrays.
for i in existing_to_restore:
    remove_trace_keys(fecs_dict[i], raw_signal_keys)

# A fecS02 is needed only to calculate the two alignment offsets. Downsample
# just its four required single-channel signals and retain the two raw values
# needed for dini as scalars.
for i in alignment_reference_fecs:
    fecs_dict[i]["dini_1x"] = fecs_dict[i]["d"][0] - fecs_dict[i]["d1x"][0]
    fecs_dict[i]["dini_2x"] = fecs_dict[i]["d"][0] - fecs_dict[i]["d2x"][0]

    for key in ("f1x", "f2x", "d1x", "d2x"):
        fecs_dict[i][f"{key}lf"] = downsample_signals(
            fecs_dict[i][key],
            ds_factor,
            fs_original=original_sampling_rate,
            filter_type=filter_type,
        )

    remove_trace_keys(fecs_dict[i], raw_signal_keys)

# Downsample one passive trace and immediately release its raw arrays instead
# of retaining raw and downsampled copies for the entire dataset.
for i in passive_to_recalculate:
    for source_key, output_key in (
        ("trappos1x", "trappos1xlf"),
        ("d", "dlf"),
        ("f", "flf"),
        ("f1x", "f1xlf"),
        ("f2x", "f2xlf"),
        ("d1x", "d1xlf"),
        ("d2x", "d2xlf"),
    ):
        fecs_dict[i][output_key] = downsample_signals(
            fecs_dict[i][source_key],
            ds_factor,
            fs_original=original_sampling_rate,
            filter_type=filter_type,
        )

    fecs_dict[i]["tlf"] = (
        np.arange(len(fecs_dict[i]["dlf"])) / sampling_ratelf
    )
    remove_trace_keys(fecs_dict[i], raw_signal_keys)
    gc.collect()
    
#%% align d1x and d2x channels

ds_factor_aln = round(sampling_ratelf/alignment_target_freq)

if single_channel:
    # align fecS02
    for base in traces_base:
        print(base)
        fecS02 = f"{base}_fecS02"
        state = fecs_dict[fecS02]["states"][0]
    
        WLCref = copy.deepcopy(WLCp)
        WLCref[var_component][1] = state
    
        set_of_traces = [
            trace_id
            for trace_id in get_trace_group(base, passive, alltraces)
            if trace_id in passive_to_recalculate
        ]
    
        for ch in ("1x", "2x"):
    
            dini = fecs_dict[fecS02][f"dini_{ch}"]
            dlf = fecs_dict[fecS02][f"d{ch}lf"]
            
            if ch == "1x":
                flf = -fecs_dict[fecS02][f"f{ch}lf"]
            else:
                flf = fecs_dict[fecS02][f"f{ch}lf"]
    
            doff, foff = get_alignment_offsets(
                CK_filtfilt(dlf,ds_factor_aln) + dini, CK_filtfilt(flf,ds_factor_aln),
                [[0.55, 1], [2, 4]],
                [[0.5, 0.6], [-100, 1]],
                WLCref,
                distance_aln_indexes=None
            )
    
            for tr in set_of_traces:
                if any(s in tr for s in ("dct01","fecS02","dct02","spm02","fecS03")):
                    fecs_dict[tr][f"d{ch}_shiftedlf"] = fecs_dict[tr][f"d{ch}lf"] + doff + dini
                    if ch == "1x":
                        fecs_dict[tr][f"f{ch}_shiftedlf"] = -fecs_dict[tr][f"f{ch}lf"] + foff
                    else:
                        fecs_dict[tr][f"f{ch}_shiftedlf"] = fecs_dict[tr][f"f{ch}lf"] + foff

# No FEC arrays are used after alignment.
for i in alignment_reference_fecs:
    fecs_dict.pop(i, None)
gc.collect()

#%% get lcs with downsample signals

for i in passive_to_recalculate:
    dlf,flf = fecs_dict[i]["dlf"],fecs_dict[i]["flf"]
    fecs_dict[i]["lctrace_difflf"] = ana.quick_per_point_Lc(dlf, flf, *WLCp)

if single_channel:
    for i in passive_to_recalculate:
        for ch in ("1x", "2x"):
            dlf,flf = fecs_dict[i][f"d{ch}_shiftedlf"], fecs_dict[i][f"f{ch}_shiftedlf"]  
            fecs_dict[i][f"lctrace_{ch}lf"] = ana.quick_per_point_Lc(dlf, flf, *WLCp)

#%% Restore signals and metadata

for i in existing_passive:
    tmpdf_existing = df_lc_existing[df_lc_existing.trace == i].copy()
    tmpdict = fecs_dict[i]

    if i in existing_to_recalculate:
        old_rate = existing_sampling_rates[i]
        print(
            f"{i}: recalculated from {original_sampling_rate:g} Hz to "
            f"{target_frequency:g} Hz; saved trace was {old_rate:g} Hz"
        )

        # Preserve the retained time regions from previous manual curation on
        # the newly calculated target-frequency grid.
        trim_recalculated_trace_to_saved_times(tmpdict, tmpdf_existing)

        # Map the saved manual pause annotation and segment definitions to the
        # target-frequency grid. The old ideal trace is deliberately excluded:
        # KV steps must be refitted from the recalculated target-frequency Lc.
        restore_existing_annotation(
            tmpdict,
            tmpdf_existing,
            target_t=tmpdict["tlf"],
            restore_ideal_trace=False
        )

    else:
        # Reuse is allowed only because this trace already matches the target rate.
        restore_existing_signals(tmpdict, tmpdf_existing, single_channel=single_channel)

# The existing Lc DataFrame and its final per-trace slice are no longer needed.
df_lc_existing = None
tmpdf_existing = None
gc.collect()

# Enforce the target-frequency rule before any trace-level analysis continues.
for i in passive:
    trace_sampling_rate = get_nominal_sampling_rate(fecs_dict[i]["tlf"])
    if not np.isclose(trace_sampling_rate, target_frequency, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"{i} has sampling rate {trace_sampling_rate:g} Hz after restoration/"
            f"recalculation; expected target_frequency={target_frequency:g} Hz."
        )


#%% FIND SEGMENTS
target_sampling_rate = 25
target_ds_factor = int(round(sampling_ratelf / target_sampling_rate))

# Segment detection is only needed for genuinely new traces. Existing traces
# retain their saved segment definitions.
for i in new_passive:
    tlf = fecs_dict[i]["tlf"]
    if "dct" in i:
        tmpdict = {}
        tmpdict["duration"] = [tlf[0],tlf[-1]]
        tmpdict["mode"] = "passive"
        fecs_dict[i]["segment_1"] = tmpdict

    elif "spm" in i:
        print(i)

        trappos1xlflf = bessel_filtfilt(fecs_dict[i]["trappos1xlf"],target_ds_factor, sampling_ratelf,decimate=True)
        
        segment_sampling_rate = sampling_ratelf / target_ds_factor
        tlflf = np.arange(len(trappos1xlflf)) / segment_sampling_rate
        _, _, pause_edges = ana.split_trappos_modes(tlflf,
                                                    trappos=trappos1xlflf,
                                                    threshold=0.001,
                                                    min_run_len=5,
                                                    min_velocity_threshold=9)
        current_time = tlflf[0]
        end_time = tlflf[-1]
        
        segment_time_intervals = []
        segment_modes = []
        
        for current_start, current_end in pause_edges:
            if current_start > current_time:
                segment_time_intervals.append([current_time, current_start])
                segment_modes.append("active")
        
            segment_time_intervals.append([current_start, current_end])
            segment_modes.append("passive")
        
            current_time = current_end
        
        if current_time < end_time:
            segment_time_intervals.append([current_time, end_time])
            segment_modes.append("active")
        
        if end_time != tlf[-1]:
            segment_time_intervals[-1][-1] = tlf[-1]
            
        for c, (sti, sm) in enumerate(zip(segment_time_intervals, segment_modes)):
            tmpdict = {}
            tmpdict["duration"] = sti
            tmpdict["mode"] = sm
            fecs_dict[i][f"segment_{c+1}"] = tmpdict
            
            mask = (tlflf >= sti[0]) & (tlflf < sti[1])
            plt.plot(tlflf[mask],trappos1xlflf[mask])
            
        plt.show()
        plt.close("all")

# Trap position was needed only for segment detection.
for i in passive:
    fecs_dict[i].pop("trappos1xlf", None)
gc.collect()

#%% GET SAMPLE RATE VARIABLES

sampling_rate = sampling_ratelf

min_pause_duration_samples = int(
    round(min_pause_duration_s * sampling_rate)
)

split_pause_window_size_idx = int(
    round(0.5 * sampling_rate)
)

target_ds_factor_4_LLPs_det = max(
    1,
    int(round(
        sampling_rate / (2 * fc_for_pause_det)
    ))
)

#%% LLPS annotation

# Existing traces already have their LLP annotation in lctraces.csv.
for i in new_passive:

    tmpdict = fecs_dict[i]

    pause_intervals_idx, pause_levels, pause_segments = get_putative_LLPS(
        tmpdict,
        sampling_rate=sampling_rate,
        channel=pause_det_channel,
        target_ds_factor_4_LLPs_det=target_ds_factor_4_LLPs_det,
        pause_loc_tol_um=pause_position_tolerance,
        mad_threshold=5,
        split_pause_window_size_idx=split_pause_window_size_idx,
        min_pause_samples=min_pause_duration_samples,
        filter_type="bessel"
    )

    fecs_dict[i]["putative_LLP_intervals_idx"] = pause_intervals_idx
    
    print(pause_intervals_idx)
    
    fecs_dict[i]["putative_LLP_levels"] = pause_levels
    fecs_dict[i]["putative_LLP_segments"] = pause_segments

    fecs_dict[i]["KV_intervals"] = get_complement_intervals(
        pause_intervals_idx,
        len(tmpdict["tlf"]),
        segment_inclusive=True
    )


# Existing recalculated traces retain their curated pause intervals, but pause
# levels and all non-pause fitting regions must be rebuilt on the newly
# calculated target-frequency Lc. This applies whether the target frequency is
# higher or lower than the saved frequency.
for i in existing_to_recalculate:
    rebuild_fit_regions_from_preserved_pauses(fecs_dict[i])


# Every trace with a newly calculated Lc must also receive a new KV fit at that
# same frequency. Tie these operations to the same trace list so recalculating
# Lc without recalculating its KV steps is impossible. Only existing traces
# restored without recalculation may reuse their saved ideal trace.
traces_to_refit = passive_to_recalculate.copy()


#%% FIT KV SEGMENTS in non putative LLPs

# Also used later by the final cleanup. Define it even when there are no new traces.
min_dwell_samples = 5 # int(np.ceil(2*filter_rise_time_10_90 * sampling_ratelf)). 
                      # if the two step filter is preserved
                      # then an increase in sampling rate will scale as the filter_rise_time_10_90
                      # scales down, so this is independent of the frequencies. 
                      # Asked chatgpt to solve this for 625 and it gave about 5 samples

for i in traces_to_refit:
    tmpdict = fecs_dict[i]
    print(f"{i}: recalculating KV segments at {sampling_rate:g} Hz")

    pause_segs = tmpdict["putative_LLP_intervals_idx"]
    segs = tmpdict["KV_intervals"]
    lc = tmpdict["lctrace_difflf"]
    t =  tmpdict["tlf"]
    
    KV_segments = []
    KV_segments_dwell_intervals = []
    
    for segment_number, (start, end) in enumerate(segs, start=1):
        # get segment and dwell intervals per segment
        lcseg = lc[start:end+1]
        KV_trace, fallback_reason = fit_kv_segment_or_single_dwell(
            lcseg,
            PF=4,
            min_step_size=step_size_tol,
            min_dwell_samples=min_dwell_samples
        )

        if fallback_reason is not None:
            print(
                f"{i}: KV interval {segment_number} [{start}, {end}] "
                f"was fitted as one dwell because {fallback_reason}"
            )

        KV_segments.append(KV_trace)
        
        # get intervals global index intervals per segment
        intervals,_ = ana.get_dwell_intervals(KV_trace,
                                        inclusive=True,
                                        offset=start
                                        )

        KV_segments_dwell_intervals.append(intervals)
        
        
    fecs_dict[i]["KV_segments"] = KV_segments
    fecs_dict[i]["KV_segments_dwell_intervals"] = KV_segments_dwell_intervals

#%% Reconstruct ideal trace

# Rebuild the ideal trace for every new or recalculated Lc. Existing traces that
# were restored without recalculation keep the ideal trace from lctraces.csv.
for i in traces_to_refit:
    tmpdict = fecs_dict[i]
    
    pause_intervals = tmpdict["putative_LLP_intervals_idx"]
    pause_segments = tmpdict["putative_LLP_segments"]
    KV_intervals = tmpdict["KV_intervals"]
    KV_segments = tmpdict["KV_segments"]
    KV_segments_dwell_intervals = tmpdict["KV_segments_dwell_intervals"]
    
    ITobj = ana.ReconstructIdealTrace(pause_intervals,
                                  pause_segments,
                                  KV_intervals,
                                  KV_segments,
                                  KV_segments_dwell_intervals,
                                  step_size_tolerance = step_size_tol)
    
    ITobj.run()
    fecs_dict[i]["ideal_trace"] = ITobj.ideal_trace

# The reconstructed ideal trace is the only fitting product required by the
# editor and final output. Release the full-length KV/pause intermediates.
fit_intermediate_keys = (
    "putative_LLP_levels",
    "putative_LLP_segments",
    "KV_intervals",
    "KV_segments",
    "KV_segments_dwell_intervals",
)
for i in passive:
    remove_trace_keys(fecs_dict[i], fit_intermediate_keys)
ITobj = None
gc.collect()

#%% Defining Manual curation and checkpoint settins  

# Each completed trace is saved independently and atomically. A fresh run
# restores these files and skips only the traces that were already completed.
manual_checkpoint_dir = os.path.join(
    "data_consolidation",
    f"{csv_basename}_manual_curation_checkpoints",
)

visualization_sampling_rate = 100
target_ds_factor_4_vis = max(
    1,
    int(round(sampling_rate / visualization_sampling_rate))
)

# A persistent discard decision supersedes any older completion checkpoint.
for trace_id in discarded_lc_traces:
    checkpoint_path = manual_checkpoint_path(manual_checkpoint_dir, trace_id)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

checkpoint_configuration_signature = repr(
    (
        csv_basename,
        float(target_frequency),
        bool(single_channel),
        WLCp,
        filter_type,
        float(step_size_tol),
        int(min_dwell_samples),
    )
)

completed_manual_traces = set()
for trace_id in passive:
    checkpoint_payload = load_manual_checkpoint(
        manual_checkpoint_dir,
        trace_id,
        checkpoint_configuration_signature,
    )
    if checkpoint_payload is not None:
        restore_manual_checkpoint(fecs_dict[trace_id], checkpoint_payload)
        completed_manual_traces.add(trace_id)

if completed_manual_traces:
    print(
        f"Recovered {len(completed_manual_traces)} completed manual trace(s); "
        "continuing with the remaining traces."
    )

traces_for_manual_curation = [
    trace_id for trace_id in passive
    if trace_id not in completed_manual_traces
]


#%% MANUAL CURATION 

# DANGEROUS: ONLY USE FOR CORRECTING PAUSES
# DO NOT TAMPER WITH THE FIT APART FROM THAT.

c = 0

while c < len(traces_for_manual_curation):

    trace_id = traces_for_manual_curation[c]
    tmpdict = fecs_dict[trace_id]

    trace = tmpdict["lctrace_difflf"]
    ideal_trace = tmpdict["ideal_trace"]

    t = tmpdict["tlf"]
    d = tmpdict["dlf"]
    f = tmpdict["flf"]

    # Single-channel signals
    d1x = tmpdict["d1xlf"]
    f1x = tmpdict["f1xlf"]
    lc1x = tmpdict["lctrace_1xlf"]

    d2x = tmpdict["d2xlf"]
    f2x = tmpdict["f2xlf"]
    lc2x = tmpdict["lctrace_2xlf"]

    # Shifted/aligned single-channel signals
    d1x_shifted = tmpdict["d1x_shiftedlf"]
    f1x_shifted = tmpdict["f1x_shiftedlf"]

    d2x_shifted = tmpdict["d2x_shiftedlf"]
    f2x_shifted = tmpdict["f2x_shiftedlf"]

    pause_intervals = tmpdict["putative_LLP_intervals_idx"]


    # Trace used only for visualization
    trace_avg = ana.downsample_lc_trace(
        d,
        f,
        target_ds_factor_4_vis,
        sampling_rate,
        WLCp,
        filter_type="boxcar",
        decimate=False,
        cutoff_factor=1
    )


    # Convert Lc traces to bp
    trace_bp = trace * 1000 / 0.34
    ideal_trace_bp = ideal_trace * 1000 / 0.34
    trace_avg_bp = trace_avg * 1000 / 0.34


    # Keep original time array.
    # EditMotorTrace can remove samples from t.
    original_t = t.copy()
    
    # Manual editor
    o = ana.EditMotorTrace(
        t,
        trace_bp,
        trace_avg_bp,
        ideal_trace_bp,
        xlim=(t[0], t[-1]),
        ylim=(np.nanmin(trace_bp) - 50, np.nanmax(trace_bp) + 50),
        xtickspace=10,
        ytickspace=50,
        trace_id=trace_id,
        step_size_tol=step_size_tol,
        min_dwell_samples=5,
        KV_PF=4,
        pause_intervals=pause_intervals,
    )

    continue_decision = o.run()

    if continue_decision == "d":
        plt.close("all")

        # Persist the decision before releasing this trace from memory. The
        # separate list prevents it from being treated as a new trace later.
        discarded_lc_traces.add(trace_id)
        save_discarded_trace_ids(
            discarded_lctraces_path,
            discarded_lc_traces,
        )

        checkpoint_path = manual_checkpoint_path(
            manual_checkpoint_dir,
            trace_id,
        )
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        fecs_dict.pop(trace_id, None)
        passive.remove(trace_id)
        traces_for_manual_curation.pop(c)
        print(f"Discarded Lc trace {trace_id}")

        o = None
        tmpdict = None
        trace = ideal_trace = trace_avg = trace_avg_bp = None
        original_t = retained_indexes = None
        d = f = d1x = f1x = lc1x = None
        d2x = f2x = lc2x = None
        d1x_shifted = f1x_shifted = None
        d2x_shifted = f2x_shifted = None
        trace_bp = ideal_trace_bp = pause_intervals = None
        gc.collect()

        # The next trace shifted into the current list position, so c must not
        # be incremented.
        continue


    # Get edited outputs
    t = o.time
    trace_bp = o.trace
    ideal_trace_bp = o.ideal_trace
    pause_intervals = o.pause_intervals
    plt.close("all")


    # Find which samples survived manual trimming.
    # t is ALREADY trimmed, so do NOT index t again. searchsorted avoids the
    # large temporary allocations made by np.isin on long traces.
    retained_indexes = exact_subset_indexes(original_t, t)


    # Apply exactly the same trim to all other signals
    d = d[retained_indexes]
    f = f[retained_indexes]

    d1x = d1x[retained_indexes]
    f1x = f1x[retained_indexes]
    lc1x = lc1x[retained_indexes]

    d2x = d2x[retained_indexes]
    f2x = f2x[retained_indexes]
    lc2x = lc2x[retained_indexes]

    d1x_shifted = d1x_shifted[retained_indexes]
    f1x_shifted = f1x_shifted[retained_indexes]

    d2x_shifted = d2x_shifted[retained_indexes]
    f2x_shifted = f2x_shifted[retained_indexes]


    # Sanity check: everything should have exactly the same length
    lengths = [
        len(t),
        len(d),
        len(f),
        len(d1x),
        len(f1x),
        len(lc1x),
        len(d2x),
        len(f2x),
        len(lc2x),
        len(d1x_shifted),
        len(f1x_shifted),
        len(d2x_shifted),
        len(f2x_shifted),
        len(trace_bp),
        len(ideal_trace_bp),
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Length mismatch after trimming {trace_id}: {lengths}"
        )


    # Save everything
    fecs_dict[trace_id]["tlf"] = t

    fecs_dict[trace_id]["dlf"] = d
    fecs_dict[trace_id]["flf"] = f

    fecs_dict[trace_id]["d1xlf"] = d1x
    fecs_dict[trace_id]["f1xlf"] = f1x
    fecs_dict[trace_id]["lctrace_1xlf"] = lc1x
    fecs_dict[trace_id]["d1x_shiftedlf"] = d1x_shifted
    fecs_dict[trace_id]["f1x_shiftedlf"] = f1x_shifted

    fecs_dict[trace_id]["d2xlf"] = d2x
    fecs_dict[trace_id]["f2xlf"] = f2x
    fecs_dict[trace_id]["lctrace_2xlf"] = lc2x
    fecs_dict[trace_id]["d2x_shiftedlf"] = d2x_shifted
    fecs_dict[trace_id]["f2x_shiftedlf"] = f2x_shifted

    fecs_dict[trace_id]["lctrace_difflf"] = trace_bp * 0.34 / 1000
    fecs_dict[trace_id]["ideal_trace"] = ideal_trace_bp * 0.34 / 1000
    fecs_dict[trace_id]["putative_LLP_intervals_idx"] = pause_intervals

    save_manual_checkpoint(
        manual_checkpoint_dir,
        trace_id,
        fecs_dict[trace_id],
        checkpoint_configuration_signature,
    )
    print(f"Saved manual checkpoint for {trace_id}")

    # Break references held by the editor/plotting variables before opening the
    # next trace. Completed data remain only in fecs_dict and the checkpoint.
    o = None
    tmpdict = None
    trace = ideal_trace = trace_avg = trace_avg_bp = None
    original_t = retained_indexes = None
    d = f = d1x = f1x = lc1x = None
    d2x = f2x = lc2x = None
    d1x_shifted = f1x_shifted = None
    d2x_shifted = f2x_shifted = None
    trace_bp = ideal_trace_bp = pause_intervals = None
    gc.collect()


    # Move through traces
    if continue_decision == "c":

        c += 1

    elif continue_decision == "r":

        if c == 0:
            print(
                "No previous trace to return to. "
                "Working on the same trace."
            )
        else:
            c -= 1

    else:
        pass

#%% finalize, plot, and stream one trace at a time

# Keeping one outer loop prevents final labels and DataFrames from accumulating
# across the dataset. Plotting temporaries are released when the plotting helper
# returns, before the DataFrame for that trace is created.
os.makedirs("data_consolidation/trace_figures", exist_ok=True)

lctrace_csv_temporary_path = lctrace_csv_path + ".tmp"
if os.path.exists(lctrace_csv_temporary_path):
    os.remove(lctrace_csv_temporary_path)

try:
    first_trace = True

    for i in passive:
        subdict = fecs_dict[i]

        ideal_LLPs_intervals = finalize_ideal_trace_and_LLPs(subdict)

        save_final_trace_figure(
            i,
            subdict,
            ideal_LLPs_intervals,
        )
        gc.collect()

        append_trace_to_lctrace_csv(
            i,
            subdict,
            lctrace_csv_temporary_path,
            write_header=first_trace,
        )
        first_trace = False

        fecs_dict.pop(i)
        subdict = None
        ideal_LLPs_intervals = None
        gc.collect()

    os.replace(lctrace_csv_temporary_path, lctrace_csv_path)

except Exception:
    if os.path.exists(lctrace_csv_temporary_path):
        os.remove(lctrace_csv_temporary_path)
    raise


#%% remove completed manual-curation checkpoints

# Delete recovery checkpoints only after the dense output CSV is safely in place.
for trace_id in passive:
    checkpoint_path = manual_checkpoint_path(manual_checkpoint_dir, trace_id)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

try:
    os.rmdir(manual_checkpoint_dir)
except OSError:
    pass

#%%
now = datetime.now()
with open (f"data_consolidation/{csv_basename}_lctrace_calculation_metadata.txt", "w+") as f:
    f.write(csv_basename + " lctrace calculation was performed on : " + str(now) + "\n\n")
    f.write("The following parameters were used:\n")
    f.write("WLC model: " + str(WLCp) + "\n")
    f.write(f"Original sampling rate was {original_sampling_rate} Hz\n")
    f.write(f"Target sampling rate was {target_frequency} Hz\n")
    f.write(f"Downsampling factor was {ds_factor}\n")
    f.write(f"Pauses were detected using channel {pause_det_channel}\n")
    f.write(f"Pause position tolerance was {pause_position_tolerance} um\n")
    f.write(f"Step size tolerance was {step_size_tol} um\n")
    f.write(f"Cutoff frequency for pause detection was {fc_for_pause_det} Hz\n")
    f.write(f"Min pause duration threshold was {min_pause_duration_s} s" + "\n")

# The results are stored in the Lc CSV and metadata file. Do not serialize the
# complete interactive namespace: it contains figures, editor objects, and
# large duplicate arrays and can exhaust the kernel during saving.
