#%% CLEAN SESSION BEFORE START

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import warnings
#%%

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt, bessel_filtfilt
import tweezers_toolbox_modules.postprocessing as post
import tweezers_toolbox_modules.analysis as ana
import shutil, os,re
from tweezers_toolbox_modules import input_handlers as hand
import dill, glob
from matplotlib.gridspec import GridSpec
from tweezers_toolbox_modules.models import ini_eWLC
import copy
import seaborn as sns

#%%

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

def downsample_df_bessel_grouped(
    df,
    downsampling_factor,
    fs_original,
    group_cols,
    cutoff_factor=1,
    order=4,
    decimate=True,
    extra_non_numeric_cols=None,
    sort_groups=False,
    dropna=False,
):
    """
    Downsample a dataframe containing multiple independent traces.

    Filtering/downsampling is applied independently within each group.

    Behavior:
      - Numeric signal columns are Bessel-filtered and optionally decimated.
      - group_cols are never filtered, even if numeric.
      - extra_non_numeric_cols are never filtered, even if numeric.
      - relative_time is never filtered.
      - If relative_time exists, it is rebuilt as a clean uniform time axis.
      - Robust to pandas excluding group columns from each group in groupby.apply().
    """

    if extra_non_numeric_cols is None:
        extra_non_numeric_cols = []

    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise KeyError(
            f"Missing group column(s): {missing_group_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    cols = list(df.columns)
    has_rel_time = "relative_time" in cols

    fs_out = fs_original / downsampling_factor if decimate else fs_original

    # Columns that should not be filtered, even if numeric
    non_filtered = set(extra_non_numeric_cols) | set(group_cols)

    if has_rel_time:
        non_filtered.add("relative_time")

    # Numeric columns that should actually be filtered
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in non_filtered]

    # Metadata columns: copied/decimated only, never filtered
    metadata_cols = [c for c in cols if c not in numeric_cols]

    def _group_key_dict(g):
        """
        Return dictionary mapping group column names to their current group value.

        This fixes pandas versions where groupby.apply() excludes the grouping
        columns from the group dataframe.
        """
        if len(group_cols) == 1:
            return {group_cols[0]: g.name}

        return dict(zip(group_cols, g.name))

    def _downsample_one_group(g):
        n = len(g)

        pos = np.arange(n)

        if decimate:
            if n > 2 * downsampling_factor:
                pos = pos[downsampling_factor:-downsampling_factor]

            pos = pos[::downsampling_factor]

        n_out = len(pos)

        group_values = _group_key_dict(g)

        filtered_data = {}

        for c in numeric_cols:
            if c not in g.columns:
                raise KeyError(
                    f"Numeric column '{c}' not found inside group {getattr(g, 'name', None)}. "
                    f"Available group columns are: {list(g.columns)}"
                )

            x = g[c].to_numpy()

            y = bessel_filtfilt(
                x,
                downsampling_factor=downsampling_factor,
                fs_original=fs_original,
                order=order,
                decimate=decimate,
                axis=-1,
                cutoff_factor=cutoff_factor,
            )

            y = np.asarray(y)

            if len(y) != n_out:
                raise ValueError(
                    f"Length mismatch in group {getattr(g, 'name', None)}, "
                    f"column '{c}': len(filtered)={len(y)}, len(expected positions)={n_out}. "
                    f"This means bessel_filtfilt() and the metadata decimation logic are not aligned."
                )

            filtered_data[c] = y

        out = pd.DataFrame(index=g.index[pos])

        # Add filtered numeric signal columns
        for c in numeric_cols:
            out[c] = filtered_data[c]

        # Add metadata/group columns by decimation only
        for c in metadata_cols:
            if c == "relative_time":
                continue

            if c in g.columns:
                out[c] = g.iloc[pos][c].to_numpy()

            elif c in group_values:
                # Handles group columns like fec_id when pandas excludes them from g
                out[c] = group_values[c]

            else:
                raise KeyError(
                    f"Metadata column '{c}' not found inside group {getattr(g, 'name', None)} "
                    f"and it is not a group key. Available group columns are: {list(g.columns)}"
                )

        # Rebuild relative_time as clean uniform sampling
        if has_rel_time:
            out["relative_time"] = np.arange(n_out, dtype=float) / fs_out

        # Restore original column order
        return out[cols]

    out_df = (
        df.groupby(
            group_cols,
            sort=sort_groups,
            group_keys=False,
            dropna=dropna,
        )
        .apply(_downsample_one_group)
    )

    return out_df

def copy_csvs(folder, destination_path, lctraces=False):
    folder_base = os.path.basename(folder)

    shutil.copy(
        f"{folder}/preprocess/{folder_base}_calpars.csv",
        destination_path
    )

    # Optional video microscopy
    video_file = (
        f"{folder}/preprocess/video_microscopy/"
        f"{folder_base}video_microscopy.csv"
    )

    if os.path.exists(video_file):
        shutil.copy(video_file, destination_path)

    shutil.copy(
        f"{folder}/preprocess/{folder_base}_preprocessed_discarded.csv",
        destination_path
    )

    shutil.copy(
        f"{folder}/postprocess/{folder_base}_postprocessed.csv",
        destination_path
    )

    shutil.copy(
        f"{folder}/postprocess/{folder_base}_postprocessed_discarded.csv",
        destination_path
    )

    shutil.copy(
        f"{folder}/annotate/{folder_base}_annotated_states.csv",
        destination_path
    )

    if lctraces:
        shutil.copy(
            f"{folder}/per_pointLc/{folder_base}_lctraces.csv",
            destination_path
        )


def copy_consolidate (consolidate_path, destination_path):
    files = glob.glob(f"{consolidate_path}/*.csv")
    for i in files:
        shutil.copy(i, destination_path)

def get_tdLc (fec, annotation_df):
    tmpdf_ann = annotation_df[annotation_df.fec_id == fec]
    tdLc = tmpdf_ann[tmpdf_ann.fec_id == fec].dLcs_sum.to_list()[-1]
    
    if np.isnan(tdLc):
        tdLc = tmpdf_ann[tmpdf_ann.fec_id == fec].tdLc_direct.to_list()[-1]
        if np.isnan(tdLc):
            print(f"cannot calculate tdLc for {fec}, returning Nan")
    return(tdLc)

def select_fec(tdLc, 
               ini_force,
               end_force,
               rmsd,
               lthreshold,
               uthreshold,
               upper_ini_force_threshold=None,
               lower_final_force_threshold=None,
               rmsd_threshold=0.4):

    failures = []

    if not (lthreshold <= tdLc <= uthreshold):
        failures.append(6)
    
    if upper_ini_force_threshold:
        if not (ini_force <  upper_ini_force_threshold):
            failures.append(7)
    
    if lower_final_force_threshold:
        if not (end_force > lower_final_force_threshold):
            failures.append(8)

    if not (rmsd < rmsd_threshold):
        failures.append(9)

    return 0 if not failures else failures

def get_transitions_from_df (fec_id, annotation_df):
    tmpdf_ann = annotation_df[annotation_df.fec_id == fec_id]
    trans_idxs = tmpdf_ann.trans_idxs.to_numpy()
    states =  tmpdf_ann.states.to_numpy()
    dLcs_sum = tmpdf_ann.dLcs_sum.to_numpy()
    dLcs = tmpdf_ann.dLcs.to_numpy()
    dExts = tmpdf_ann.dExts.to_numpy()
    dExts_sum = tmpdf_ann.dExts_sum.to_numpy()
    tdExt_direct = tmpdf_ann.tdExt_direct.to_numpy()
    tdLc_direct = tmpdf_ann.tdLc_direct.to_numpy()
    return(trans_idxs,states,dLcs_sum,dLcs, dExts, dExts_sum,tdExt_direct,tdLc_direct)

def gen_baseline_name (fec_id):
    # Getting baseline name
    underscore_ids = [x.start() for x in re.finditer("_", fec_id)]
    dte =  fec_id[0:underscore_ids[0]]
    bset = fec_id[underscore_ids[0]+1:underscore_ids[1]]
    bsls = [dte + "_" + bset + "_"+ "fec01", dte + "_" + bset + "_"+ "fec02",dte + "_" + bset + "_"+ "fec03"]
    return(bsls)

def get_fd_from_df (fec_id, df, downfact):
    columns = ["trappos1x", "molext_aln", "diffF_aln", "curve_type"]
    tmpdf = df.loc[df["fec_id"] == fec_id, columns]

    if tmpdf.empty:
        raise KeyError(f"No postprocessed data found for {fec_id}")

    # Keep only the full-length arrays used later in this script. Explicit
    # copies prevent them from retaining a larger pandas backing block.
    d = tmpdf["molext_aln"].to_numpy(copy=True)
    f = tmpdf["diffF_aln"].to_numpy(copy=True)
    trappos1x = tmpdf["trappos1x"].to_numpy(copy=True)

    trace = {
        "d": d,
        "f": f,
        "dlf": CK_filtfilt(d, downfact),
        "flf": CK_filtfilt(f, downfact),
        "trappos1xlf": CK_filtfilt(trappos1x, downfact),
        "curve_type": tmpdf["curve_type"].iat[0],
    }

    return trace

def joint_scatter_with_marginals(
    x, y, *,
    labels=None,
    legend=False,
    noise_label="noise",
    bins_top=12,
    bins_right=12,
    figsize=(7,6),
    title=None,
    xlabel='X',
    ylabel='Y',
    lw_axes=2,
    lw_hist=2,
    fontsize_labels=22,
    fontsize_ticks=20,
    grid_alpha=0.45,
    grid_linewidth=0.8,
    show_hist_ticks=True,
    vlines=None,
    hlines=None,
    linecolor='red',
    linestyle='--',
    linewidth=1.8,
    noise_color='lightgray',
    xlim=None,          # <<< ADDED
    ylim=None           # <<< ADDED
):
    """
    Scatter plot with marginal histograms.
    Supports cluster coloring with a *consistent* color code:
      -1: noise (noise_color)
       0,1,2,...: stable colors from tab10 (wrapped if > 9)
    """

    import numpy as np


    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, "x and y must have same shape"

    # ---------- handle labels & colors ----------
    label_to_color = {}
    if labels is not None:
        labels = np.asarray(labels)
        assert labels.shape == x.shape

        unique_labels = np.unique(labels)
        base_cmap = plt.get_cmap('tab10')

        for lab in unique_labels:
            if lab == -1:
                label_to_color[lab] = noise_color
            else:
                label_to_color[lab] = base_cmap(int(lab) % base_cmap.N)

        colors = [label_to_color[lab] for lab in labels]
    else:
        colors = 'black'

    # ---------- layout ----------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0, wspace=0)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_histx   = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy   = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

    # ---------- scatter ----------
    ax_scatter.scatter(x, y, c=colors, s=25, alpha=0.9, edgecolor='none')
    ax_scatter.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax_scatter.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax_scatter.tick_params(axis='both', which='both',
                           labelsize=fontsize_ticks, width=lw_axes,
                           length=6, direction='inout')
    ax_scatter.grid(True, linestyle='--', alpha=grid_alpha,
                    linewidth=grid_linewidth)
    for spine in ax_scatter.spines.values():
        spine.set_linewidth(lw_axes)

    # ---------- CUSTOM LIMITS (key part) ----------
    if xlim is not None:
        ax_scatter.set_xlim(xlim)
    if ylim is not None:
        ax_scatter.set_ylim(ylim)

    # ---------- optional reference lines ----------
    if vlines is not None:
        for xv in np.atleast_1d(vlines):
            ax_scatter.axvline(x=xv, color=linecolor,
                               linestyle=linestyle, linewidth=linewidth)
    if hlines is not None:
        for yh in np.atleast_1d(hlines):
            ax_scatter.axhline(y=yh, color=linecolor,
                               linestyle=linestyle, linewidth=linewidth)

    # ---------- top histogram ----------
    ax_histx.hist(x, bins=bins_top, color='gray',
                  histtype='stepfilled',
                  edgecolor='black', linewidth=lw_hist)
    ax_histx.grid(True, linestyle=':',
                  alpha=grid_alpha, linewidth=grid_linewidth)
    ax_histx.tick_params(axis='x', bottom=False, labelbottom=False)
    if show_hist_ticks:
        ax_histx.tick_params(axis='y', labelsize=fontsize_ticks,
                             width=lw_axes, length=5)
    else:
        ax_histx.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax_histx.spines.values():
        spine.set_linewidth(lw_axes)

    # ---------- right histogram ----------
    ax_histy.hist(y, bins=bins_right, color='gray',
                  histtype='stepfilled',
                  edgecolor='black', linewidth=lw_hist,
                  orientation='horizontal')
    ax_histy.grid(True, linestyle=':',
                  alpha=grid_alpha, linewidth=grid_linewidth)
    ax_histy.tick_params(axis='y', left=False, labelleft=False)
    if show_hist_ticks:
        ax_histy.tick_params(axis='x', labelsize=fontsize_ticks,
                             width=lw_axes, length=5)
    else:
        ax_histy.tick_params(axis='x', bottom=False, labelbottom=False)
    for spine in ax_histy.spines.values():
        spine.set_linewidth(lw_axes)

    # ---------- legend ----------
    if legend and labels is not None:
        handles, names = [], []
        for lab in np.unique(labels):
            col = label_to_color[lab]
            name = noise_label if lab == -1 else f"cluster {lab}"
            h = plt.Line2D([0], [0], marker='o', linestyle='none',
                           markerfacecolor=col, markersize=10,
                           markeredgewidth=0)
            handles.append(h)
            names.append(name)

        fig.legend(handles, names, loc='center',
                   ncol=min(2, len(handles)),
                   fontsize=16, frameon=True,
                   bbox_to_anchor=(0.4, -0.05))
        bottom = 0.2
    else:
        bottom = 0.12

    # ---------- layout & title ----------
    plt.subplots_adjust(bottom=bottom, hspace=0, wspace=0)

    if title:
        fig.suptitle(title, fontsize=fontsize_labels, y=0.98)

    return fig, ax_scatter, ax_histx, ax_histy

def get_trace_base (trace):
    underscore_indexes = [x.start() for x in re.finditer("_", trace)]
    trace_base = trace[:underscore_indexes[-1]]
    return(trace_base)

# ---------- helper ----------
def get_group(name):
    m = re.search(r"(fecS\d+|dct\d+|spm\d+)", str(name))
    return m.group(1) if m else str(name)


def plot_all_PM_segments(
    fec_dict,
    set_of_traces,
    *eWLCp,
    fig_name="plot",
    outdir="",
    save=False,
    show=True,
    xlim=(0.5, 0.9),
    ylim=(0, 35),
    trace_alpha=0.7,
    state_alpha=1.0,
    state_lw=2,
    state_ls="--",
    annotation_text=None,
    title=None,
    title_color="black",
    comparison_segment=None,
    error_code=None,
):

    all_groups = sorted({get_group(k) for k in fec_dict.keys()})
    palette = sns.color_palette("tab10", n_colors=max(20, len(all_groups)))
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(all_groups)}

    pf = np.arange(0.01, 30, 0.01)
    WLCpars_mod = copy.deepcopy(eWLCp)

    if save and outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.figure()
    plt.ylim(*ylim)
    plt.xlim(*xlim)

    trace_handles = []
    state_items = []

    for c in set_of_traces:
        if error_code != None:
            if error_code == 0:
                base_color = "blue"
                base_color_wlc = "black"
            else:
                base_color = "red"
                base_color_wlc = "black"
        else:
            base_color = group_color[get_group(c)]
            base_color_wlc = base_color
                
        
        h_trace, = plt.plot(
            fec_dict[c]["dlf_shifted"],
            fec_dict[c]["flf_shifted"],
            label=c,
            color=base_color,
            alpha=trace_alpha
        )
        trace_handles.append(h_trace)

        states_c = sorted(fec_dict[c].get("states_shifted", []))
        for st in states_c:
            WLCpars_mod[var_component][1] = st
            pdist = ini_eWLC(1, pf, *WLCpars_mod)

            h_state, = plt.plot(
                pdist, pf, state_ls,
                color=base_color_wlc,
                alpha=state_alpha,
                linewidth=state_lw,
                label=f"{st:.3f} um"
            )
            state_items.append((st, h_state))

    if comparison_segment is not None:
        comparison_d, comparison_f = comparison_segment
        h_comparison, = plt.plot(
            comparison_d,
            comparison_f,
            color="green",
            linewidth=3,
            label="RMSD comparison segment",
        )
        trace_handles.append(h_comparison)

    leg1 = plt.legend(handles=trace_handles, loc="upper left", title="Traces")
    plt.gca().add_artist(leg1)

    state_items.sort(key=lambda t: t[0])
    state_handles = [h for _, h in state_items]

    plt.xlabel("Extension (um)")
    plt.ylabel("Force (pN)")

    if title is not None:
        plt.title(
            str(title).upper(),
            color=title_color,
            fontweight="bold",
            fontsize=18,
        )

    if state_handles:
        plt.legend(
            handles=state_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            title="States"
        )

    # ---- SIMPLE TEXT TOP RIGHT ----
    if annotation_text:
        plt.text(
            0.98, 0.98,
            annotation_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="top"
        )

    if save:
        if outdir:
            plt.savefig(os.path.join(outdir, f"{fig_name}.png"), dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def quick_discard (fecs_dict,set_of_traces, rmsd,lc_dev,exp_trans_size, error_code,*WLCp_basal, 
                   var_lc_component = 0,    xlim=(0.5, 0.9),
                       ylim=(0, 35),):

    kept = error_code == 0
    status = "KEPT" if kept else "DISCARDED"
    status_color = "green" if kept else "red"

    plot_all_PM_segments(fecs_dict,set_of_traces,
                         *refWLC, 
                         annotation_text=f"RMSD = {rmsd:.3f}\nLc deviation= {1000*(lc_dev-exp_trans_size):.3f} nm",
                         title=status,
                         title_color=status_color,
                         xlim=xlim,
                         ylim=ylim)  
    
    user_input = hand.option_handler("keep (k), discard (d), pass (p), or return (r)? ",
                                      valid_values=["k","d","p", "r"])
    
    if user_input == "k":
        error_code = 0
        return(error_code, "c")
    elif user_input == "d":
        error_code = 5
        return(error_code, "c")
    elif user_input == "p":
        return(error_code, "c")
    else:
        return(error_code, "r")

def get_trace_group(base, passive, alltraces):
    fecs = [
        f"{base}_fecS00",
        f"{base}_fecS01",
        f"{base}_fecS02",
        f"{base}_fecS03",
    ]

    dct_smp = [x for x in passive if base in x]
    traces = fecs + dct_smp

    return [x for x in traces if x in alltraces]


def get_alignment_offsets(
    dlf,
    flf,
    dist_bounds,
    force_bounds,
    WLCpars,
    distance_aln_indexes=None,
    only_dist_aln=False,
    first_transition=None,
):
    print(
    "\nget_alignment_offsets:",
    "only_dist_aln =", only_dist_aln,
    type(only_dist_aln)
)
    iter_aln_fec = post.AlignFEC(
        dlf,
        flf,
        dist_bounds,
        force_bounds,
        *WLCpars,
        custom_function=False,
        distance_aln_indexes=distance_aln_indexes,
        respect_trans_ind=first_transition,
    )

    if only_dist_aln:
        iter_aln_fec.iterative_dist_aln()

        doff = (iter_aln_fec.x_aln - dlf)[0]
        foff = 0

    else:
        iter_aln_fec.iterative_fd_aln()

        doff = (iter_aln_fec.x_aln - dlf)[0]
        foff = (iter_aln_fec.F_aln - flf)[0]

    return doff, foff


def contains_any(name, labels):
    return any(label in name for label in labels)


def apply_shift_to_trace(
    fecs_dict,
    tr,
    doff=0,
    foff=0,
    state_offset=None,
):
    """
    Apply a shift to one trace.

    If shifted arrays already exist, add the new shift to them.
    Otherwise, start from the raw arrays.
    """
    trace = fecs_dict[tr]

    # Distance
    if "d_shifted" in trace:
        trace["d_shifted"] += doff
        trace["dlf_shifted"] += doff
    else:
        trace["d_shifted"] = trace.pop("d")
        trace["d_shifted"] += doff
        trace["dlf_shifted"] = trace["dlf"] + doff

    # Force
    if "f_shifted" in trace:
        trace["f_shifted"] += foff
        trace["flf_shifted"] += foff
    else:
        trace["f_shifted"] = trace.pop("f")
        trace["f_shifted"] += foff
        trace["flf_shifted"] = trace["flf"] + foff

    # States
    if state_offset is not None and "fec" in tr:
        if "states_shifted" in trace:
            trace["states_shifted"] = (
                trace["states_shifted"] + state_offset
            )
        else:
            trace["states_shifted"] = (
                trace["states"] + state_offset
            )


def apply_shift_to_traces(
    fecs_dict,
    traces,
    doff=0,
    foff=0,
    state_offset=None,
):
    for tr in traces:
        apply_shift_to_trace(
            fecs_dict,
            tr,
            doff=doff,
            foff=foff,
            state_offset=state_offset,
        )


def get_pre_rupture_window(
    trans_idxs,
    offsets=(200, 12, 3),
    default_pre_rupture=300,
):
    rupture = trans_idxs[-1]

    if len(trans_idxs) == 1:
        return default_pre_rupture, rupture

    for offset in offsets:
        pre_rupture = trans_idxs[-2] + offset

        if pre_rupture <= rupture:
            return pre_rupture, rupture

    return trans_idxs[-2] + offsets[-1], rupture


def realign_base_to_beginning(base, align_force=True):

    fec_S01 = f"{base}_fecS01"
    fec_S02 = f"{base}_fecS02"

    traces = get_trace_group(
        base,
        passive,
        alltraces,
    )

    # ---------------------------------------------------------
    # Local helper: beginning alignment with fallback range
    # ---------------------------------------------------------

    def beginning_alignment(
        dlf,
        flf,
        WLCpars,
        first_transition=None,
        only_dist_aln=False,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)

            doff, foff = get_alignment_offsets(
                dlf,
                flf,
                ranges_for_dist_aln,
                ranges_for_force_aln,
                WLCpars,
                distance_aln_indexes=None,
                only_dist_aln=only_dist_aln,
                first_transition=first_transition,
            )

        empty_slice = any(
            "Mean of empty slice" in str(w.message)
            for w in caught
        )

        if empty_slice:
            expanded_dist_range = [
                ranges_for_dist_aln[0],
                [ranges_for_dist_aln[1][0], 10],
            ]

            print(
                "Empty distance alignment range. "
                f"Retrying with {expanded_dist_range}"
            )

            doff, foff = get_alignment_offsets(
                dlf,
                flf,
                expanded_dist_range,
                ranges_for_force_aln,
                WLCpars,
                distance_aln_indexes=None,
                only_dist_aln=only_dist_aln,
                first_transition=first_transition,
            )

        return doff, foff

    # ---------------------------------------------------------
    # Align fecS01 to initial WLC
    # ---------------------------------------------------------

    if "states_shifted" in fecs_dict[fec_S01]:
        start_state = fecs_dict[fec_S01]["states_shifted"][0]
    else:
        start_state = fecs_dict[fec_S01]["states"][0]

    start_transition = fecs_dict[fec_S01]["trans_idxs"][0]

    target_state = WLCpini[var_component][1]
    state_offset = target_state - start_state

    if "dlf_shifted" in fecs_dict[fec_S01]:
        dlf = fecs_dict[fec_S01]["dlf_shifted"]
    else:
        dlf = fecs_dict[fec_S01]["dlf"]

    if "flf_shifted" in fecs_dict[fec_S01]:
        flf = fecs_dict[fec_S01]["flf_shifted"]
    else:
        flf = fecs_dict[fec_S01]["flf"]

    doff, foff = beginning_alignment(
        dlf,
        flf,
        WLCpini,
        first_transition=start_transition,
        only_dist_aln=not align_force,
    )

    apply_shift_to_traces(
        fecs_dict,
        traces,
        doff=doff,
        foff=0,
        state_offset=state_offset,
    )

    # ---------------------------------------------------------
    # Force alignment requested
    # ---------------------------------------------------------

    if align_force:

        early_traces = [
            tr for tr in traces
            if contains_any(
                tr,
                ("fecS00", "dct01", "fecS01"),
            )
        ]

        apply_shift_to_traces(
            fecs_dict,
            early_traces,
            doff=0,
            foff=foff,
        )

        # -----------------------------------------------------
        # Align fecS02
        # -----------------------------------------------------

        dlf = fecs_dict[fec_S02]["dlf_shifted"]
        flf = fecs_dict[fec_S02]["flf_shifted"]

        state = fecs_dict[fec_S02]["states_shifted"][0]

        WLCpref = copy.deepcopy(WLCpini)
        WLCpref[var_component][1] = state

        doff, foff = beginning_alignment(
            dlf,
            flf,
            WLCpref,
            only_dist_aln=False,
        )

        late_traces = [
            tr for tr in traces
            if contains_any(
                tr,
                ("fecS02", "dct02", "spm02", "fecS03"),
            )
        ]

        apply_shift_to_traces(
            fecs_dict,
            late_traces,
            doff=doff,
            foff=foff,
        )

    plot_all_PM_segments(
        fecs_dict,
        traces,
        *refWLC,
        ylim=(-1, 15),
    )
    plt.close("all")


def realign_base_to_end(
    base,
    only_if_needed=False,
    align_force=True,
):
    fec_S03 = f"{base}_fecS03"

    traces = get_trace_group(
        base,
        passive,
        alltraces,
    )

    # Use shifted state if it already exists
    if "states_shifted" in fecs_dict[fec_S03]:
        last_state = fecs_dict[fec_S03]["states_shifted"][-1]
    else:
        last_state = fecs_dict[fec_S03]["states"][-1]

    target_state = refWLC[var_component][1]

    # ---------------------------------------------------------
    # Already at the correct final state
    # ---------------------------------------------------------

    if only_if_needed and last_state == target_state:

        # Create shifted arrays if they do not exist,
        # otherwise leave existing shifted arrays unchanged.
        apply_shift_to_traces(
            fecs_dict,
            traces,
            doff=0,
            foff=0,
            state_offset=0,
        )

        return

    state_offset = target_state - last_state

    # ---------------------------------------------------------
    # Pre-rupture region
    # ---------------------------------------------------------

    pre_rupture, rupture = get_pre_rupture_window(
        fecs_dict[fec_S03]["trans_idxs"],
        offsets=(200, 12, 3),
    )

    # Use shifted data if available
    if "dlf_shifted" in fecs_dict[fec_S03]:
        dlf = fecs_dict[fec_S03]["dlf_shifted"]
    else:
        dlf = fecs_dict[fec_S03]["dlf"]

    if "flf_shifted" in fecs_dict[fec_S03]:
        flf = fecs_dict[fec_S03]["flf_shifted"]
    else:
        flf = fecs_dict[fec_S03]["flf"]

    # ---------------------------------------------------------
    # Align to final WLC
    # ---------------------------------------------------------

    doff, foff = get_alignment_offsets(
        dlf,
        flf,
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        refWLC,
        distance_aln_indexes=[pre_rupture, rupture],
        only_dist_aln=not align_force,
    )

    apply_shift_to_traces(
        fecs_dict,
        traces,
        doff=doff,
        foff=foff,
        state_offset=state_offset,
    )

    plot_all_PM_segments(
        fecs_dict,
        traces,
        *refWLC,
        ylim=(-1, 25),
    )
    plt.close("all")

def zero_force(time, diffF, zero_window=0.25, max_offset=3):
    """
    Zero differential force using the mean force over the final time window.

    Parameters
    ----------
    time : array-like
        Time array in seconds.

    diffF : array-like
        Differential force array.

    zero_window : float, optional
        Duration at the end of the trace used to estimate the force offset.
        Default is 0.25 s.

    max_offset : float, optional
        Maximum absolute offset allowed for correction.
        Default is 3 pN.

    Returns
    -------
    diffF_zeroed : ndarray
        Force trace with the offset subtracted.
    """
    time = np.asarray(time)
    diffF = np.asarray(diffF)

    mask = time >= time[-1] - zero_window

    mean_offset = np.mean(diffF[mask])

    if abs(mean_offset) < max_offset:
        return diffF - mean_offset

    return diffF

#%%

#Go to a folder containing multiple datasets analyzed until annotation
cons_name = "1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_20nMChd1_1mMATP_1umBeads"
folders = glob.glob("2*") 
add_consolidate = False
remove_folders = False

if remove_folders:
    #folders.remove("260115_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_1umBeads")
    #folders.remove("260630_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_1umBeads")
    #folders.remove("260701_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_1umBeads")
    #folders.remove("260702_1xBiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_1umBeads")
    folders = []
    
copy_lctraces = False # only True if lctraces calculated
date_str = datetime.now().strftime("%y%m%d")

# MODIFY THIS NUMBER TO MATCH THE WLCp TUPLE INDEX OF THE WLC THAT WILL CHANGE.
# For example, say WLCp = ([Lp_DNA, Lc_DNA, St_DNA],[Lp_protein, Lc_protein])
# and you are unfolding protein. Then you should select var_component = 1.
var_component = 0 # what Lc will change? 

expected_transition_size = 269*0.34/1000 # (EF-Tu from PDB 1EFC)
rupture_force_threshold = 1

# Fully folded WLC, only used if aligning to begining
WLCpini =  ([50, 0.68997], [0.65, 0.01296])

# Fully unfolded WLC
WLCpend = ([50,  0.68997+expected_transition_size], [0.65, 0.01296])

# If need to realign to the begining (initial state). Otherwise, realignment
# will be done to the end if there is missmatches between states.
realign_begining = True

# If on then all will be realigned to the end, does not matter if missmatch or not
# will only work if realign_begining is false
if not realign_begining:
    realign_end_all = True

# If realign to begining is ON please define the distance and force ranges of your trace
# to perform distance and force alignments
if realign_begining == True:
    ranges_for_dist_aln = [[0.55, 1], [2, 4]]
    ranges_for_force_aln = [[0.5, 0.6], [-100, 1]] 

# Final review?
curate_annotation = True

# Parameters for filtering based on size
# This is in um. Lower and upper define range of sizes
# of traces to be kept. For instance, +/- 5 nm around
# The expected transition size
# Hexasome expected transition is a range between the size of the sliding arm
# to the same size plus 105*0.34
lower_size_bound = (expected_transition_size-5/1000)
upper_size_bound = (expected_transition_size+(49.3+5)/1000)

# PARAMETERS FOR FILTERING BASED ON RMSD AGAINST A REFERENCE WLC
if realign_begining:
    refWLC = WLCpini
else:
    refWLC = WLCpend # I use the fully unfolded WLC as reference if I did not realign to a
                     # initial state.
    
rupture_force_detection = "peaks"
# Define dictionary of options. Modify accordingly to your methods and needs
if rupture_force_detection == "peaks":
    trans_find_pars = {"rupture_force_detection" : "peaks",
                "force_cutoff" : 1,
                "threshold" : 0.5,
                "distance" : 5,
                "window" : 5}

elif rupture_force_detection == "hmm":
    trans_find_pars = {"rupture_force_detection" : "hmm",
                "force_cutoff" : 2,
                "max_states" : 10}

align_force = False

#%%

try:
    os.makedirs("data_consolidation/data/")

except:
    pass

try:
    os.makedirs('data_consolidation/consolidate_pics/')
except:
    pass

for folder in folders:
    copy_csvs(folder, "data_consolidation/data/", lctraces = copy_lctraces)


if add_consolidate:
    copy_consolidate ("data_consolidation", "data_consolidation/data/")

#%%

#post
dfpost = [pd.read_csv(f"data_consolidation/data/{os.path.basename(folder)}_postprocessed.csv") for folder in folders]
dfpost = dfpost + [pd.read_csv(x) for x in glob.glob("data_consolidation/data/*_postprocessed_consolidate.csv")]
dfpost = pd.concat(dfpost)

#ann
dfann = [pd.read_csv(f"data_consolidation/data/{os.path.basename(folder)}_annotated_states.csv") for folder in folders]
dfann = dfann + [pd.read_csv(x) for x in glob.glob("data_consolidation/data/*_annotated_states_consolidate.csv")]
dfann = pd.concat(dfann)

#%%

alltraces = list(set(dfpost.fec_id))
traces_base = list(set([get_trace_base (x) for x in alltraces]))
passive = [x for x in alltraces if "fec" not in x]
fecs = [x for x in alltraces if "fec"  in x]
fecsS03 = [x for x in fecs if "fecS03" in x]
fecsS00_02 = [x for x in fecs if ("fecS03" not in x) and "fecS01" not in x]

#%%
fecs_dict = {}

for i in alltraces:
    fecs_dict[i] = get_fd_from_df(i, dfpost, 25)
    fecs_dict[i]["error_code"] = 0

    if "fec" in i:
        fecs_dict[i]["tdLc"] = get_tdLc (i, dfann)
        
        (
        fecs_dict[i]["trans_idxs"],
        fecs_dict[i]["states"],
        fecs_dict[i]["dLcs_sum"],
        fecs_dict[i]["dLcs"],
        fecs_dict[i]["dExts"], 
        fecs_dict[i]["dExts_sum"],
        fecs_dict[i]["tdExt_direct"],
        fecs_dict[i]["tdLc_direct"]
        ) =  get_transitions_from_df (i, dfann)

del dfann

#%%

#traces_base.remove("260208_19_01")

if realign_begining:
    for base in traces_base:
        realign_base_to_beginning(
            base,
            align_force=align_force
        )

elif realign_end_all:
    for base in traces_base:
        realign_base_to_end(
            base,
            only_if_needed=False,
            align_force=align_force
        )

else:
    for base in traces_base:
        realign_base_to_end(
            base,
            only_if_needed=True,
            align_force=align_force
        )

# Release figures created while checking the alignments before manual curation.
plt.close("all")

#%% curate fecS03

#fecsS03.remove("260208_19_01_fecS03")

if curate_annotation:
    i = 0
    while i < len(fecsS03):
        fec_id = fecsS03[i]
        fec_data = fecs_dict[fec_id]
    
        d = fec_data["d_shifted"]
        dlf = fec_data["dlf_shifted"]
    
        f =  fec_data["f_shifted"]
        flf = fec_data["flf_shifted"]
        trans_idxs = fec_data["trans_idxs"]
        stts = fec_data["states_shifted"]
        
        error_code = fec_data["error_code"]
        
        o = ana.EditStates(fec_id,
                           dlf, 
                           flf,
                           d,
                           f,  
                           trans_idxs,
                           trans_find_pars,
                           stts, 
                           refWLC[var_component][1],
                           error_code, 
                           [[0.53,1.1],[0,50]],
                           *refWLC, 
                           var_lc_component = var_component)
        
        

        continue_decision = o.run()
    
        fecs_dict[fec_id]["trans_idxs"] = o.trans_idxs
        fecs_dict[fec_id]["states_shifted"] = o.states
        fecs_dict[fec_id]["dLcs_sum"] = o.dLcs_sum
        fecs_dict[fec_id]["dLcs"] = o.dLcs
        
        if o.dLcs_sum[-1]:
            fecs_dict[fec_id]["tdLc"] = o.dLcs_sum[-1]
        else:
            pass

        # EditStates has finished using its interactive figure.
        plt.close("all")
    
        if continue_decision == "c":
            i += 1
        elif continue_decision == "r":
            if i ==0:
                print("No previous fec to return to. Working on the same FEC.")
            else:
                i -= 1
        else:
            pass

    dill.dump_session("data_consolidation/checkpoint1.pkl")

#%% Update fecS01

for i in traces_base:
    fecs_dict[f"{i}_fecS01"]["dLcs"] = fecs_dict[f"{i}_fecS03"]["states_shifted"][0] - fecs_dict[f"{i}_fecS01"]["states_shifted"]
    
#%% RECALCULATE MIRROR POSITION EXTENSIONS WITH FINAL RIPS FOR FECSS03

trans_params = {"nfit" : 20, "navg" : 3, "FAvgRange" : 0.05, "FFitRange" : 1,"padding": 3,
              "Lp": 50, "S": None}

for i in fecsS03:
    fec = fecs_dict[i]
    trans_idxs = fec["trans_idxs"]
    curve_type = fec["curve_type"]
    d = fec["trappos1xlf"]
    f = fec["flf"]

    if curve_type == "R":
        d = np.flip(d)
        f = np.flip(f)
        trans_idxs = len(f) - 1 - trans_idxs
        
    trans_idxs = np.sort(fec["trans_idxs"])
    
    if len(trans_idxs) > 1:
        trans_obj = ana.FindTransExtensions(i,d, f, trans_idxs,
                                            trans_params = trans_params, debug=False)
        trans_obj.run()
        fecs_dict[i]["trans_obj"] = trans_obj

# SAVE IN DICTIONARY
for i in fecsS03:
    lf_trans_idxs = fecs_dict[i]["trans_idxs"] 
    if len(lf_trans_idxs) > 1:
        trans_obj = fecs_dict[i]["trans_obj"]
        dExts, _, dExts_sum, _, _, _, _ = unpack_trans_obj (trans_obj)
    else:
        dExts, _,dExts_sum, _,_, _, _ = 0,0,0,0,0,0,0
        
    fecs_dict[i]["dExts"] = dExts
    fecs_dict[i]["dExts_sum"] = dExts_sum
        

#%% curate rest of fecs

if curate_annotation:

    i = 0
    while i < len(fecsS00_02):
        fec_id = fecsS00_02[i]
        fec_data = fecs_dict[fec_id]
        trace_base = get_trace_base(fec_id)
    
        d = fec_data["d_shifted"]
        dlf = fec_data["dlf_shifted"]
    
        f =  fec_data["f_shifted"]
        flf = fec_data["flf_shifted"]
        trans_idxs = fec_data["trans_idxs"]
        stts = fec_data["states_shifted"]
        
        error_code = fec_data["error_code"]
        
        o = ana.EditStates(fec_id,
                           dlf, 
                           flf,
                           d,
                           f,  
                           trans_idxs,
                           trans_find_pars,
                           stts, 
                           refWLC[var_component][1],
                           error_code, 
                           [[0.53,1.1],[0,30]],
                           *refWLC, 
                           var_lc_component = var_component)
        

        
        continue_decision = o.run()
    
        fecs_dict[fec_id]["states_shifted"] = o.states
        fecs_dict[fec_id]["dLcs_sum"] = o.dLcs_sum
        fecs_dict[fec_id]["dLcs"] = fecs_dict[trace_base+"_fecS03"]["states_shifted"][0] - o.states
        fecs_dict[fec_id]["trans_idxs"] = o.trans_idxs
        fecs_dict[fec_id]["states_shifted"] = o.states


        # EditStates has finished using its interactive figure.
        plt.close("all")
    
        if continue_decision == "c":
            i += 1
        elif continue_decision == "r":
            if i ==0:
                print("No previous fec to return to. Working on the same FEC.")
            else:
                i -= 1
        else:
            pass
    
    # Save the full session in case something bad happens during later stages.
    dill.dump_session("data_consolidation/checkpoint2.pkl")

#%%

alltdLcs = np.array([  fecs_dict[x+"_fecS03"]["dLcs_sum"][-1]+fecs_dict[x+"_fecS01"]["dLcs"][0] for x in traces_base])*1000

# I care about ini force on the fec right before remodelling
all_ini_sliding_force = np.array([  fecs_dict[x+"_fecS02"]["flf"][-1] for x in traces_base   ])

#plot in nm
joint_scatter_with_marginals(
    alltdLcs-expected_transition_size*1000, all_ini_sliding_force,
    bins_top=24,
    bins_right=12,
    xlabel='Contour length deviation (nm)',
    ylabel='Initial rupture force (pN)',
    show_hist_ticks=False,
    vlines=(1000*(lower_size_bound-expected_transition_size), 
            1000*(upper_size_bound-expected_transition_size)),
    hlines=None,
    labels = None,
    legend = False,
    xlim=(-200,200))

plt.show()
plt.close()
            
#%% automatic filtering

pf = np.arange(0, 40, 0.01)
pdist = ini_eWLC(1, pf, *refWLC)

for i in traces_base:
    fec_S01 = f"{i}_fecS01"
    fec_S02 = f"{i}_fecS02"
    fec_S03 = f"{i}_fecS03"
    dct_smp = [x for x in passive if i in x]

    tdLc = (
        fecs_dict[fec_S03]["dLcs_sum"][-1]
        + fecs_dict[fec_S01]["dLcs"][0]
    )

    ini_trans_index = fecs_dict[fec_S02]["trans_idxs"][0]
    end_trans_index = fecs_dict[fec_S03]["trans_idxs"][-1]

    ini_rep_force = fecs_dict[fec_S02]["flf_shifted"][ini_trans_index]
    end_rep_force = fecs_dict[fec_S03]["flf_shifted"][end_trans_index]

    if realign_begining:
        dlf = fecs_dict[fec_S01]["dlf_shifted"]
        flf = fecs_dict[fec_S01]["flf_shifted"]

        rmsd = ana.calculate_rmsd(
            dlf,
            flf,
            *refWLC,
            distance_range=ranges_for_dist_aln[0],
            force_range=[2, ranges_for_dist_aln[1][1]],
            already_segmented=False,
        )

        mask = (
            (dlf > ranges_for_dist_aln[0][0])
            & (dlf < ranges_for_dist_aln[0][1])
            & (flf > 2)
            & (flf < ranges_for_dist_aln[1][1] + 1)
        )

        dlf_segment = dlf[mask]
        flf_segment = flf[mask]

    else:
        dlf = fecs_dict[fec_S03]["dlf_shifted"]
        flf = fecs_dict[fec_S03]["flf_shifted"]

        padding = 5
        pre = fecs_dict[fec_S03]["trans_idxs"][-2]
        last = fecs_dict[fec_S03]["trans_idxs"][-1]

        rmsd = ana.calculate_rmsd(
            dlf[pre + padding:last + 1],
            flf[pre + padding:last + 1],
            *refWLC,
            already_segmented=True,
        )

        dlf_segment = dlf[pre + padding:last + 1]
        flf_segment = flf[pre + padding:last + 1]

    fecs_dict[fec_S01]["rmsd"] = rmsd

    fecs_dict[fec_S01]["error_code"] = select_fec(
        tdLc,
        ini_trans_index,
        end_trans_index,
        rmsd,
        lower_size_bound,
        upper_size_bound,
        upper_ini_force_threshold=None,
        lower_final_force_threshold=None,
        rmsd_threshold=0.3,
    )

    for c in [fec_S02, fec_S03] + dct_smp:
        fecs_dict[c]["error_code"] = fecs_dict[fec_S01]["error_code"]

    plt.xlim(0.5, 1.3)
    plt.ylim(-1, 40)

    plt.plot(
        pdist,
        pf,
        color="black",
        linestyle="--",
        label="ref_eWLC",
    )

    plt.annotate(
        f"RMSD = {rmsd:.2f}",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    plt.annotate(
        f"Lc deviation = {1000 * (tdLc - expected_transition_size):.2f} nm",
        xy=(0.98, 0.9),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    if realign_begining:
        plt.annotate(
            fec_S01,
            xy=(0.98, 0.82),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
    else:
        plt.annotate(
            fec_S03,
            xy=(0.98, 0.82),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )

    if fecs_dict[fec_S01]["error_code"] == 0:
        plt.plot(dlf, flf, label=fec_S01, color="blue")
    else:
        plt.plot(dlf, flf, label=fec_S01, color="red")
        print(fecs_dict[fec_S01]["error_code"])

    plt.plot(
        dlf_segment,
        flf_segment,
        color="green",
        label="compared segment",
    )

    plt.show()
    plt.close()

#%% manual quick  discard

i = 0
while i < len(traces_base):
    trace_base = traces_base[i]
    fec_S01 = f"{trace_base}_fecS01"
    fec_S02 = f"{trace_base}_fecS02"
    fec_S03 = f"{trace_base}_fecS03"
    error_code = fecs_dict[fec_S01]["error_code"]
    dct_smp = [x for x in passive if trace_base in x]
    set_of_traces = [fec_S01,fec_S02,fec_S03] + dct_smp
    try:
        fec_S00 = f"{trace_base}_fecS00"
        fecs_dict[fec_S00]
        set_of_traces = set_of_traces + [fec_S00]
    except:
        pass
    
    tdLc = fecs_dict[fec_S03]["dLcs_sum"][-1]+fecs_dict[fec_S01]["dLcs"][0]    
    rmsd = fecs_dict[fec_S01]["rmsd"] 
    
    error_code,continue_decision = quick_discard (fecs_dict,
                                                  set_of_traces, 
                                                  rmsd,
                                                  tdLc,
                                                  expected_transition_size,
                                                  error_code,
                                                  *refWLC, 
                                                  var_lc_component = var_component)

    plt.close("all")

    
    for c in set_of_traces:
        fecs_dict[c]["error_code"] = error_code
    
    if continue_decision == "c":
        i += 1
    elif continue_decision == "r":
        if i ==0:
            print("No previous fec to return to. Working on the same FEC.")
        else:
            i -= 1
    else:
        pass

dill.dump_session("data_consolidation/checkpoint3.pkl")

#%%
    
labels = np.array([0 if fecs_dict[x+"_fecS01"]["error_code"] == 0 else 1 for x in traces_base])
joint_scatter_with_marginals(
    alltdLcs-expected_transition_size*1000, all_ini_sliding_force,
    bins_top=24,
    bins_right=12,
    xlabel='Contour length deviation (nm)',
    ylabel='Initial rupture force (pN)',
    show_hist_ticks=False,
    vlines=[ 1000*(lower_size_bound-expected_transition_size),1000*(upper_size_bound-expected_transition_size)],
    hlines=None,
    labels = labels,
    legend = True,
    xlim=(-100,100))

plt.savefig(f"data_consolidation/{date_str}_size_filter_dev_cat.png",dpi = 300)
plt.show()
plt.close() 
    
#%%

# fecs to keep
final_traces_base = []
for i in traces_base:
    if fecs_dict[i+"_fecS01"]["error_code"] == 0:
        final_traces_base.append(i)

#baselines to keep
dfcalpars = [pd.read_csv(f"data_consolidation/data/{os.path.basename(folder)}_calpars.csv") for folder in folders]
dfcalpars = dfcalpars + [pd.read_csv(x) for x in glob.glob("data_consolidation/data/*_calpars_consolidate.csv")]
dfcalpars = pd.concat(dfcalpars)

allbaselines = list(set(dfcalpars.baseline_id))
final_baselines = []
for i in final_traces_base:
    bsls = gen_baseline_name(i)
    for c in bsls:
        if c in allbaselines:
            final_baselines.append(c)
        
#final all traces
final_alltraces = []
for i in final_traces_base:
    for c in alltraces:
        if i in c:
            final_alltraces.append(c)

#final fecs
final_fecs = [x for x in final_alltraces if "fec" in x]

#%% plotting

for i in final_traces_base:
    fec_S01 = f"{i}_fecS01"
    fec_S02 = f"{i}_fecS02"
    fec_S03 = f"{i}_fecS03"
    dct_smp = [x for x in passive if i in x]
    set_of_traces = [fec_S01,fec_S02,fec_S03] + dct_smp
    
    try:
        fec_S00 = f"{i}_fecS00"
        fecs_dict[fec_S00]
        set_of_traces = set_of_traces + [fec_S00]
    except:
        pass
    
    plot_all_PM_segments(
        fecs_dict,
        set_of_traces,
        *refWLC,
        save=True,
        show=True,
        fig_name=f"data_consolidation/consolidate_pics/{i}",
    )
    plt.close("all")

#%%

dfpost_consolidate = []
for i in final_alltraces:
    tmpdf = dfpost[dfpost["fec_id"] == i]
    tmpdf.loc[:,"molext_aln"] = fecs_dict[i]["d_shifted"]            
    tmpdf.loc[:,"diffF_aln"] = fecs_dict[i]["f_shifted"]
    dfpost_consolidate.append(tmpdf)

try:    
    dfpost_consolidate = pd.concat(dfpost_consolidate)
except:
    dfpost_consolidate = pd.DataFrame(columns = dfpost.columns)

columns_keep = ['relative_time', 'trappos1x', 'cal_trappos1x', 'molext', 'molext_aln',
               'cal_molext', 'bead1_disp',
               'bead2_disp', 'diff_bead_disp',
               'force1x', 'force2x', 'diffF',
               'diffF_aln','rupture_index', 'bead_set',
               'mol_id', 'curve_type', 'curve_number', 'date', 'error_code', 'fec_id']

dfpost_consolidate = dfpost_consolidate[columns_keep]
dfpost_consolidate.to_csv(f"data_consolidation/{date_str}_{cons_name}_postprocessed_consolidate.csv", index = False)

#%%

dfcalpars_consolidate = []
for i in final_baselines:
    tmpdf = dfcalpars[dfcalpars["baseline_id"] == i]
    dfcalpars_consolidate.append(tmpdf)

try:
    dfcalpars_consolidate = pd.concat(dfcalpars_consolidate)
except:
    dfcalpars_consolidate = pd.DataFrame(columns = dfcalpars.columns)
    dfcalpars_consolidate["keq_corrected"] = np.nan
    
dfcalpars_consolidate.to_csv (f"data_consolidation/{date_str}_{cons_name}_calpars_consolidate.csv", index = False)

#%%

dfann_consolidate = []
for i in final_fecs:

    tmpdf = pd.DataFrame()
    curve_type = fecs_dict[i]["curve_type"]
    
    if curve_type == "R":
        tmpdf["trans_idxs"] = np.flip(fecs_dict[i]["trans_idxs"])
    elif curve_type == "S":
        tmpdf["trans_idxs"] = fecs_dict[i]["trans_idxs"]
    else:
        tmpdf["trans_idxs"] = fecs_dict[i]["trans_idxs"]
     
    tmpdf["dExts"] = fecs_dict[i]["dExts"]
    tmpdf["dExts_sum"] = fecs_dict[i]["dExts_sum"]
    tmpdf["dLcs"] = fecs_dict[i]["dLcs"]
    tmpdf["dLcs_sum"] = fecs_dict[i]["dLcs_sum"]
    tmpdf["states"] = fecs_dict[i]["states_shifted"]
    tmpdf["tdExt_direct"] = fecs_dict[i]["tdExt_direct"][0]
    tmpdf["tdLc_direct"] = fecs_dict[i]["tdLc_direct"][0]
    tmpdf["fec_id"] = i
    dfann_consolidate.append(tmpdf)

dfann_consolidate = pd.concat(dfann_consolidate)
dfann_consolidate.to_csv(f"data_consolidation/{date_str}_{cons_name}_annotated_states_consolidate.csv", index = False)

#%%

# assemble extra discard
dfpre_discarded = [pd.read_csv(f"data_consolidation/data/{os.path.basename(folder)}_preprocessed_discarded.csv") for folder in folders]

try:
    dfpre_discarded = pd.concat(dfpre_discarded)
except:
    dfpre_discarded = pd.DataFrame()

dfpost_discarded = [pd.read_csv(f"data_consolidation/data/{os.path.basename(folder)}_postprocessed_discarded.csv") for folder in folders]
dfpost_discarded = dfpost_discarded + [pd.read_csv(x) for x in glob.glob("data_consolidation/data/*_discarded_consolidate.csv")]
dfpost_discarded = pd.concat(dfpost_discarded)

dfpost_extra_discarded = dfpost[~dfpost.fec_id.isin(final_alltraces)].copy()
dfpost_extra_discarded["error_code"] = dfpost_extra_discarded["fec_id"].map(
    lambda x: fecs_dict.get(x, {}).get("error_code", 0)
)

dfdiscarded_consolidate = pd.concat([dfpre_discarded, dfpost_discarded,dfpost_extra_discarded])  
dfdiscarded_consolidate.to_csv(f"data_consolidation/{date_str}_{cons_name}_discarded_consolidate.csv", index = False)

#%%
try:
    shutil.rmtree("data_consolidation/data")
except:
    pass

try:    
    os.remove("data_consolidation/checkpoint1.pkl")
except:
    pass

try:    
    os.remove("data_consolidation/checkpoint2.pkl")
except:
    pass

try:    
    os.remove("data_consolidation/checkpoint3.pkl")
except:
    pass

#%%
        
now = datetime.now()
with open (f"data_consolidation/{date_str}_{cons_name}_consolidate.txt", "w+") as f:
    f.write(cons_name + " consolidation was performed on : " + str(now) + "\n\n")
    f.write("The following parameters were used:\n")
    f.write("Expected transition size was: " + str(expected_transition_size) + " um\n")
    f.write("eWLC parameters for ini state were: " + str(WLCpini) + "\n")

    f.write("eWLC parameters for end states were: " + str(WLCpend) + "\n")
    f.write("Realign to start was: " + str(realign_begining) + "\n")
    
    if realign_begining:
        f.write(f"Force and distance ranges for distance alignment: {ranges_for_dist_aln}\n")
        f.write(f"Force and distance ranges for force alignment: {ranges_for_force_aln}\n")
    
    f.write("Curate annotation was: " + str(curate_annotation) + "\n")
    f.write("Min rupture force allowed was: " + str(rupture_force_threshold) + "pN\n")
    f.write(f"Min transition size from ini state to end state ({WLCpend[var_component][1]} um) was: {lower_size_bound}\n")
    f.write(f"Max transition size from ini state to end state ({WLCpend[var_component][1]} um) was: {upper_size_bound}\n")
    f.write("Folders used were:\n")
    for i in folders:
        f.write(os.path.basename(i) + "\n")

#%%
dill.dump_session(f"data_consolidation/{date_str}_{cons_name}_consolidate.pkl")
