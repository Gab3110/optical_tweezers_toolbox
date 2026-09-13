'''
Description  : Memory-safe consolidation of annotated force-extension curves.
               The script combines individual datasets and/or older
               consolidations, optionally realigns and reviews annotations,
               filters traces, and writes consolidated outputs.

               Large CSV files are streamed in chunks. Only one complete FEC
               is held in memory at a time. Manual decisions are checkpointed
               after every FEC, and the complete Spyder namespace is never
               serialized with dill.

Author       : Gabriel Jimenez-Avalos, Ph.D. student
Affiliation  : T. C. Jenkins Department of Biophysics, Johns Hopkins University
Version      : 5.0
Date         : 2026/09/11
'''


# %% CLEAN SESSION BEFORE START

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

_ipython = get_ipython() if get_ipython is not None else None
if _ipython is not None:
    _ipython.run_line_magic("reset", "-sf")


# %% IMPORT MODULES

import gc
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# %% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
import tweezers_toolbox_modules.postprocessing as post
import tweezers_toolbox_modules.analysis as ana
from tweezers_toolbox_modules import fec_plotting as fplt
from tweezers_toolbox_modules import input_handlers as hand
from tweezers_toolbox_modules.models import ini_eWLC

# %% DEFINE FUNCTIONS

def _source_signature(path):
    """Return information used to detect replacement of an input CSV."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_signature(value):
    """Return a deterministic signature for JSON-compatible settings."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path, value):
    """Write JSON completely before atomically replacing its destination."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _atomic_savez(path, *, compress, **arrays):
    """Save NumPy arrays without serializing the interactive namespace."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    save_function = np.savez_compressed if compress else np.savez
    with open(temporary_path, "wb") as handle:
        save_function(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_jsonable(value):
    """Convert NumPy values to objects accepted by json.dump."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    return value


def _array_from_record(record, key, dtype=float):
    return np.asarray(record[key], dtype=dtype).reshape(-1).copy()


def _unique_scalar(frame, column, fec_id):
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"FEC {fec_id!r} contains multiple {column} values: "
            f"{values.tolist()}"
        )
    return values.iloc[0]


def _iter_complete_fecs_csv(path, usecols=None, *, chunksize):
    """
    Yield complete contiguous fec_id groups from a chunked CSV reader.

    The last group of a chunk is buffered because it may continue in the next
    chunk. Repeated, noncontiguous fec_id blocks are rejected.
    """
    pending = None
    emitted_fec_ids = set()
    dtype = {"fec_id": "string", "curve_type": "string"}

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=dtype,
    ):
        if chunk.empty:
            continue
        if chunk["fec_id"].isna().any():
            raise ValueError(f"Missing fec_id values were found in {path}.")

        if pending is not None:
            data = pd.concat([pending, chunk], ignore_index=True, copy=False)
            del pending
        else:
            data = chunk

        fec_values = data["fec_id"].astype(str).to_numpy(copy=False)
        boundaries = np.flatnonzero(fec_values[1:] != fec_values[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(data)]

        for start, end in zip(starts[:-1], ends[:-1]):
            fec_id = str(fec_values[start])
            if fec_id in emitted_fec_ids:
                raise ValueError(
                    f"FEC {fec_id!r} appears in multiple noncontiguous "
                    f"blocks in {path}. Group the CSV by fec_id first."
                )
            emitted_fec_ids.add(fec_id)
            yield fec_id, data.iloc[start:end].copy()

        pending = data.iloc[starts[-1]:ends[-1]].copy()
        del data, chunk, fec_values, boundaries, starts, ends

    if pending is not None and not pending.empty:
        fec_id = str(pending["fec_id"].iloc[0])
        if fec_id in emitted_fec_ids:
            raise ValueError(
                f"FEC {fec_id!r} appears in multiple noncontiguous blocks "
                f"in {path}. Group the CSV by fec_id first."
            )
        yield fec_id, pending


def _iter_complete_fecs_sources(paths, usecols=None, *, chunksize):
    """Yield FEC groups across files and reject duplicate FEC identifiers."""
    emitted_fec_ids = set()
    for path in paths:
        for fec_id, frame in _iter_complete_fecs_csv(
            path,
            usecols=usecols,
            chunksize=chunksize,
        ):
            if fec_id in emitted_fec_ids:
                raise ValueError(
                    f"FEC {fec_id!r} occurs in more than one input file. "
                    "Remove the duplicated dataset or consolidation."
                )
            emitted_fec_ids.add(fec_id)
            yield fec_id, frame


def _required_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input CSV was not found: {path}")
    return path


def resolve_input_paths(
    folders,
    consolidation_dir,
    *,
    add_consolidate,
    excluded_outputs,
):
    """Resolve source CSVs directly, avoiding a second full copy on disk."""
    paths = {
        "pre_discarded": [],
        "post_discarded": [],
        "calpars": [],
        "video": [],
        "post": [],
        "annotation": [],
    }

    for folder in folders:
        folder = Path(folder)
        folder_base = folder.name
        paths["pre_discarded"].append(
            _required_file(
                folder / "preprocess" /
                f"{folder_base}_preprocessed_discarded.csv"
            )
        )
        paths["post_discarded"].append(
            _required_file(
                folder / "postprocess" /
                f"{folder_base}_postprocessed_discarded.csv"
            )
        )
        paths["calpars"].append(
            _required_file(
                folder / "preprocess" / f"{folder_base}_calpars.csv"
            )
        )
        paths["video"].append(
            _required_file(
                folder / "preprocess" / "video_microscopy" /
                f"{folder_base}video_microscopy.csv"
            )
        )
        paths["post"].append(
            _required_file(
                folder / "postprocess" / f"{folder_base}_postprocessed.csv"
            )
        )
        paths["annotation"].append(
            _required_file(
                folder / "annotate" /
                f"{folder_base}_annotated_states.csv"
            )
        )

    if add_consolidate:
        consolidation_dir = Path(consolidation_dir)
        patterns = {
            "post_discarded": "*_discarded_consolidate.csv",
            "calpars": "*_calpars_consolidate.csv",
            "video": "*_video_microscopy_consolidate.csv",
            "post": "*_postprocessed_consolidate.csv",
            "annotation": "*_annotated_states_consolidate.csv",
        }
        excluded = {str(Path(path).resolve()) for path in excluded_outputs}
        for category, pattern in patterns.items():
            for path in sorted(consolidation_dir.glob(pattern)):
                if str(path.resolve()) not in excluded:
                    paths[category].append(path)

    for category in ("calpars", "video", "post", "annotation"):
        if not paths[category]:
            raise FileNotFoundError(
                f"No {category} input CSVs were found. Check folders and "
                "add_consolidate."
            )

    return paths


def _read_annotation_records(paths, *, chunksize):
    """Read only small per-transition records, one FEC at a time."""
    columns = [
        "fec_id",
        "trans_idxs",
        "states",
        "dLcs_sum",
        "dLcs",
        "tdLc_direct",
    ]
    records = {}
    for fec_id, frame in _iter_complete_fecs_sources(
        paths,
        usecols=columns,
        chunksize=chunksize,
    ):
        trans_idxs = frame["trans_idxs"].to_numpy(dtype=int, copy=True)
        states = frame["states"].to_numpy(dtype=float, copy=True)
        dLcs_sum = frame["dLcs_sum"].to_numpy(dtype=float, copy=True)
        dLcs = frame["dLcs"].to_numpy(dtype=float, copy=True)

        if len(trans_idxs) == 0:
            raise ValueError(f"FEC {fec_id!r} has no annotation rows.")

        tdLc = float(dLcs_sum[-1])
        if np.isnan(tdLc):
            tdLc = float(frame["tdLc_direct"].iloc[-1])
            if np.isnan(tdLc):
                print(f"Cannot calculate tdLc for {fec_id}; retaining NaN.")

        records[fec_id] = {
            "input_trans_idxs": trans_idxs.tolist(),
            "input_states": states.tolist(),
            "input_dLcs_sum": dLcs_sum.tolist(),
            "input_dLcs": dLcs.tolist(),
            "input_tdLc": tdLc,
        }
        del frame, trans_idxs, states, dLcs_sum, dLcs

    return records


def _build_or_load_trace_cache(
    post_paths,
    annotation_paths,
    *,
    downfact,
    chunksize,
    compress,
    rebuild,
):
    """Build disk-backed per-FEC arrays using bounded memory."""
    post_sources = [_source_signature(path) for path in post_paths]
    annotation_sources = [
        _source_signature(path) for path in annotation_paths
    ]
    sources = {
        "postprocessed": post_sources,
        "annotation": annotation_sources,
        "downfact": int(downfact),
    }

    annotation_records = _read_annotation_records(
        annotation_paths,
        chunksize=chunksize,
    )

    if MANIFEST_PATH.exists() and not rebuild:
        existing = _read_json(MANIFEST_PATH)
        if existing.get("sources") != sources:
            raise RuntimeError(
                "The consolidation inputs changed after the cache was built. "
                "If that change is intentional, set "
                "REBUILD_CONSOLIDATION_CACHE = True and rerun the cache cell."
            )
        all_present = all(
            (CHECKPOINT_DIR / item["trace_file"]).exists()
            for item in existing.get("traces", [])
        )
        if all_present:
            print(
                f"Using existing consolidation cache with "
                f"{len(existing['traces'])} FECs."
            )
            return existing, annotation_records
        print("The trace cache is incomplete; rebuilding it.")

    if rebuild:
        for old_cache in TRACE_CACHE_DIR.glob("*.npz"):
            old_cache.unlink()

    required_columns = [
        "fec_id",
        "rupture_index",
        "trappos1x",
        "molext_aln",
        "diffF_aln",
        "curve_type",
    ]
    trace_manifest = []

    print("Building the per-FEC consolidation cache...")
    for trace_number, (fec_id, frame) in enumerate(
        _iter_complete_fecs_sources(
            post_paths,
            usecols=required_columns,
            chunksize=chunksize,
        )
    ):
        if fec_id not in annotation_records:
            raise KeyError(
                f"No annotation record was found for FEC {fec_id!r}."
            )

        curve_type = str(_unique_scalar(frame, "curve_type", fec_id))
        rupture_index = int(
            _unique_scalar(frame, "rupture_index", fec_id)
        )
        trappos1x = frame["trappos1x"].to_numpy(copy=True)
        d = frame["molext_aln"].to_numpy(copy=True)
        force = frame["diffF_aln"].to_numpy(copy=True)

        trappos1xlf = CK_filtfilt(trappos1x, downfact)
        dlf = CK_filtfilt(d, downfact)
        flf = CK_filtfilt(force, downfact)

        relative_path = Path("traces") / f"{trace_number:06d}.npz"
        _atomic_savez(
            CHECKPOINT_DIR / relative_path,
            compress=compress,
            trappos1x=trappos1x,
            d=d,
            force=force,
            trappos1xlf=trappos1xlf,
            dlf=dlf,
            flf=flf,
        )
        trace_manifest.append(
            {
                "fec_id": fec_id,
                "curve_type": curve_type,
                "rupture_index": rupture_index,
                "n_rows": int(len(frame)),
                "trace_file": str(relative_path),
            }
        )
        print(f"Cached {trace_number + 1}: {fec_id}")

        del (
            frame,
            trappos1x,
            d,
            force,
            trappos1xlf,
            dlf,
            flf,
        )
        gc.collect()

    missing_post = sorted(set(annotation_records) - {
        item["fec_id"] for item in trace_manifest
    })
    if missing_post:
        raise KeyError(
            "Annotations exist without postprocessed traces: "
            f"{missing_post[:10]}"
        )

    manifest = {
        "version": 1,
        "created": datetime.now().isoformat(),
        "sources": sources,
        "traces": trace_manifest,
    }
    _atomic_write_json(MANIFEST_PATH, manifest)
    print(f"Trace cache complete: {len(trace_manifest)} FECs.")
    return manifest, annotation_records


def _restore_input_annotation(record):
    """Restore the annotation imported from the source CSV."""
    record["trans_idxs"] = list(record["input_trans_idxs"])
    record["states"] = list(record["input_states"])
    record["dLcs_sum"] = list(record["input_dLcs_sum"])
    record["dLcs"] = list(record["input_dLcs"])
    record["tdLc"] = float(record["input_tdLc"])


def _clear_derived_record(record):
    keys = [
        "doff",
        "foff",
        "state_offset",
        "aligned",
        "alignment_signature",
        "annotation_curated",
        "annotation_signature",
        "annotation_curated_at",
        "dExts",
        "dExts_sum",
        "tdExt_direct",
        "tdLc_direct",
        "extensions_signature",
        "rmsd",
        "auto_error_code",
        "error_code",
        "filter_signature",
        "filter_curated",
        "filter_curation_signature",
        "filter_curated_at",
    ]
    for key in keys:
        record.pop(key, None)


def _clear_after_annotation(record):
    for key in [
        "dExts",
        "dExts_sum",
        "tdExt_direct",
        "tdLc_direct",
        "extensions_signature",
        "rmsd",
        "auto_error_code",
        "error_code",
        "filter_signature",
        "filter_curated",
        "filter_curation_signature",
        "filter_curated_at",
    ]:
        record.pop(key, None)


def _load_or_create_state(manifest, annotation_records, *, rebuild):
    """Load compact restart state or initialize it from annotations."""
    if STATE_PATH.exists() and not rebuild:
        state = _read_json(STATE_PATH)
        if state.get("sources") != manifest["sources"]:
            raise RuntimeError(
                "The saved consolidation state belongs to different inputs. "
                "Set REBUILD_CONSOLIDATION_CACHE = True only when you intend "
                "to restart with the current inputs."
            )
    else:
        state = {
            "version": 1,
            "sources": manifest["sources"],
            "stage_signatures": {},
            "traces": {},
        }

    for item in manifest["traces"]:
        fec_id = item["fec_id"]
        imported = annotation_records[fec_id]
        record = state["traces"].setdefault(fec_id, {})
        for key, value in imported.items():
            record.setdefault(key, value)
        if "trans_idxs" not in record:
            _restore_input_annotation(record)
        record.setdefault("error_code", 0)

    _atomic_write_json(STATE_PATH, state)
    return state


def _load_trace_arrays(trace_info, *keys):
    path = CHECKPOINT_DIR / trace_info["trace_file"]
    with np.load(path, allow_pickle=False) as saved:
        return tuple(saved[key].copy() for key in keys)


def unpack_trans_obj(trans_obj):
    total_path = trans_obj.total_direct_path
    tdExt_direct = trans_obj.all_trans_info[total_path]["dExt"]
    tdLc_direct = trans_obj.all_trans_info[total_path]["dLc"]
    paths = list(trans_obj.all_trans_info)

    if len(paths) != 1:
        paths = [path for path in paths if path != total_path]

    dExts = [trans_obj.all_trans_info[path]["dExt"] for path in paths]
    dLcs = [trans_obj.all_trans_info[path]["dLc"] for path in paths]

    dExts.append(0)
    dLcs.append(0)
    dExts_sum = np.cumsum([0] + dExts[:-1]).tolist()
    dLcs_sum = np.cumsum([0] + dLcs[:-1]).tolist()
    dLcs_sum_complement = np.cumsum(
        [0] + list(np.flip(dLcs[:-1]))
    ).tolist()

    return (
        dExts,
        dLcs,
        dExts_sum,
        dLcs_sum,
        dLcs_sum_complement,
        tdExt_direct,
        tdLc_direct,
    )


def select_fec(
    tdLc,
    ini_force,
    end_force,
    rmsd,
    lthreshold,
    uthreshold,
    upper_ini_force_threshold=None,
    lower_final_force_threshold=None,
    rmsd_threshold=0.4,
):
    failures = []

    if not (lthreshold <= tdLc <= uthreshold):
        failures.append(6)
    if upper_ini_force_threshold is not None:
        if not ini_force < upper_ini_force_threshold:
            failures.append(7)
    if lower_final_force_threshold is not None:
        if not end_force > lower_final_force_threshold:
            failures.append(8)
    if not rmsd < rmsd_threshold:
        failures.append(9)

    return 0 if not failures else failures


def gen_baseline_name(fec_id):
    underscore_ids = [match.start() for match in re.finditer("_", fec_id)]
    if len(underscore_ids) < 2:
        raise ValueError(
            f"Cannot derive a baseline ID from FEC name {fec_id!r}."
        )
    date = fec_id[:underscore_ids[0]]
    bead_set = fec_id[underscore_ids[0] + 1:underscore_ids[1]]
    baseline_path_name = f"{date}_{bead_set}_fec00.h5"
    baseline_id = Path(baseline_path_name).stem
    return baseline_path_name, baseline_id


def quick_discard(
    fec_id,
    rmsd,
    lc_dev,
    exp_trans_size,
    dlf,
    flf,
    d,
    force,
    trans_idxs,
    states,
    error_code,
    plot_bounds,
    *WLCp_basal,
    var_lc_component=0,
):
    fec_color = "black" if error_code == 0 else "red"
    fplt.plot_individual_fec(
        dlf,
        flf,
        fec_id,
        fec_color,
        plot_bounds[0],
        plot_bounds[1],
        "",
        *WLCp_basal,
        var_component=var_lc_component,
        distances_hf=d,
        forces_hf=force,
        scatter=False,
        annot=(
            f"RMSD = {rmsd:.3f}\n"
            f"Lc deviation= {1000 * (lc_dev - exp_trans_size):.3f} nm"
        ),
        savefig=False,
        rupture_index=False,
        lcstates=states,
        states_end_indexes=trans_idxs,
        lc_trajectory=False,
    )

    user_input = hand.option_handler(
        "keep (k), discard (d), pass (p), or return (r)? ",
        valid_values=["k", "d", "r", "p"],
    )

    if user_input == "k":
        return 0, "c"
    if user_input == "d":
        return 5, "c"
    if user_input == "p":
        return error_code, "c"
    return error_code, "r"


def joint_scatter_with_marginals(
    x,
    y,
    *,
    labels=None,
    legend=False,
    noise_label="noise",
    bins_top=12,
    bins_right=12,
    figsize=(7, 6),
    title=None,
    xlabel="X",
    ylabel="Y",
    lw_axes=2,
    lw_hist=2,
    fontsize_labels=22,
    fontsize_ticks=20,
    grid_alpha=0.45,
    grid_linewidth=0.8,
    show_hist_ticks=True,
    vlines=None,
    hlines=None,
    linecolor="red",
    linestyle="--",
    linewidth=1.8,
    noise_color="lightgray",
    xlim=None,
    ylim=None,
):
    """Create a scatter plot with top and right marginal histograms."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    label_to_color = {}
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != x.shape:
            raise ValueError("labels must have the same shape as x and y.")
        base_cmap = plt.get_cmap("tab10")
        for label in np.unique(labels):
            label_to_color[label] = (
                noise_color
                if label == -1
                else base_cmap(int(label) % base_cmap.N)
            )
        colors = [label_to_color[label] for label in labels]
    else:
        colors = "black"

    fig = plt.figure(figsize=figsize)
    grid = GridSpec(4, 4, figure=fig, hspace=0, wspace=0)
    ax_scatter = fig.add_subplot(grid[1:4, 0:3])
    ax_histx = fig.add_subplot(grid[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(grid[1:4, 3], sharey=ax_scatter)

    ax_scatter.scatter(x, y, c=colors, s=25, alpha=0.9, edgecolor="none")
    ax_scatter.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax_scatter.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax_scatter.tick_params(
        axis="both",
        which="both",
        labelsize=fontsize_ticks,
        width=lw_axes,
        length=6,
        direction="inout",
    )
    ax_scatter.grid(
        True,
        linestyle="--",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
    )
    for spine in ax_scatter.spines.values():
        spine.set_linewidth(lw_axes)

    if xlim is not None:
        ax_scatter.set_xlim(xlim)
    if ylim is not None:
        ax_scatter.set_ylim(ylim)
    if vlines is not None:
        for x_value in np.atleast_1d(vlines):
            ax_scatter.axvline(
                x=x_value,
                color=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
            )
    if hlines is not None:
        for y_value in np.atleast_1d(hlines):
            ax_scatter.axhline(
                y=y_value,
                color=linecolor,
                linestyle=linestyle,
                linewidth=linewidth,
            )

    ax_histx.hist(
        x,
        bins=bins_top,
        color="gray",
        histtype="stepfilled",
        edgecolor="black",
        linewidth=lw_hist,
    )
    ax_histx.grid(
        True,
        linestyle=":",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
    )
    ax_histx.tick_params(axis="x", bottom=False, labelbottom=False)
    if show_hist_ticks:
        ax_histx.tick_params(
            axis="y", labelsize=fontsize_ticks, width=lw_axes, length=5
        )
    else:
        ax_histx.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax_histx.spines.values():
        spine.set_linewidth(lw_axes)

    ax_histy.hist(
        y,
        bins=bins_right,
        color="gray",
        histtype="stepfilled",
        edgecolor="black",
        linewidth=lw_hist,
        orientation="horizontal",
    )
    ax_histy.grid(
        True,
        linestyle=":",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
    )
    ax_histy.tick_params(axis="y", left=False, labelleft=False)
    if show_hist_ticks:
        ax_histy.tick_params(
            axis="x", labelsize=fontsize_ticks, width=lw_axes, length=5
        )
    else:
        ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_histy.spines.values():
        spine.set_linewidth(lw_axes)

    if legend and labels is not None:
        handles = []
        names = []
        for label in np.unique(labels):
            color = label_to_color[label]
            name = noise_label if label == -1 else f"cluster {label}"
            handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markersize=10,
                markeredgewidth=0,
            )
            handles.append(handle)
            names.append(name)
        fig.legend(
            handles,
            names,
            loc="center",
            ncol=min(2, len(handles)),
            fontsize=16,
            frameon=True,
            bbox_to_anchor=(0.4, -0.05),
        )
        bottom = 0.2
    else:
        bottom = 0.12

    plt.subplots_adjust(bottom=bottom, hspace=0, wspace=0)
    if title:
        fig.suptitle(title, fontsize=fontsize_labels, y=0.98)
    return fig, ax_scatter, ax_histx, ax_histy


def get_alignment_offsets(
    dlf,
    flf,
    dist_bounds,
    force_bounds,
    WLCpars,
    *,
    distance_aln_indexes=None,
    only_dist_aln=False,
    first_transition=None,
):
    alignment = post.AlignFEC(
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
        alignment.iterative_dist_aln()
    else:
        alignment.iterative_fd_aln()

    doff = float(alignment.x_aln[0] - dlf[0])
    foff = float(alignment.F_aln[0] - flf[0])
    return doff, foff


def get_pre_rupture_window(trans_idxs, offsets):
    rupture = int(trans_idxs[-1])
    for offset in offsets:
        pre_rupture = int(trans_idxs[-2] + offset)
        if pre_rupture <= rupture:
            return pre_rupture, rupture
    return int(trans_idxs[-2] + offsets[-1]), rupture


def _shifted_trace_arrays(trace_info, record):
    """Load one FEC and apply its scalar alignment offsets in place."""
    d, force, dlf, flf = _load_trace_arrays(
        trace_info,
        "d",
        "force",
        "dlf",
        "flf",
    )
    d += float(record["doff"])
    dlf += float(record["doff"])
    force += float(record["foff"])
    flf += float(record["foff"])
    return d, force, dlf, flf


def _plot_shifted_fec(
    fec_id,
    dlf,
    flf,
    WLCpars,
    target_state,
    *,
    plot_bounds,
    output_path,
    var_lc_component,
):
    fplt.plot_individual_fec(
        dlf,
        flf,
        fec_id,
        "black",
        plot_bounds[0],
        plot_bounds[1],
        str(output_path),
        *WLCpars,
        var_component=var_lc_component,
        lcstates=[target_state],
    )
    plt.close("all")


def run_alignment(
    *,
    realign_begining,
    realign_end_all,
    WLCpini,
    WLCpend,
    var_lc_component,
    ranges_for_dist_aln,
    ranges_for_force_aln,
    plot_bounds,
    pre_rupture_offsets,
    force=False,
):
    """Align and checkpoint each FEC while keeping one trace in RAM."""
    signature = _stable_signature(
        {
            "sources": manifest["sources"],
            "realign_begining": realign_begining,
            "realign_end_all": realign_end_all,
            "WLCpini": WLCpini,
            "WLCpend": WLCpend,
            "var_lc_component": var_lc_component,
            "ranges_for_dist_aln": ranges_for_dist_aln,
            "ranges_for_force_aln": ranges_for_force_aln,
            "pre_rupture_offsets": pre_rupture_offsets,
        }
    )
    old_signature = consolidation_state["stage_signatures"].get("alignment")
    if force or old_signature != signature:
        for record in consolidation_state["traces"].values():
            _clear_derived_record(record)
            _restore_input_annotation(record)
            record["error_code"] = 0
        consolidation_state["stage_signatures"]["alignment"] = signature
        _atomic_write_json(STATE_PATH, consolidation_state)

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = consolidation_state["traces"][fec_id]
        if record.get("alignment_signature") == signature:
            continue

        dlf, flf = _load_trace_arrays(trace_info, "dlf", "flf")
        input_states = _array_from_record(record, "input_states")
        trans_idxs = _array_from_record(record, "input_trans_idxs", dtype=int)
        if len(input_states) == 0 or len(trans_idxs) == 0:
            raise ValueError(f"FEC {fec_id!r} has an empty annotation.")

        if realign_begining:
            target_state = float(WLCpini[var_lc_component][1])
            state_offset = target_state - float(input_states[0])
            doff, foff = get_alignment_offsets(
                dlf,
                flf,
                ranges_for_dist_aln,
                ranges_for_force_aln,
                WLCpini,
                first_transition=int(trans_idxs[0]),
            )
            plot_WLC = WLCpini
        else:
            target_state = float(WLCpend[var_lc_component][1])
            state_offset = target_state - float(input_states[-1])
            if not realign_end_all and input_states[-1] == target_state:
                doff = 0.0
                foff = 0.0
                state_offset = 0.0
            else:
                if len(trans_idxs) < 2:
                    raise ValueError(
                        f"FEC {fec_id!r} needs at least two transitions for "
                        "end alignment."
                    )
                pre_rupture, rupture = get_pre_rupture_window(
                    trans_idxs,
                    pre_rupture_offsets,
                )
                doff, foff = get_alignment_offsets(
                    dlf,
                    flf,
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    WLCpend,
                    distance_aln_indexes=[pre_rupture, rupture],
                    only_dist_aln=True,
                )
            plot_WLC = WLCpend

        record["doff"] = float(doff)
        record["foff"] = float(foff)
        record["state_offset"] = float(state_offset)
        record["states"] = (input_states + state_offset).tolist()
        record["aligned"] = True
        record["alignment_signature"] = signature
        _atomic_write_json(STATE_PATH, consolidation_state)

        dlf += doff
        flf += foff
        _plot_shifted_fec(
            fec_id,
            dlf,
            flf,
            plot_WLC,
            target_state,
            plot_bounds=plot_bounds,
            output_path=ALIGNMENT_PICS_DIR / f"{fec_id}.png",
            var_lc_component=var_lc_component,
        )
        print(f"Alignment checkpoint saved: {fec_id}")
        del dlf, flf, input_states, trans_idxs
        gc.collect()


def _start_index(flag_key, start_index, stage_label):
    if start_index is None:
        unfinished = [
            index
            for index, fec_id in enumerate(fecs)
            if not consolidation_state["traces"][fec_id].get(flag_key, False)
        ]
        if unfinished:
            return unfinished[0]
        print(
            f"All FECs have completed {stage_label}. Starting a complete "
            "review from the first FEC while retaining saved decisions."
        )
        return 0
    if isinstance(start_index, str):
        if start_index not in trace_info_by_id:
            raise KeyError(f"Unknown fec_id: {start_index}")
        return fecs.index(start_index)

    index = int(start_index)
    if index < 0 or index >= len(fecs):
        raise IndexError(f"start_index must be from 0 to {len(fecs) - 1}.")
    return index


def curate_annotations(
    parameters,
    plot_bounds,
    refWLC,
    *,
    var_lc_component,
    start_index=None,
    force=False,
):
    """Review state annotations and checkpoint after every FEC."""
    alignment_signature = consolidation_state["stage_signatures"].get(
        "alignment"
    )
    if alignment_signature is None:
        raise RuntimeError("Run the alignment cell before annotation curation.")

    signature = _stable_signature(
        {
            "alignment_signature": alignment_signature,
            "parameters": parameters,
            "plot_bounds": plot_bounds,
            "refWLC": refWLC,
            "var_lc_component": var_lc_component,
        }
    )
    old_signature = consolidation_state["stage_signatures"].get(
        "annotation_curation"
    )
    if force or old_signature != signature:
        for record in consolidation_state["traces"].values():
            record["trans_idxs"] = list(record["input_trans_idxs"])
            record["states"] = (
                np.asarray(record["input_states"], dtype=float)
                + float(record["state_offset"])
            ).tolist()
            record["dLcs_sum"] = list(record["input_dLcs_sum"])
            record["dLcs"] = list(record["input_dLcs"])
            record["tdLc"] = float(record["input_tdLc"])
            record.pop("annotation_curated", None)
            record.pop("annotation_signature", None)
            record.pop("annotation_curated_at", None)
            _clear_after_annotation(record)
        consolidation_state["stage_signatures"][
            "annotation_curation"
        ] = signature
        _atomic_write_json(STATE_PATH, consolidation_state)

    current_index = _start_index(
        "annotation_curated",
        start_index,
        "annotation curation",
    )

    while current_index < len(fecs):
        fec_id = fecs[current_index]
        trace_info = trace_info_by_id[fec_id]
        record = consolidation_state["traces"][fec_id]
        d, force_array, dlf, flf = _shifted_trace_arrays(trace_info, record)
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        states = _array_from_record(record, "states")

        editor = ana.EditStates(
            fec_id,
            dlf,
            flf,
            d,
            force_array,
            trans_idxs,
            parameters,
            states,
            refWLC[var_lc_component][1],
            record.get("error_code", 0),
            plot_bounds,
            *refWLC,
            var_lc_component=var_lc_component,
        )
        continue_decision = editor.run()

        new_trans_idxs = np.asarray(editor.trans_idxs, dtype=int).reshape(-1)
        new_states = np.asarray(editor.states, dtype=float).reshape(-1)
        new_dLcs_sum = np.asarray(editor.dLcs_sum, dtype=float).reshape(-1)
        new_dLcs = np.asarray(editor.dLcs, dtype=float).reshape(-1)

        record["trans_idxs"] = new_trans_idxs.tolist()
        record["states"] = new_states.tolist()
        record["dLcs_sum"] = new_dLcs_sum.tolist()
        record["dLcs"] = new_dLcs.tolist()
        if len(new_dLcs_sum) and new_dLcs_sum[-1]:
            record["tdLc"] = float(new_dLcs_sum[-1])
        record["annotation_curated"] = True
        record["annotation_signature"] = signature
        record["annotation_curated_at"] = datetime.now().isoformat()
        _clear_after_annotation(record)
        _atomic_write_json(STATE_PATH, consolidation_state)

        print(
            f"Annotation checkpoint saved: {fec_id} "
            f"({current_index + 1}/{len(fecs)})"
        )
        del (
            d,
            force_array,
            dlf,
            flf,
            trans_idxs,
            states,
            new_trans_idxs,
            new_states,
            new_dLcs_sum,
            new_dLcs,
            editor,
        )
        plt.close("all")
        gc.collect()

        if continue_decision == "c":
            current_index += 1
        elif continue_decision == "r":
            if current_index == 0:
                print("No previous FEC. Remaining on the first FEC.")
            else:
                current_index -= 1
        else:
            print(
                f"Unrecognized decision {continue_decision!r}; remaining on "
                "the current FEC."
            )


def calculate_transition_extensions(trans_params, *, force=False):
    """Calculate and checkpoint extensions without retaining trans_obj."""
    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = consolidation_state["traces"][fec_id]
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        local_signature = _stable_signature(
            {
                "trans_idxs": trans_idxs.tolist(),
                "trans_params": trans_params,
                "trace_source": trace_info["trace_file"],
            }
        )
        if not force and record.get("extensions_signature") == local_signature:
            continue

        trappos1xlf, flf = _load_trace_arrays(
            trace_info,
            "trappos1xlf",
            "flf",
        )
        analysis_trans_idxs = trans_idxs.copy()
        if trace_info["curve_type"] == "R":
            trappos1xlf = np.flip(trappos1xlf)
            flf = np.flip(flf)
            analysis_trans_idxs = len(flf) - 1 - analysis_trans_idxs
        analysis_trans_idxs = np.sort(analysis_trans_idxs)

        if len(analysis_trans_idxs) > 1:
            trans_obj = ana.FindTransExtensions(
                fec_id,
                trappos1xlf,
                flf,
                analysis_trans_idxs,
                trans_params=trans_params,
                debug=False,
            )
            trans_obj.run()
            (
                dExts,
                _,
                dExts_sum,
                _,
                _,
                tdExt_direct,
                tdLc_direct,
            ) = unpack_trans_obj(trans_obj)
        else:
            trans_obj = None
            dExts = 0.0
            dExts_sum = 0.0
            tdExt_direct = 0.0
            tdLc_direct = 0.0

        record["dExts"] = _as_jsonable(dExts)
        record["dExts_sum"] = _as_jsonable(dExts_sum)
        record["tdExt_direct"] = _as_jsonable(tdExt_direct)
        record["tdLc_direct"] = _as_jsonable(tdLc_direct)
        record["extensions_signature"] = local_signature
        _atomic_write_json(STATE_PATH, consolidation_state)
        print(f"Extension checkpoint saved: {fec_id}")

        del (
            trans_idxs,
            trappos1xlf,
            flf,
            analysis_trans_idxs,
            trans_obj,
        )
        gc.collect()


def get_filter_summary_arrays():
    tdLcs = []
    initial_forces = []
    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = consolidation_state["traces"][fec_id]
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        if len(trans_idxs) == 0:
            raise ValueError(f"FEC {fec_id!r} has no transition indexes.")
        (flf,) = _load_trace_arrays(trace_info, "flf")
        tdLcs.append(float(_array_from_record(record, "dLcs_sum")[-1]))
        initial_forces.append(float(flf[trans_idxs[0]]))
        del trans_idxs, flf
    return np.asarray(tdLcs) * 1000, np.asarray(initial_forces)


def run_automatic_filtering(
    refWLC,
    *,
    expected_transition_size,
    realign_begining,
    ranges_for_dist_aln,
    lower_size_bound,
    upper_size_bound,
    upper_ini_force_threshold,
    lower_final_force_threshold,
    rmsd_threshold,
    show_plots,
    plot_bounds,
    force=False,
):
    """Calculate automatic filters one trace at a time."""
    model_force = np.arange(0, 40, 0.01)
    model_distance = ini_eWLC(1, model_force, *refWLC)

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = consolidation_state["traces"][fec_id]
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        dLcs_sum = _array_from_record(record, "dLcs_sum")
        local_signature = _stable_signature(
            {
                "trans_idxs": trans_idxs.tolist(),
                "dLcs_sum": dLcs_sum.tolist(),
                "doff": record["doff"],
                "foff": record["foff"],
                "refWLC": refWLC,
                "realign_begining": realign_begining,
                "ranges_for_dist_aln": ranges_for_dist_aln,
                "lower_size_bound": lower_size_bound,
                "upper_size_bound": upper_size_bound,
                "upper_ini_force_threshold": upper_ini_force_threshold,
                "lower_final_force_threshold": lower_final_force_threshold,
                "rmsd_threshold": rmsd_threshold,
            }
        )

        already_done = (
            not force and record.get("filter_signature") == local_signature
        )
        if already_done and not show_plots:
            continue

        d, force_array, dlf, flf = _shifted_trace_arrays(trace_info, record)
        if len(trans_idxs) == 0:
            raise ValueError(f"FEC {fec_id!r} has no transition indexes.")

        tdLc = float(dLcs_sum[-1])
        # Preserve the original behavior: representative forces come from the
        # unshifted low-bandwidth force channel. They currently do not affect
        # filtering because both force thresholds are None in the call below.
        (unshifted_flf,) = _load_trace_arrays(trace_info, "flf")
        initial_force = float(unshifted_flf[trans_idxs[0]])
        final_force = float(unshifted_flf[trans_idxs[-1]])

        if realign_begining:
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
            if len(trans_idxs) < 2:
                raise ValueError(
                    f"FEC {fec_id!r} needs two transitions for end-state "
                    "RMSD calculation."
                )
            padding = 5
            previous, last = trans_idxs[-2], trans_idxs[-1]
            dlf_segment = dlf[previous + padding:last + 1]
            flf_segment = flf[previous + padding:last + 1]
            rmsd = ana.calculate_rmsd(
                dlf_segment,
                flf_segment,
                *refWLC,
                already_segmented=True,
            )

        auto_error_code = select_fec(
            tdLc,
            initial_force,
            final_force,
            rmsd,
            lower_size_bound,
            upper_size_bound,
            upper_ini_force_threshold=upper_ini_force_threshold,
            lower_final_force_threshold=lower_final_force_threshold,
            rmsd_threshold=rmsd_threshold,
        )

        if not already_done:
            record["tdLc"] = tdLc
            record["rmsd"] = float(rmsd)
            record["auto_error_code"] = _as_jsonable(auto_error_code)
            record["error_code"] = _as_jsonable(auto_error_code)
            record["filter_signature"] = local_signature
            record.pop("filter_curated", None)
            record.pop("filter_curation_signature", None)
            record.pop("filter_curated_at", None)
            _atomic_write_json(STATE_PATH, consolidation_state)
            print(f"Automatic-filter checkpoint saved: {fec_id}")

        if show_plots:
            plt.figure()
            plt.xlim(plot_bounds[0])
            plt.ylim(plot_bounds[1])
            plt.plot(
                model_distance,
                model_force,
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
            plt.annotate(
                fec_id,
                xy=(0.98, 0.82),
                xycoords="axes fraction",
                ha="right",
                va="top",
            )
            color = "blue" if auto_error_code == 0 else "red"
            plt.plot(dlf, flf, label=fec_id, color=color)
            plt.plot(
                dlf_segment,
                flf_segment,
                color="green",
                label="compared segment",
            )
            plt.show()
            plt.close("all")

        del (
            trans_idxs,
            dLcs_sum,
            d,
            force_array,
            dlf,
            flf,
            unshifted_flf,
            dlf_segment,
            flf_segment,
        )
        gc.collect()


def curate_filtering(
    plot_bounds,
    refWLC,
    *,
    expected_transition_size,
    var_lc_component,
    start_index=None,
    force=False,
):
    """Review keep/discard decisions and checkpoint every response."""
    current_index = _start_index(
        "filter_curated",
        start_index,
        "filter curation",
    )

    while current_index < len(fecs):
        fec_id = fecs[current_index]
        trace_info = trace_info_by_id[fec_id]
        record = consolidation_state["traces"][fec_id]
        if "filter_signature" not in record:
            raise RuntimeError(
                f"FEC {fec_id!r} has not completed automatic filtering."
            )
        curation_signature = _stable_signature(
            {
                "filter_signature": record["filter_signature"],
                "plot_bounds": plot_bounds,
                "refWLC": refWLC,
                "expected_transition_size": expected_transition_size,
                "var_lc_component": var_lc_component,
            }
        )
        if force or record.get(
            "filter_curation_signature"
        ) != curation_signature:
            record["error_code"] = record["auto_error_code"]
            record.pop("filter_curated", None)

        d, force_array, dlf, flf = _shifted_trace_arrays(trace_info, record)
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        states = _array_from_record(record, "states")
        error_code, continue_decision = quick_discard(
            fec_id,
            float(record["rmsd"]),
            float(record["tdLc"]),
            expected_transition_size,
            dlf,
            flf,
            d,
            force_array,
            trans_idxs,
            states,
            record["error_code"],
            plot_bounds,
            *refWLC,
            var_lc_component=var_lc_component,
        )
        record["error_code"] = _as_jsonable(error_code)
        record["filter_curated"] = True
        record["filter_curation_signature"] = curation_signature
        record["filter_curated_at"] = datetime.now().isoformat()
        _atomic_write_json(STATE_PATH, consolidation_state)

        print(
            f"Filter checkpoint saved: {fec_id} "
            f"({current_index + 1}/{len(fecs)})"
        )
        del d, force_array, dlf, flf, trans_idxs, states
        plt.close("all")
        gc.collect()

        if continue_decision == "c":
            current_index += 1
        elif continue_decision == "r":
            if current_index == 0:
                print("No previous FEC. Remaining on the first FEC.")
            else:
                current_index -= 1
        else:
            print(
                f"Unrecognized decision {continue_decision!r}; remaining on "
                "the current FEC."
            )


def get_final_fecs_and_baselines():
    final_fecs = [
        fec_id
        for fec_id in fecs
        if consolidation_state["traces"][fec_id].get("error_code") == 0
    ]
    final_baselines = []
    seen = set()
    for fec_id in final_fecs:
        _, baseline_id = gen_baseline_name(fec_id)
        if baseline_id not in seen:
            seen.add(baseline_id)
            final_baselines.append(baseline_id)
    return final_fecs, final_baselines


def save_final_plots(
    final_fecs,
    refWLC,
    plot_bounds,
    *,
    var_lc_component,
):
    """Save accepted FEC plots one at a time."""
    for fec_id in final_fecs:
        trace_info = trace_info_by_id[fec_id]
        record = consolidation_state["traces"][fec_id]
        d, force_array, dlf, flf = _shifted_trace_arrays(trace_info, record)
        states = _array_from_record(record, "states")
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)

        fplt.plot_individual_fec(
            dlf,
            flf,
            fec_id,
            "black",
            plot_bounds[0],
            plot_bounds[1],
            str(CONSOLIDATE_PICS_DIR / f"{fec_id}.png"),
            *refWLC,
            var_component=var_lc_component,
            distances_hf=d,
            forces_hf=force_array,
            scatter=False,
            annot=False,
            savefig=True,
            rupture_index=False,
            lcstates=states,
            states_end_indexes=trans_idxs,
            lc_trajectory=False,
        )
        plt.close("all")
        print(f"Saved final plot: {fec_id}")
        del d, force_array, dlf, flf, states, trans_idxs
        gc.collect()


def _union_csv_columns(paths):
    columns = []
    seen = set()
    for path in paths:
        for column in pd.read_csv(path, nrows=0).columns:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def _temporary_csv_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".tmp")


def _initialize_csv(path, columns):
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _append_csv(frame, path, columns):
    if frame.empty:
        return
    frame.reindex(columns=columns).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
    )


def _broadcast_annotation_value(value, length, column):
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(length, array.item())
    array = array.reshape(-1)
    if len(array) != length:
        raise ValueError(
            f"Annotation column {column!r} has {len(array)} values; "
            f"expected {length}."
        )
    return array


def write_consolidated_csvs(
    input_paths,
    final_fecs,
    final_baselines,
    output_paths,
    *,
    chunksize,
):
    """Build all final CSVs in temporary files using bounded memory."""
    final_fec_set = set(final_fecs)
    final_baseline_set = set(final_baselines)
    temporary_paths = {
        key: _temporary_csv_path(path) for key, path in output_paths.items()
    }

    calpars_columns = _union_csv_columns(input_paths["calpars"])
    if "keq_corrected" not in calpars_columns:
        calpars_columns.append("keq_corrected")
    _initialize_csv(temporary_paths["calpars"], calpars_columns)
    for path in input_paths["calpars"]:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            selected = chunk[
                chunk["baseline_id"].astype(str).isin(final_baseline_set)
            ].copy()
            selected["keq_corrected"] = np.nan
            _append_csv(selected, temporary_paths["calpars"], calpars_columns)
            del chunk, selected

    video_columns = _union_csv_columns(input_paths["video"])
    _initialize_csv(temporary_paths["video"], video_columns)
    for path in input_paths["video"]:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            selected = chunk[
                chunk["baseline_id"].astype(str).isin(final_baseline_set)
            ].copy()
            _append_csv(selected, temporary_paths["video"], video_columns)
            del chunk, selected

    post_columns = _union_csv_columns(input_paths["post"])
    _initialize_csv(temporary_paths["post"], post_columns)
    doff_by_fec = {
        fec_id: float(consolidation_state["traces"][fec_id]["doff"])
        for fec_id in final_fecs
    }
    foff_by_fec = {
        fec_id: float(consolidation_state["traces"][fec_id]["foff"])
        for fec_id in final_fecs
    }
    for path in input_paths["post"]:
        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            dtype={"fec_id": "string"},
        ):
            selected = chunk[chunk["fec_id"].isin(final_fec_set)].copy()
            if not selected.empty:
                selected["molext_aln"] = (
                    selected["molext_aln"].to_numpy()
                    + selected["fec_id"].map(doff_by_fec).to_numpy()
                )
                selected["diffF_aln"] = (
                    selected["diffF_aln"].to_numpy()
                    + selected["fec_id"].map(foff_by_fec).to_numpy()
                )
            _append_csv(selected, temporary_paths["post"], post_columns)
            del chunk, selected

    annotation_columns = [
        "trans_idxs",
        "dExts",
        "dExts_sum",
        "dLcs",
        "dLcs_sum",
        "states",
        "tdExt_direct",
        "tdLc_direct",
        "fec_id",
    ]
    _initialize_csv(temporary_paths["annotation"], annotation_columns)
    for fec_id in final_fecs:
        record = consolidation_state["traces"][fec_id]
        for required in (
            "dExts",
            "dExts_sum",
            "tdExt_direct",
            "tdLc_direct",
        ):
            if required not in record:
                raise RuntimeError(
                    f"FEC {fec_id!r} is missing {required}; run the "
                    "transition-extension cell first."
                )
        trans_idxs = _array_from_record(record, "trans_idxs", dtype=int)
        if trace_info_by_id[fec_id]["curve_type"] == "R":
            trans_idxs = np.flip(trans_idxs)
        length = len(trans_idxs)
        frame = pd.DataFrame({"trans_idxs": trans_idxs})
        values = {
            "dExts": record["dExts"],
            "dExts_sum": record["dExts_sum"],
            "dLcs": record["dLcs"],
            "dLcs_sum": record["dLcs_sum"],
            "states": record["states"],
            "tdExt_direct": record["tdExt_direct"],
            "tdLc_direct": record["tdLc_direct"],
        }
        for column, value in values.items():
            frame[column] = _broadcast_annotation_value(
                value,
                length,
                column,
            )
        frame["fec_id"] = fec_id
        _append_csv(
            frame,
            temporary_paths["annotation"],
            annotation_columns,
        )
        del frame, trans_idxs

    discarded_sources = (
        input_paths["pre_discarded"] + input_paths["post_discarded"]
    )
    discarded_columns = _union_csv_columns(
        discarded_sources + input_paths["post"]
    )
    if "error_code" not in discarded_columns:
        discarded_columns.append("error_code")
    _initialize_csv(temporary_paths["discarded"], discarded_columns)

    for path in discarded_sources:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            _append_csv(
                chunk,
                temporary_paths["discarded"],
                discarded_columns,
            )
            del chunk

    error_by_fec = {
        fec_id: consolidation_state["traces"][fec_id]["error_code"]
        for fec_id in fecs
    }
    for path in input_paths["post"]:
        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            dtype={"fec_id": "string"},
        ):
            rejected = chunk[~chunk["fec_id"].isin(final_fec_set)].copy()
            if not rejected.empty:
                rejected["error_code"] = rejected["fec_id"].map(error_by_fec)
            _append_csv(
                rejected,
                temporary_paths["discarded"],
                discarded_columns,
            )
            del chunk, rejected

    for key, output_path in output_paths.items():
        os.replace(temporary_paths[key], output_path)
        print(f"Saved: {output_path}")


def save_metadata(
    metadata_path,
    *,
    cons_name,
    expected_transition_size,
    refWLC,
    realign_begining,
    ranges_for_dist_aln,
    ranges_for_force_aln,
    curate_annotation,
    rupture_force_threshold,
    lower_size_bound,
    upper_size_bound,
    folders,
    input_paths,
):
    """Save settings and the exact source file list."""
    with open(metadata_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"{cons_name} consolidation was performed on: "
            f"{datetime.now()}\n\n"
        )
        handle.write("The following parameters were used:\n")
        handle.write(
            f"Expected transition size was: {expected_transition_size} um\n"
        )
        handle.write(f"Reference eWLC parameters were: {refWLC}\n")
        handle.write(f"Realign to start was: {realign_begining}\n")
        if realign_begining:
            handle.write(
                "Force and distance ranges for distance alignment: "
                f"{ranges_for_dist_aln}\n"
            )
            handle.write(
                "Force and distance ranges for force alignment: "
                f"{ranges_for_force_aln}\n"
            )
        handle.write(f"Curate annotation was: {curate_annotation}\n")
        handle.write(
            f"Min rupture force allowed was: {rupture_force_threshold} pN\n"
        )
        handle.write(
            "Min transition size from initial to end state was: "
            f"{lower_size_bound}\n"
        )
        handle.write(
            "Max transition size from initial to end state was: "
            f"{upper_size_bound}\n"
        )
        handle.write("Folders requested were:\n")
        for folder in folders:
            handle.write(f"{Path(folder).name}\n")
        handle.write("\nExact source CSVs were:\n")
        for category, paths in input_paths.items():
            for path in paths:
                handle.write(f"{category}: {Path(path).resolve()}\n")
        handle.write(
            "\nPer-FEC recovery state was retained at:\n"
            f"{STATE_PATH.resolve()}\n"
        )
    print(f"Saved: {metadata_path}")


# %% DEFINE GENERAL VARIABLES

# Hex alignment: (0.34 * (985 + 1016 + 40) + 3.193) / 1000
# Nucleosome alignment: (0.34 * (985 + 1016) + 8.99) / 1000

# Run this script from the directory containing the datasets to consolidate.
cons_name = (
    "BiotBact1_1016-601-Bact2_269NBsaILig_"
    "STdimerTArich_90mMKCl_2500uMMgCl2_15ng_ul_TYRNA_1umBeads"
)
folders = sorted(glob.glob("2*"))
add_consolidate = True
remove_folders = True

if remove_folders:
    # Remove selected folders here, or retain folders = [] to consolidate only
    # previously consolidated CSVs.
    # folders.remove("260115_example_dataset")
    folders = []

# Retained only to make the change explicit: the original copied Lc-trace CSVs
# but never read them. The optimized script performs no unused Lc-trace copy.
copy_lctraces = False

date_str = datetime.now().strftime("%y%m%d")
var_component = 0
expected_transition_size = 269 * 0.34 / 1000
rupture_force_threshold = 1

WLCpini = ([50, 0.68997, 1200], [0.65, 0.01296])
WLCpend = (
    [50, 0.68997 + expected_transition_size, 1200],
    [0.65, 0.01296],
)

realign_begining = True
realign_end_all = False
if not realign_begining:
    realign_end_all = True

if realign_begining:
    ranges_for_dist_aln = [[0.55, 0.7], [2, 4]]
    ranges_for_force_aln = [[0.5, 0.6], [-100, 1]]
else:
    ranges_for_dist_aln = None
    ranges_for_force_aln = None

curate_annotation = True
lower_size_bound = expected_transition_size - 5 / 1000
upper_size_bound = expected_transition_size + (49.3 + 5) / 1000
refWLC = WLCpini if realign_begining else WLCpend

rupture_force_detection = "peaks"
if rupture_force_detection == "peaks":
    trans_find_pars = {
        "rupture_force_detection": "peaks",
        "force_cutoff": 1,
        "threshold": 0.5,
        "distance": 5,
        "window": 5,
    }
elif rupture_force_detection == "hmm":
    trans_find_pars = {
        "rupture_force_detection": "hmm",
        "force_cutoff": 2,
        "max_states": 10,
    }
else:
    raise ValueError("rupture_force_detection must be 'peaks' or 'hmm'.")

trans_params = {
    "nfit": 22,
    "navg": 2,
    "FAvgRange": 0.05,
    "FFitRange": 1,
    "padding": 3,
    "Lp": 50,
    "S": 1200,
}

# Memory/checkpoint settings.
downfact = 25
CSV_CHUNKSIZE = 500_000
COMPRESS_CHECKPOINTS = False
REBUILD_CONSOLIDATION_CACHE = True
SHOW_AUTOMATED_FILTER_PLOTS = True

# Plot settings are supplied explicitly when each function is called.
FINAL_PLOT_BOUNDS = [[0.54, 0.9], [-2, 30]]
ALIGNMENT_PLOT_BOUNDS = [[0.4, 1.2], [-1, 30]]
ANNOTATION_PLOT_BOUNDS = FINAL_PLOT_BOUNDS
AUTOFILTER_PLOT_BOUNDS = FINAL_PLOT_BOUNDS
FILTER_CURATION_PLOT_BOUNDS = FINAL_PLOT_BOUNDS
PRE_RUPTURE_OFFSETS = (200, 12, 3)


# %% DEFINE PATHS

CONSOLIDATION_DIR = Path("data_consolidation")
CONSOLIDATE_PICS_DIR = CONSOLIDATION_DIR / "consolidate_pics"
ALIGNMENT_PICS_DIR = CONSOLIDATION_DIR / "alignment_pics"

OUTPUT_PATHS = {
    "calpars": CONSOLIDATION_DIR /
    f"{date_str}_{cons_name}_calpars_consolidate.csv",
    "video": CONSOLIDATION_DIR /
    f"{date_str}_{cons_name}_video_microscopy_consolidate.csv",
    "post": CONSOLIDATION_DIR /
    f"{date_str}_{cons_name}_postprocessed_consolidate.csv",
    "annotation": CONSOLIDATION_DIR /
    f"{date_str}_{cons_name}_annotated_states_consolidate.csv",
    "discarded": CONSOLIDATION_DIR /
    f"{date_str}_{cons_name}_discarded_consolidate.csv",
}
METADATA_PATH = (
    CONSOLIDATION_DIR / f"{date_str}_{cons_name}_consolidate.txt"
)
FILTER_PLOT_PATH = (
    CONSOLIDATION_DIR / f"{date_str}_size_filter_dev_cat.png"
)

CHECKPOINT_DIR = CONSOLIDATION_DIR / "checkpoints" / cons_name
TRACE_CACHE_DIR = CHECKPOINT_DIR / "traces"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"
STATE_PATH = CHECKPOINT_DIR / "state.json"

CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
CONSOLIDATE_PICS_DIR.mkdir(parents=True, exist_ok=True)
ALIGNMENT_PICS_DIR.mkdir(parents=True, exist_ok=True)
TRACE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# %% RESOLVE INPUT FILES WITHOUT COPYING THEM

input_paths = resolve_input_paths(
    folders,
    CONSOLIDATION_DIR,
    add_consolidate=add_consolidate,
    excluded_outputs=list(OUTPUT_PATHS.values()),
)

# %% BUILD OR LOAD PER-FEC CACHE

manifest, annotation_records = _build_or_load_trace_cache(
    input_paths["post"],
    input_paths["annotation"],
    downfact=downfact,
    chunksize=CSV_CHUNKSIZE,
    compress=COMPRESS_CHECKPOINTS,
    rebuild=REBUILD_CONSOLIDATION_CACHE,
)
consolidation_state = _load_or_create_state(
    manifest,
    annotation_records,
    rebuild=REBUILD_CONSOLIDATION_CACHE,
)
trace_info_by_id = {item["fec_id"]: item for item in manifest["traces"]}
fecs = [item["fec_id"] for item in manifest["traces"]]
del annotation_records
gc.collect()


# %% REALIGN FECS

run_alignment(
    realign_begining=realign_begining,
    realign_end_all=realign_end_all,
    WLCpini=WLCpini,
    WLCpend=WLCpend,
    var_lc_component=var_component,
    ranges_for_dist_aln=ranges_for_dist_aln,
    ranges_for_force_aln=ranges_for_force_aln,
    plot_bounds=ALIGNMENT_PLOT_BOUNDS,
    pre_rupture_offsets=PRE_RUPTURE_OFFSETS,
    force=False,
)


# %% MANUAL CURATION OF ANNOTATIONS

# Incomplete work resumes at the first unfinished FEC. After completion,
# rerunning this cell reviews all FECs from the beginning while preserving the
# saved annotations.
if curate_annotation:
    curate_annotations(
        trans_find_pars,
        ANNOTATION_PLOT_BOUNDS,
        refWLC,
        var_lc_component=var_component,
        start_index=None,
        force=False,
    )


# %% CALCULATE TRANSITION EXTENSIONS

calculate_transition_extensions(trans_params, force=False)


# %% PLOT SIZE AND INITIAL FORCE BEFORE FILTERING

alltdLcs, all_ini_sliding_force = get_filter_summary_arrays()
joint_scatter_with_marginals(
    alltdLcs - expected_transition_size * 1000,
    all_ini_sliding_force,
    bins_top=24,
    bins_right=12,
    xlabel="Contour length deviation (nm)",
    ylabel="Initial rupture force (pN)",
    show_hist_ticks=False,
    vlines=(
        1000 * (lower_size_bound - expected_transition_size),
        1000 * (upper_size_bound - expected_transition_size),
    ),
    hlines=None,
    labels=None,
    legend=False,
    xlim=(-50, 60),
)
plt.show()
plt.close("all")


# %% AUTOMATED FILTERING

# These None values preserve the original behavior: size is active, whereas
# representative initial/final-force thresholds are currently disabled.
run_automatic_filtering(
    refWLC,
    expected_transition_size=expected_transition_size,
    realign_begining=realign_begining,
    ranges_for_dist_aln=ranges_for_dist_aln,
    lower_size_bound=lower_size_bound,
    upper_size_bound=upper_size_bound,
    upper_ini_force_threshold=None,
    lower_final_force_threshold=None,
    rmsd_threshold=100,
    show_plots=SHOW_AUTOMATED_FILTER_PLOTS,
    plot_bounds=AUTOFILTER_PLOT_BOUNDS,
    force=False,
)


# %% CURATE AUTOMATED FILTERING

# Pass keeps the current automatic/manual code, keep sets 0, discard sets 5,
# and return goes to the previous FEC. Every response is saved immediately.
curate_filtering(
    FILTER_CURATION_PLOT_BOUNDS,
    refWLC,
    expected_transition_size=expected_transition_size,
    var_lc_component=var_component,
    start_index=None,
    force=False,
)


# %% PLOT AND SAVE FINAL FILTER CATEGORIES

labels = np.asarray(
    [
        0
        if consolidation_state["traces"][fec_id]["error_code"] == 0
        else 1
        for fec_id in fecs
    ]
)
joint_scatter_with_marginals(
    alltdLcs - expected_transition_size * 1000,
    all_ini_sliding_force,
    bins_top=24,
    bins_right=12,
    xlabel="Contour length deviation (nm)",
    ylabel="Initial rupture force (pN)",
    show_hist_ticks=False,
    vlines=[
        1000 * (lower_size_bound - expected_transition_size),
        1000 * (upper_size_bound - expected_transition_size),
    ],
    hlines=None,
    labels=labels,
    legend=True,
    xlim=(-60, 60),
)
plt.savefig(FILTER_PLOT_PATH, dpi=300)
plt.show()
plt.close("all")
del labels, alltdLcs, all_ini_sliding_force
gc.collect()


# %% DETERMINE FINAL FECS AND BASELINES

final_fecs, final_baselines = get_final_fecs_and_baselines()
print(f"Keeping {len(final_fecs)} of {len(fecs)} FECs.")


# %% SAVE FINAL FEC PLOTS

save_final_plots(
    final_fecs,
    refWLC,
    FINAL_PLOT_BOUNDS,
    var_lc_component=var_component,
)


# %% WRITE CONSOLIDATED CSV FILES

write_consolidated_csvs(
    input_paths,
    final_fecs,
    final_baselines,
    OUTPUT_PATHS,
    chunksize=CSV_CHUNKSIZE,
)


# %% SAVE METADATA

save_metadata(
    METADATA_PATH,
    cons_name=cons_name,
    expected_transition_size=expected_transition_size,
    refWLC=refWLC,
    realign_begining=realign_begining,
    ranges_for_dist_aln=ranges_for_dist_aln,
    ranges_for_force_aln=ranges_for_force_aln,
    curate_annotation=curate_annotation,
    rupture_force_threshold=rupture_force_threshold,
    lower_size_bound=lower_size_bound,
    upper_size_bound=upper_size_bound,
    folders=folders,
    input_paths=input_paths,
)


# %% FINISHED

plt.close("all")
gc.collect()
print(
    "Consolidation finished successfully. Per-FEC checkpoints were retained; "
    "no dill session dump is required."
)
