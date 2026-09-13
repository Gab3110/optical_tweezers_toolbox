'''
Description  : This script works with the main dataframe generated as an
               output of the script batch_fec_preprocess. It can align the trap position
               and diffF of each FEC to an arbitrary eWLC model, storing
               this aligned channels into separate columns of the main dataframe.
               The script also automatically detect the tether
               rupture index using an adapted version of one of Christian Kaiser
               scripts, so the user can use it to trim the data by slicing for
               plotting purposes. After performing these processes, the script
               loops over each fec, so the user can manually keep or discard the
               trace as well as to correct the aligning and rupture index.
               Results are stored in new output dataframes, with restartable
               per-FEC checkpoints instead of a whole-session pickle.


Author       : Gabriel Jiménez-Avalos, PhD. student.
Affiliation  : T. C. Jenkins department of Biophysics. Johns Hopkins University.
Email        : gjimene5@jhu.edu
Version      : 5.0
Date         : 2025/04/06


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
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import CK_filtfilt
from tweezers_toolbox_modules import fec_plotting as fplt
from tweezers_toolbox_modules import postprocessing as post

# %% GENERAL UTILITIES

def _source_signature(path):
    """Return information used to detect replacement/modification of a CSV."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_signature(value):
    """Return a deterministic SHA-256 signature for JSON-compatible values."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path, value):
    """Write JSON in the destination directory, then atomically replace it."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _atomic_savez(path, **arrays):
    """Atomically save NumPy arrays without holding the whole session in RAM."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    save_function = np.savez_compressed if COMPRESS_CHECKPOINTS else np.savez

    # Passing an open file prevents NumPy from automatically adding '.npz' to
    # the temporary filename.
    with open(temporary_path, "wb") as handle:
        save_function(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_complete_fecs_csv(
    path,
    usecols=None,
    *,
    chunksize,
):
    """
    Yield one complete, contiguous fec_id group at a time from a CSV.

    The last group in each CSV chunk is retained because it may continue in the
    next chunk. A repeated noncontiguous fec_id is rejected rather than silently
    producing an incorrect output.
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

        # Every run except the final run is known to be complete.
        for start, end in zip(starts[:-1], ends[:-1]):
            fec_id = str(fec_values[start])
            if fec_id in emitted_fec_ids:
                raise ValueError(
                    f"fec_id {fec_id!r} appears in multiple noncontiguous "
                    f"blocks in {path}. Sort/group the CSV by fec_id first."
                )
            emitted_fec_ids.add(fec_id)
            yield fec_id, data.iloc[start:end].copy()

        pending = data.iloc[starts[-1]:ends[-1]].copy()
        del data, chunk, fec_values, boundaries, starts, ends

    if pending is not None and not pending.empty:
        fec_id = str(pending["fec_id"].iloc[0])
        if fec_id in emitted_fec_ids:
            raise ValueError(
                f"fec_id {fec_id!r} appears in multiple noncontiguous blocks "
                f"in {path}. Sort/group the CSV by fec_id first."
            )
        yield fec_id, pending


def _subsample_for_overview(
    x,
    y,
    *,
    max_points,
):
    """Thin only displayed overview lines; never alter analysis arrays."""
    if max_points is None or max_points <= 0 or len(x) <= max_points:
        return x, y
    stride = int(np.ceil(len(x) / max_points))
    return x[::stride], y[::stride]


def _build_or_load_trace_cache(input_path, rebuild=False):
    """Build a small per-FEC raw-array cache using chunked CSV input."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    source = _source_signature(input_path)

    if MANIFEST_PATH.exists() and not rebuild:
        existing_manifest = _read_json(MANIFEST_PATH)
        if existing_manifest.get("source") != source:
            raise RuntimeError(
                "The preprocessed CSV changed after this checkpoint directory "
                "was created. If this replacement is intentional, set "
                "REBUILD_TRACE_CACHE = True and rerun the cache cell."
            )

        all_present = all(
            (CHECKPOINT_DIR / item["raw_file"]).exists()
            for item in existing_manifest.get("traces", [])
        )
        if all_present:
            print(
                f"Using existing trace cache with "
                f"{len(existing_manifest['traces'])} FECs."
            )
            return existing_manifest

        print("The trace cache is incomplete; rebuilding missing cache data.")

    analysis_columns = ["fec_id", "molext", "diffF", "curve_type"]
    trace_manifest = []

    print("Building per-FEC trace cache...")
    for trace_number, (fec_id, trace_df) in enumerate(
        _iter_complete_fecs_csv(
            input_path,
            usecols=analysis_columns,
            chunksize=CSV_CHUNKSIZE,
        )
    ):
        curve_types = trace_df["curve_type"].drop_duplicates()
        if len(curve_types) != 1:
            raise ValueError(
                f"FEC {fec_id!r} contains multiple curve_type values: "
                f"{curve_types.tolist()}"
            )

        d = trace_df["molext"].to_numpy(copy=True)
        f = trace_df["diffF"].to_numpy(copy=True)
        raw_relative_path = Path("raw") / f"{trace_number:06d}.npz"

        _atomic_savez(CHECKPOINT_DIR / raw_relative_path, d=d, f=f)

        trace_manifest.append(
            {
                "fec_id": fec_id,
                "curve_type": str(curve_types.iloc[0]),
                "n_rows": int(len(trace_df)),
                "raw_file": str(raw_relative_path),
                "fine_file": str(
                    Path("fine_aligned") / f"{trace_number:06d}.npz"
                ),
                "curated_file": str(
                    Path("curated") / f"{trace_number:06d}.npz"
                ),
            }
        )

        print(f"Cached {trace_number + 1}: {fec_id}")
        del trace_df, d, f, curve_types

    manifest = {
        "version": 1,
        "source": source,
        "created": datetime.now().isoformat(),
        "traces": trace_manifest,
    }
    _atomic_write_json(MANIFEST_PATH, manifest)
    print(f"Trace cache complete: {len(trace_manifest)} FECs.")
    return manifest


def _load_or_create_state(manifest, rebuild=False):
    """Load small restart state or create a new state document."""
    if STATE_PATH.exists() and not rebuild:
        loaded_state = _read_json(STATE_PATH)
        if loaded_state.get("source") != manifest["source"]:
            raise RuntimeError(
                "The saved analysis state belongs to a different input CSV. "
                "Set REBUILD_TRACE_CACHE = True only if replacing the source "
                "and intentionally starting a new analysis state."
            )
    else:
        loaded_state = {
            "version": 1,
            "source": manifest["source"],
            "stage_signatures": {},
            "traces": {},
        }

    for item in manifest["traces"]:
        loaded_state["traces"].setdefault(item["fec_id"], {})

    _atomic_write_json(STATE_PATH, loaded_state)
    return loaded_state


def _prepare_stage(stage_name, signature, keys_to_clear, force=False):
    """Invalidate only this stage and its dependent state when required."""
    previous_signature = analysis_state["stage_signatures"].get(stage_name)
    if force or previous_signature != signature:
        for record in analysis_state["traces"].values():
            for key in keys_to_clear:
                record.pop(key, None)
        analysis_state["stage_signatures"][stage_name] = signature
        _atomic_write_json(STATE_PATH, analysis_state)


def _load_raw_arrays(trace_info):
    path = CHECKPOINT_DIR / trace_info["raw_file"]
    with np.load(path, allow_pickle=False) as saved:
        return saved["d"].copy(), saved["f"].copy()


def _load_raw_force(trace_info):
    """Load only force when distance is not needed."""
    path = CHECKPOINT_DIR / trace_info["raw_file"]
    with np.load(path, allow_pickle=False) as saved:
        return saved["f"].copy()


def _load_curated_arrays(trace_info):
    path = CHECKPOINT_DIR / trace_info["curated_file"]
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["d_aln"].copy(),
            saved["f_aln"].copy(),
            saved["dlf_aln"].copy(),
            saved["flf_aln"].copy(),
        )


def _load_curated_full_arrays(trace_info):
    """Load only the full-bandwidth curated arrays."""
    path = CHECKPOINT_DIR / trace_info["curated_file"]
    with np.load(path, allow_pickle=False) as saved:
        return saved["d_aln"].copy(), saved["f_aln"].copy()


def _load_fine_arrays(trace_info):
    """Load the exact low-bandwidth arrays produced by fine alignment."""
    path = CHECKPOINT_DIR / trace_info["fine_file"]
    with np.load(path, allow_pickle=False) as saved:
        return saved["dlf_aln"].copy(), saved["flf_aln"].copy()


def _current_offsets(fec_id):
    record = analysis_state["traces"][fec_id]
    if "doff" not in record or "foff" not in record:
        raise RuntimeError(
            f"FEC {fec_id!r} has not completed fine alignment. Run the "
            "alignment cell before trimming or curation."
        )
    return float(record["doff"]), float(record["foff"])

def run_automatic_trimming(force=False):
    """Detect and checkpoint a rupture index while holding one FEC in RAM."""
    fine_signature = analysis_state["stage_signatures"].get("fine_alignment")
    if fine_signature is None:
        raise RuntimeError("Run the alignment initialization before trimming.")

    signature = _stable_signature(
        {
            "fine_signature": fine_signature,
            "autotrim": autotrim,
            "maxforce": maxforce,
            "downfact": downfact,
        }
    )
    _prepare_stage(
        "automatic_trimming",
        signature,
        keys_to_clear=[
            "auto_rupture_index",
            "rupture_index",
            "discard",
            "curated",
            "curation_signature",
        ],
        force=force,
    )

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        curve_type = trace_info["curve_type"]
        record = analysis_state["traces"][fec_id]

        if "auto_rupture_index" in record:
            continue

        f_aln = _load_raw_force(trace_info)
        _, foff = _current_offsets(fec_id)
        f_aln += foff

        if autotrim:
            if curve_type == "S":
                rupture_index = post.find_tether_rupture(
                    f_aln,
                    25,
                    1,
                    10,
                    "stretch",
                    maxforce=maxforce,
                )
            elif curve_type == "R":
                rupture_index = post.find_tether_rupture(
                    f_aln,
                    25,
                    1,
                    10,
                    "relax",
                    maxforce=maxforce,
                )
            else:
                rupture_index = len(f_aln)
        else:
            if curve_type == "S":
                rupture_index = len(f_aln)
            elif curve_type == "R":
                rupture_index = 0
            else:
                rupture_index = len(f_aln)

        record["auto_rupture_index"] = int(rupture_index)
        _atomic_write_json(STATE_PATH, analysis_state)
        print(f"Trimming complete: {fec_id}")

        del f_aln
        gc.collect()

def run_rough_alignment(force=False):
    """Calculate and checkpoint one rough distance offset per FEC."""
    if not align:
        return

    signature = _stable_signature(
        {
            "source": manifest["source"],
            "align": align,
            "WLCp": WLCp,
            "downfact": downfact,
            "rough_bounds": rough_bounds,
        }
    )
    _prepare_stage(
        "rough_alignment",
        signature,
        keys_to_clear=[
            "rough_doff",
            "doff",
            "foff",
            "auto_rupture_index",
            "rupture_index",
            "discard",
            "curated",
            "curation_signature",
        ],
        force=force,
    )

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 5))

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = analysis_state["traces"][fec_id]
        d, f = _load_raw_arrays(trace_info)
        dlf = CK_filtfilt(d, downfact)
        flf = CK_filtfilt(f, downfact)

        if "rough_doff" not in record:
            rough_aln_fec = post.AlignFEC(
                dlf,
                flf,
                rough_bounds,
                [],
                *WLCp,
                custom_function=False,
            )
            rough_aln_fec.dist_aln()

            # Do not allocate a full (x_aln - dlf) temporary array.
            rough_doff = float(rough_aln_fec.x_aln[0] - dlf[0])
            record["rough_doff"] = rough_doff
            _atomic_write_json(STATE_PATH, analysis_state)
            x_aligned = rough_aln_fec.x_aln
        else:
            rough_doff = float(record["rough_doff"])
            x_aligned = dlf + rough_doff
            rough_aln_fec = None

        x_plot, f_plot = _subsample_for_overview(
            x_aligned,
            flf,
            max_points=MAX_OVERVIEW_POINTS_PER_FEC,
        )
        ax.plot(x_plot, f_plot)
        print(f"Rough alignment complete: {fec_id}")

        del d, f, dlf, flf, x_aligned, x_plot, f_plot, rough_aln_fec

    ax.set_xlabel("Extension (µm)", fontsize=20)
    ax.set_ylabel("Force (pN)", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    fig.tight_layout()
    gc.collect()
    return fig, ax

def run_fine_alignment(force=False):
    """Calculate final scalar offsets and checkpoint after every FEC."""
    if not align:
        return

    rough_signature = analysis_state["stage_signatures"].get(
        "rough_alignment"
    )
    if rough_signature is None:
        raise RuntimeError("Run rough alignment before fine alignment.")

    signature = _stable_signature(
        {
            "rough_signature": rough_signature,
            "dist_aln_bounds": dist_aln_bounds,
            "WLCp": WLCp,
            "var_component": var_component,
            "downfact": downfact,
        }
    )
    _prepare_stage(
        "fine_alignment",
        signature,
        keys_to_clear=[
            "doff",
            "foff",
            "auto_rupture_index",
            "rupture_index",
            "discard",
            "curated",
            "curation_signature",
        ],
        force=force,
    )

    # Release overview/rough-line arrays before opening many individual plots.
    plt.close("all")

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = analysis_state["traces"][fec_id]
        if "rough_doff" not in record:
            raise RuntimeError(f"FEC {fec_id!r} has no rough alignment offset.")

        d, f = _load_raw_arrays(trace_info)
        dlf = CK_filtfilt(d, downfact)
        flf = CK_filtfilt(f, downfact)

        fine_path = CHECKPOINT_DIR / trace_info["fine_file"]

        if (
            "doff" not in record
            or "foff" not in record
            or not fine_path.exists()
        ):
            rough_doff = float(record["rough_doff"])
            iter_aln_fec = post.AlignFEC(
                dlf + rough_doff,
                flf,
                dist_aln_bounds,
                [],
                *WLCp,
                custom_function=False,
            )
            iter_aln_fec.iterative_dist_aln()

            # Read one pair of elements instead of creating full subtraction
            # arrays merely to extract their first value.
            doff = float(iter_aln_fec.x_aln[0] - dlf[0])
            foff = float(iter_aln_fec.F_aln[0] - flf[0])
            dlf_aln = np.asarray(iter_aln_fec.x_aln).copy()
            flf_aln = np.asarray(iter_aln_fec.F_aln).copy()

            # Save the exact low-bandwidth results returned by AlignFEC. This
            # avoids assuming its transformation is only a scalar translation.
            _atomic_savez(
                fine_path,
                dlf_aln=dlf_aln,
                flf_aln=flf_aln,
            )
            record["doff"] = doff
            record["foff"] = foff
            _atomic_write_json(STATE_PATH, analysis_state)
        else:
            doff = float(record["doff"])
            foff = float(record["foff"])
            dlf_aln, flf_aln = _load_fine_arrays(trace_info)
            iter_aln_fec = None

        fplt.plot_individual_fec(
            dlf_aln,
            flf_aln,
            fec_id,
            "black",
            [0.5, 1.2],
            [-1, 40],
            str(PREPROCESS_PICS_DIR / f"{fec_id}.png"),
            *WLCp,
            lcstates=[WLCp[var_component][1]],
            var_component=var_component,
        )
        plt.close("all")
        print(f"Fine alignment complete: {fec_id}")

        del d, f, dlf, flf, dlf_aln, flf_aln, iter_aln_fec
        gc.collect()


def initialize_without_alignment(force=False):
    """Set zero offsets while retaining the original align=False behavior."""
    signature = _stable_signature(
        {"source": manifest["source"], "align": False}
    )
    _prepare_stage(
        "fine_alignment",
        signature,
        keys_to_clear=[
            "doff",
            "foff",
            "auto_rupture_index",
            "rupture_index",
            "discard",
            "curated",
            "curation_signature",
        ],
        force=force,
    )

    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = analysis_state["traces"][trace_info["fec_id"]]
        d, f = _load_raw_arrays(trace_info)
        dlf_aln = CK_filtfilt(d, downfact)
        flf_aln = CK_filtfilt(f, downfact)
        _atomic_savez(
            CHECKPOINT_DIR / trace_info["fine_file"],
            dlf_aln=dlf_aln,
            flf_aln=flf_aln,
        )
        record["rough_doff"] = 0.0
        record["doff"] = 0.0
        record["foff"] = 0.0
        _atomic_write_json(STATE_PATH, analysis_state)
        print(f"No-alignment initialization complete: {fec_id}")
        del d, f, dlf_aln, flf_aln
        gc.collect()

def curate_traces(start_index=None, force=False):
    """
    Run interactive curation and checkpoint after every action.

    Parameters
    ----------
    start_index : None, int, or str
        None resumes at the first uncurated FEC. If every FEC is already
        curated, None starts a complete review from the first FEC. An integer
        starts at that position, and a string starts at that fec_id. Previously
        curated FECs can also be revisited with the return command.
    force : bool
        If True, invalidate all previous manual curation and start again.
    """
    trim_signature = analysis_state["stage_signatures"].get(
        "automatic_trimming"
    )
    if trim_signature is None:
        raise RuntimeError("Run automatic trimming before manual curation.")

    signature = _stable_signature(
        {
            "trim_signature": trim_signature,
            "curation_plot_limits": curation_plot_limits,
            "maxforce": maxforce,
            "WLCp": WLCp,
            "var_component": var_component,
            "downfact": downfact,
        }
    )
    _prepare_stage(
        "manual_curation",
        signature,
        keys_to_clear=[
            "rupture_index",
            "discard",
            "curated",
            "curation_signature",
        ],
        force=force,
    )

    if start_index is None:
        uncurated = [
            index
            for index, fec_id in enumerate(fecs)
            if not analysis_state["traces"][fec_id].get("curated", False)
        ]
        if not uncurated:
            print(
                "All FECs are already curated. Starting a complete review "
                "from the first FEC."
            )
            current_index = 0
        else:
            current_index = uncurated[0]
    elif isinstance(start_index, str):
        if start_index not in trace_info_by_id:
            raise KeyError(f"Unknown fec_id: {start_index}")
        current_index = fecs.index(start_index)
    else:
        current_index = int(start_index)

    if current_index < 0 or current_index >= len(fecs):
        raise IndexError(
            f"start_index must be between 0 and {len(fecs) - 1}."
        )

    while current_index < len(fecs):
        fec_id = fecs[current_index]
        trace_info = trace_info_by_id[fec_id]
        curve_type = trace_info["curve_type"]
        record = analysis_state["traces"][fec_id]

        curated_path = CHECKPOINT_DIR / trace_info["curated_file"]
        can_restore = (
            record.get("curated", False)
            and record.get("curation_signature") == signature
            and curated_path.exists()
        )

        if can_restore:
            d_aln, f_aln, dlf_aln, flf_aln = _load_curated_arrays(trace_info)
            rupture_index = int(record["rupture_index"])
            discard = int(record["discard"])
        else:
            d_aln, f_aln = _load_raw_arrays(trace_info)
            doff, foff = _current_offsets(fec_id)
            # These arrays are private copies loaded from the cache, so shifting
            # them in place avoids two additional full-trace allocations.
            d_aln += doff
            f_aln += foff
            dlf_aln, flf_aln = _load_fine_arrays(trace_info)
            rupture_index = int(record["auto_rupture_index"])
            discard = int(record.get("discard", 0))

        curation = post.Curate_FECPostprocessing(
            fec_id,
            d_aln,
            f_aln,
            dlf_aln,
            flf_aln,
            rupture_index,
            discard,
            curation_plot_limits,
            maxforce,
            *WLCp,
            curve_type=curve_type,
            var_component=var_component,
        )
        continue_decision = curation.run()

        # Copy out of the GUI object before closing figures or deleting it.
        curated_d = np.asarray(curation.d_aln).copy()
        curated_f = np.asarray(curation.f_aln).copy()
        curated_dlf = np.asarray(curation.dlf_aln).copy()
        curated_flf = np.asarray(curation.flf_aln).copy()
        curated_rupture_index = int(curation.rupture_index)
        curated_discard = int(curation.discard)

        expected_rows = int(trace_info["n_rows"])
        if len(curated_d) != expected_rows or len(curated_f) != expected_rows:
            raise ValueError(
                f"Curation changed the full-bandwidth length of {fec_id!r}. "
                f"Expected {expected_rows}, received d={len(curated_d)} and "
                f"f={len(curated_f)}. The final CSV requires one aligned value "
                "for every input row."
            )

        # Save arrays first. If a crash occurs before the JSON update, the FEC
        # is safely treated as uncurated and shown again on resume.
        _atomic_savez(
            curated_path,
            d_aln=curated_d,
            f_aln=curated_f,
            dlf_aln=curated_dlf,
            flf_aln=curated_flf,
        )
        record["rupture_index"] = curated_rupture_index
        record["discard"] = curated_discard
        record["curated"] = True
        record["curation_signature"] = signature
        record["curated_at"] = datetime.now().isoformat()
        _atomic_write_json(STATE_PATH, analysis_state)

        print(
            f"Checkpoint saved: {fec_id} "
            f"({current_index + 1}/{len(fecs)})"
        )

        del (
            d_aln,
            f_aln,
            dlf_aln,
            flf_aln,
            curated_d,
            curated_f,
            curated_dlf,
            curated_flf,
            curation,
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
                f"Unrecognized curation decision {continue_decision!r}; "
                "remaining on the current FEC."
            )

def save_postprocessed_plots():
    """Save plots one at a time without retaining Matplotlib figures."""
    for trace_info in manifest["traces"]:
        fec_id = trace_info["fec_id"]
        record = analysis_state["traces"][fec_id]

        if not record.get("curated", False):
            raise RuntimeError(
                f"FEC {fec_id!r} has not been curated. Finish curation before "
                "saving final plots."
            )
        if int(record["discard"]) != 0:
            continue

        d_aln, f_aln = _load_curated_full_arrays(trace_info)
        rupture_index = int(record["rupture_index"])
        curve_type = trace_info["curve_type"]

        if curve_type == "S":
            d_plot_hf = d_aln
            f_plot_hf = f_aln
        else:
            # np.flip returns a view here, so it does not duplicate the arrays.
            d_plot_hf = np.flip(d_aln)
            f_plot_hf = np.flip(f_aln)

        d_plot_lf = CK_filtfilt(d_plot_hf, 25)
        f_plot_lf = CK_filtfilt(f_plot_hf, 25)

        fplt.plot_individual_fec(
            d_plot_lf,
            f_plot_lf,
            fec_id,
            "black",
            [0.45, 1.2],
            [-1, 40],
            str(POSTPROCESS_PICS_DIR / f"{fec_id}.png"),
            *WLCp,
            var_component=0,
            distances_hf=d_plot_hf,
            forces_hf=f_plot_hf,
            scatter=False,
            annot=False,
            savefig=True,
            rupture_index=rupture_index,
            lcstates=[WLCp[0][1]],
            states_end_indexes=False,
            lc_trajectory=False,
        )
        plt.close("all")
        print(f"Saved final plot: {fec_id}")

        del (
            d_aln,
            f_aln,
            d_plot_hf,
            f_plot_hf,
            d_plot_lf,
            f_plot_lf,
        )
        gc.collect()

def write_final_csvs():
    """Write both output CSVs one complete FEC at a time."""
    kept_temporary = KEPT_OUTPUT_PATH.with_name(KEPT_OUTPUT_PATH.name + ".tmp")
    discarded_temporary = DISCARDED_OUTPUT_PATH.with_name(
        DISCARDED_OUTPUT_PATH.name + ".tmp"
    )

    # Truncate any incomplete temporary output from an interrupted prior run and
    # establish the exact column headers even when a category has zero traces.
    pd.DataFrame(columns=SORTED_COLS).to_csv(kept_temporary, index=False)
    pd.DataFrame(columns=SORTED_COLS).to_csv(discarded_temporary, index=False)

    emitted = set()
    for fec_id, trace_df in _iter_complete_fecs_csv(
        INPUT_PATH,
        usecols=None,
        chunksize=CSV_CHUNKSIZE,
    ):
        if fec_id not in trace_info_by_id:
            raise KeyError(f"FEC {fec_id!r} is missing from the trace manifest.")

        trace_info = trace_info_by_id[fec_id]
        record = analysis_state["traces"][fec_id]
        if not record.get("curated", False):
            raise RuntimeError(
                f"FEC {fec_id!r} has not been curated; final CSVs were not "
                "replaced."
            )

        d_aln, f_aln = _load_curated_full_arrays(trace_info)
        if len(trace_df) != len(d_aln) or len(trace_df) != len(f_aln):
            raise ValueError(
                f"Length mismatch for {fec_id!r}: CSV has {len(trace_df)} "
                f"rows, d_aln has {len(d_aln)}, and f_aln has {len(f_aln)}."
            )

        trace_df["rupture_index"] = int(record["rupture_index"])
        if align:
            trace_df["molext_aln"] = d_aln
            trace_df["diffF_aln"] = f_aln
        else:
            # Preserve the original script's output behavior when align=False.
            trace_df["molext_aln"] = np.nan
            trace_df["diffF_aln"] = np.nan

        # The DataFrame now owns the output columns; release checkpoint arrays
        # before formatting/writing the rest of the trace.
        del d_aln, f_aln

        discard = int(record["discard"])
        if discard == 5:
            trace_df["error_code"] = 5

        missing_columns = [
            column for column in SORTED_COLS if column not in trace_df.columns
        ]
        if missing_columns:
            raise KeyError(
                f"The source CSV is missing required columns: {missing_columns}"
            )

        if discard == 0:
            trace_df.to_csv(
                kept_temporary,
                mode="a",
                header=False,
                index=False,
                columns=SORTED_COLS,
            )
        elif discard == 5:
            trace_df.to_csv(
                discarded_temporary,
                mode="a",
                header=False,
                index=False,
                columns=SORTED_COLS,
            )

        emitted.add(fec_id)
        print(f"Wrote final data: {fec_id}")
        del trace_df
        gc.collect()

    expected = set(fecs)
    if emitted != expected:
        missing = sorted(expected - emitted)
        unexpected = sorted(emitted - expected)
        raise RuntimeError(
            f"Final CSV verification failed. Missing={missing}, "
            f"unexpected={unexpected}."
        )

    # Replace final files only after the complete write and validation succeeds.
    os.replace(kept_temporary, KEPT_OUTPUT_PATH)
    os.replace(discarded_temporary, DISCARDED_OUTPUT_PATH)
    print(f"Saved: {KEPT_OUTPUT_PATH}")
    print(f"Saved: {DISCARDED_OUTPUT_PATH}")

def plot_all_traces():
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 5))

    for trace_info in manifest["traces"]:
        d, f = _load_raw_arrays(trace_info)
        dlf = CK_filtfilt(d, downfact)
        flf = CK_filtfilt(f, downfact)
        d_plot, f_plot = _subsample_for_overview(
            dlf,
            flf,
            max_points=MAX_OVERVIEW_POINTS_PER_FEC,
        )
        ax.plot(d_plot, f_plot)
        del d, f, dlf, flf, d_plot, f_plot

    ax.set_xlabel("Extension (µm)", fontsize=20)
    ax.set_ylabel("Force (pN)", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    fig.tight_layout()
    return fig, ax

def calculate_filtering_statistics():
    """Calculate statistics without retaining either final output CSV in RAM."""
    records = analysis_state["traces"]
    kept = sum(int(records[fec_id]["discard"]) == 0 for fec_id in fecs)
    discarded_5 = sum(
        int(records[fec_id]["discard"]) == 5 for fec_id in fecs
    )

    if PREPROCESS_DISCARDED_PATH.exists():
        try:
            prior_discarded = pd.read_csv(
                PREPROCESS_DISCARDED_PATH,
                usecols=["fec_id", "error_code"],
                dtype={"fec_id": "string"},
            )
        except pd.errors.EmptyDataError:
            prior_discarded = pd.DataFrame(columns=["fec_id", "error_code"])
    else:
        prior_discarded = pd.DataFrame(columns=["fec_id", "error_code"])

    discarded_1 = prior_discarded.loc[
        prior_discarded["error_code"] == 1, "fec_id"
    ].nunique()
    discarded_2 = prior_discarded.loc[
        prior_discarded["error_code"] == 2, "fec_id"
    ].nunique()
    discarded_3 = prior_discarded.loc[
        prior_discarded["error_code"] == 3, "fec_id"
    ].nunique()

    category_sizes = [
        int(kept),
        int(discarded_1),
        int(discarded_2),
        int(discarded_3),
        int(discarded_5),
    ]
    labels = [
        "Pass",
        "Baseline not found",
        "FEC not found",
        "Piezo calibration has bad quality",
        "Manually discarded",
    ]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 9))
    if sum(category_sizes) > 0:
        ax.pie(
            category_sizes,
            autopct=lambda percentage: (
                f"{percentage:.2f}%" if percentage > 0 else ""
            ),
            textprops={"fontsize": 20},
        )
        ax.legend(labels, fontsize=15)
    else:
        ax.text(0.5, 0.5, "No traces", ha="center", va="center", fontsize=20)
        ax.axis("off")

    fig.savefig(FILTER_PLOT_PATH, format="png", dpi=300)
    plt.close(fig)

    discarded_total = (
        int(prior_discarded["fec_id"].nunique()) + int(discarded_5)
    )
    print(f"Kept FECs: {kept}")
    print(f"Discarded FECs: {discarded_total}")

    del prior_discarded
    return {
        "kept": int(kept),
        "discarded_1": int(discarded_1),
        "discarded_2": int(discarded_2),
        "discarded_3": int(discarded_3),
        "discarded_5": int(discarded_5),
        "discarded_total": int(discarded_total),
    }


def save_metadata():
    """Save the postprocessing settings, summary, and recovery-state path."""
    now = datetime.now()
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"{folder} was postprocessed on: {now}\n\n")
        handle.write(f"Source CSV: {INPUT_PATH}\n")
        handle.write(f"Chunk size: {CSV_CHUNKSIZE} rows\n")
        handle.write(f"Downsampling factor: {downfact}\n")
        handle.write(f"Automatic trimming: {autotrim}\n")
        handle.write(f"Maximum force: {maxforce}\n")
        handle.write(f"Filtering statistics: {filtering_statistics}\n\n")

        if align:
            handle.write("Alignment was on.\n")
            if end_alignment:
                handle.write("Alignment was done to the end state.\n")
            else:
                handle.write("Alignment was done to the initial state.\n")
            handle.write(
                "FECs were aligned to an eWLC/WLC model with the following "
                f"parameters:\n{WLCp}\n"
            )
            handle.write(
                f"Rough alignment bounds: {rough_bounds}\n"
                f"Fine alignment bounds: {dist_aln_bounds}\n"
                f"Variable WLC component: {var_component}\n"
            )
        else:
            handle.write("Alignment was off.\n")

        handle.write(
            "\nRecovery state was saved after every completed trace at:\n"
            f"{STATE_PATH}\n"
        )

    print(f"Saved: {METADATA_PATH}")

# %% USER SETTINGS

# Experiment name / filename prefix.
folder = (
    "260906_BiotBact1_1016-601-Bact2_269NBsaILig_STdimerTArich_90mMKCl_2500uMMgCl2_15ng_ul_TYRNA_1umBeads"
)

# Reference eWLC/WLC model. Each component is [Lp, Lc, S] or [Lp, Lc].
WLCp = ([50, 0.7854, 1200], [0.65, 0.01296])
var_component = 0

# Alignment/filtering settings.
downfact = 25
align = True
end_alignment = True
rough_bounds = [[1, 3], [3, 20]]
dist_aln_bounds = [[1.01, 1.4], [5, 20]]

# Automatic trimming settings.
autotrim = True
maxforce = 40

# Curation plot limits.
curation_plot_limits = [[0.5, 1.2], [-1, 40]]

# Memory/disk settings.
# A smaller value reduces peak RAM but reads the CSV in more chunks.
CSV_CHUNKSIZE = 500_000

# Only overview figures are visually thinned. Calculations always use every
# downsampled point, and individual plots retain the original behavior.
MAX_OVERVIEW_POINTS_PER_FEC = 10_000

# False is recommended. np.savez_compressed uses less disk but more CPU and can
# require larger temporary buffers while checkpoints are being written.
COMPRESS_CHECKPOINTS = False

# Set this to True only if the source CSV changed and you intentionally want to
# rebuild the on-disk trace cache. Existing files are overwritten, not deleted.
REBUILD_TRACE_CACHE = False


# %% PATHS AND OUTPUT COLUMNS

PREPROCESS_DIR = Path("preprocess")
POSTPROCESS_DIR = Path("postprocess")
PREPROCESS_PICS_DIR = PREPROCESS_DIR / "preprocessed_pics"
POSTPROCESS_PICS_DIR = POSTPROCESS_DIR / "postprocess_pics"

INPUT_PATH = PREPROCESS_DIR / f"{folder}_preprocessed.csv"
PREPROCESS_DISCARDED_PATH = (
    PREPROCESS_DIR / f"{folder}_preprocessed_discarded.csv"
)
KEPT_OUTPUT_PATH = POSTPROCESS_DIR / f"{folder}_postprocessed.csv"
DISCARDED_OUTPUT_PATH = (
    POSTPROCESS_DIR / f"{folder}_postprocessed_discarded.csv"
)
METADATA_PATH = POSTPROCESS_DIR / f"{folder}_postprocessing_metadata.txt"
FILTER_PLOT_PATH = Path(f"{folder}_filt_piechart.png")

CHECKPOINT_DIR = POSTPROCESS_DIR / "checkpoints" / folder
RAW_TRACE_DIR = CHECKPOINT_DIR / "raw"
FINE_TRACE_DIR = CHECKPOINT_DIR / "fine_aligned"
CURATED_TRACE_DIR = CHECKPOINT_DIR / "curated"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"
STATE_PATH = CHECKPOINT_DIR / "state.json"

SORTED_COLS = [
    "relative_time",
    "trappos1x",
    "cal_trappos1x",
    "molext",
    "molext_aln",
    "cal_molext",
    "bead1_disp",
    "bead2_disp",
    "diff_bead_disp",
    "force1x",
    "force2x",
    "diffF",
    "diffF_aln",
    "rupture_index",
    "bead_set",
    "mol_id",
    "curve_type",
    "curve_number",
    "date",
    "error_code",
    "fec_id",
]

POSTPROCESS_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESS_PICS_DIR.mkdir(parents=True, exist_ok=True)
POSTPROCESS_PICS_DIR.mkdir(parents=True, exist_ok=True)
RAW_TRACE_DIR.mkdir(parents=True, exist_ok=True)
FINE_TRACE_DIR.mkdir(parents=True, exist_ok=True)
CURATED_TRACE_DIR.mkdir(parents=True, exist_ok=True)

# %% BUILD OR LOAD THE PER-FEC TRACE CACHE

manifest = _build_or_load_trace_cache(
    INPUT_PATH,
    rebuild=REBUILD_TRACE_CACHE,
)
analysis_state = _load_or_create_state(
    manifest,
    rebuild=REBUILD_TRACE_CACHE,
)
trace_info_by_id = {item["fec_id"]: item for item in manifest["traces"]}
fecs = [item["fec_id"] for item in manifest["traces"]]


# %% PLOT ALL TRACES TO DEFINE ROUGH RANGES

overview_fig, overview_ax = plot_all_traces()
# Set overview_ax.set_xlim(...) interactively if needed.

# %% ROUGH ALIGNMENT

if align:
    rough_fig, rough_ax = run_rough_alignment(force=False)
else:
    rough_fig, rough_ax = None, None

# %% FINE ALIGNMENT

if align:
    run_fine_alignment(force=False)
else:
    initialize_without_alignment(force=False)

# %% AUTOMATIC TRIMMING

run_automatic_trimming(force=False)

# %% MANUAL CURATION OF ALIGNMENT AND TRIMMING

curate_traces()

# %% SAVE INDIVIDUAL PLOTS

save_postprocessed_plots()

# %% STREAM FINAL KEPT AND DISCARDED CSV FILES

write_final_csvs()

# %% FILTERING STATISTICS

filtering_statistics = calculate_filtering_statistics()

# %% SAVE METADATA

save_metadata()

# %% FINISHED

plt.close("all")
gc.collect()
print("Postprocessing finished successfully.")
