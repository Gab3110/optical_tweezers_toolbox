import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
from scipy.signal import find_peaks
import copy
import ruptures as rpt
import scipy.signal as signal
from matplotlib.widgets import RectangleSelector
from sfHMM.step import CB_GaussStep

#%%
sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.general_utils import mad, running_mad, running_std
from tweezers_toolbox_modules.models import ini_eWLC
from tweezers_toolbox_modules import input_handlers as hand
from tweezers_toolbox_modules import fitting
from tweezers_toolbox_modules import plotting as fplt
from tweezers_toolbox_modules.preprocessing import bessel_filtfilt, butter_filtfilt, CK_filtfilt, savgol_filt
from tweezers_toolbox_modules.general_utils import solve_equation
from tweezers_toolbox_modules.models import peeling_rigid_body_contribution

#%% General functions

def calculate_transition_work(
    extension,
    force,
    eWLC_pars_final_state,
    n_grid=10000
):

    extension = np.asarray(extension)
    force = np.asarray(force)

    # Area under the experimental FEC
    area_fec = np.trapezoid(
        force,
        extension
    )

    # Smooth extension grid over exactly the same interval
    x_final_state = np.linspace(
        extension[0],
        extension[-1],
        n_grid
    )

    # Final-state model: extension -> force
    pF_final_state = ini_eWLC(
        0,
        x_final_state,
        *eWLC_pars_final_state
    )

    # Area under the final-state model
    area_final_state = np.trapezoid(
        pF_final_state,
        x_final_state
    )

    # Corrected transition work
    transition_work = area_fec - area_final_state

    return (
        transition_work,
        area_fec,
        area_final_state,
        x_final_state,
        pF_final_state
    )

def get_ideal_putative_LLPS(
    ideal_trace,
    mad_thresh=5,
):
    # Get all dwells
    dwell_intervals, dwell_levels = get_dwell_intervals(
        ideal_trace,
        inclusive=True,
    )

    # Dwell lengths in samples
    dwell_lengths = (
        dwell_intervals[:, 1]
        - dwell_intervals[:, 0]
        + 1
    )

    # MAD criterion
    med, mad_value, _ = mad(dwell_lengths)

    threshold = med + mad_thresh * mad_value

    # Select pauses
    pause_mask = dwell_lengths > threshold

    pause_intervals = dwell_intervals[pause_mask]
    pause_levels = dwell_levels[pause_mask]
    pause_lengths = dwell_lengths[pause_mask]

    return (
        pause_intervals,
        pause_levels,
        pause_lengths,
        threshold,
    )


def downsample_lc_trace(
    d,
    f,
    downsampling_factor,
    fs_original,
    WLC_model,
    filter_type="bessel",
    order=4,
    decimate=True,
    cutoff_factor=0.8,
    polyorder=1
):
    def downsample_fd(
        d,
        f,
        downsampling_factor,
        fs_original,
        filter_type,
        order,
        decimate,
        cutoff_factor,
        polyorder
    ):

        if filter_type == "bessel":
            dlf = bessel_filtfilt(
                d,
                downsampling_factor,
                fs_original,
                order=order,
                decimate=decimate,
                cutoff_factor=cutoff_factor,
            )

            flf = bessel_filtfilt(
                f,
                downsampling_factor,
                fs_original,
                order=order,
                decimate=decimate,
                cutoff_factor=cutoff_factor,
            )

        elif filter_type == "butter":
            dlf = butter_filtfilt(
                d,
                downsampling_factor,
                fs_original,
                order=order,
                decimate=decimate,
                cutoff_factor=cutoff_factor,
            )

            flf = butter_filtfilt(
                f,
                downsampling_factor,
                fs_original,
                order=order,
                decimate=decimate,
                cutoff_factor=cutoff_factor,
            )

        elif filter_type == "boxcar":
            dlf = CK_filtfilt(
                d,
                downsampling_factor,
                decimate=decimate
            )

            flf = CK_filtfilt(
                f,
                downsampling_factor,
                decimate=decimate
            )

        elif filter_type == "savgol":
            dlf = savgol_filt(
                d,
                downsampling_factor=downsampling_factor,
                polyorder=polyorder,
                decimate=decimate
            )

            flf = savgol_filt(
                f,
                downsampling_factor=downsampling_factor,
                polyorder=polyorder,
                decimate=decimate
            )

        else:
            raise ValueError(
                "Enter a valid filter type: bessel, boxcar, butter, or savgol"
            )

        return dlf, flf


    dlf, flf = downsample_fd(
        d,
        f,
        downsampling_factor,
        fs_original,
        filter_type,
        order=order,
        decimate=decimate,
        cutoff_factor=cutoff_factor,
        polyorder=polyorder
    )

    lclf = quick_per_point_Lc(dlf, flf, *WLC_model)

    return lclf


def calculate_rmsd (distances, 
                    forces, 
                    *args,
                    distance_range=None, 
                    force_range=None,
                    already_segmented = True,
                    respect_transition = None):

    dfit = distances
    ffit = forces
    
    if not already_segmented:
        if respect_transition:
            dfit = distances[:respect_transition+1]
            ffit = forces[:respect_transition+1]

        mask =  (dfit > distance_range[0]) & \
                (dfit < distance_range[1]) & \
                (ffit > force_range[0]) & \
                (ffit < force_range[1])
    
        dfit = dfit[mask]
        ffit = ffit[mask]

    Fbar = ini_eWLC(0, dfit, *args)
    rmsd = np.sqrt(((ffit-Fbar)**2).mean())
    return(rmsd)

def sliding_progressive_rmse(wsize, measured_forces, predicted_forces):
    """
    Calculate the local RMSE between measured and predicted forces using a
    centered sliding window.

    The returned array has length:

        len(measured_forces) - wsize + 1

    No artificial values are appended.
    """

    measured_forces = np.asarray(measured_forces, dtype=float)
    predicted_forces = np.asarray(predicted_forces, dtype=float)

    if measured_forces.shape != predicted_forces.shape:
        raise ValueError(
            "measured_forces and predicted_forces must have the same shape."
        )

    if wsize < 1 or wsize % 2 == 0:
        raise ValueError("wsize must be a positive odd integer.")

    if wsize > len(measured_forces):
        raise ValueError("wsize cannot exceed the signal length.")

    squared_residuals = (measured_forces - predicted_forces) ** 2

    return np.sqrt(
        np.convolve(
            squared_residuals,
            np.ones(wsize) / wsize,
            mode="valid"
        )
    )

def sliding_progressive_rmse_old (wsize, measured_forces, predicted_forces):
    '''
    
    Description
    -----------
    This function performs a sliding window rmsd calculation between predicted
    and measured forces, centered around a nth data point. For
    a given data point in the force-extension curve, there is an associated
    sliding window rmsd that  has been calculated using the n-1, to n+1
    data points of the measured and predicted force vectors. In such a way, 
    the nth item of the rmsd vector represents the local deviation of that point
    in the fec with respect to the predicted forces according to the eWLC model.
    
    Parameters
    ----------
    wsize            : An odd integer stating the window size.
    measured_forces  : forces measured by the tweezers during force-ramp 
                       experiment
    predicted_forces : forces predicted using the corresponding extensions
                       using the eWLC model.

    Returns
    -------
    A numpy vector containing the computed rmsds. There is a 1:1 correspondance 
    of this vector to the data array vector.
    

    '''

    ## Empty list to append sliding window rmsds
    rmsds = []
    
    ## Defining variables that define the start and end of the window
    start = 0
    end = 0
    
    ## Loop until window exceed the last point of the list
    while end + wsize <= len(measured_forces):
        ## Calculate the rmsd between measured and theoretical forces within
        ## a given window
        rmsd = np.linalg.norm(measured_forces[start:end+wsize]\
                              - predicted_forces[start:end+wsize]) / np.sqrt(
                                  len(predicted_forces[start:end+wsize]))
        ## Append the value to the list
        rmsds.append(rmsd)
        
        ## Slide the window by 1
        start = start + 1
        end = end + 1
    
    ## Convert the rmsds to an array
    rmsds = np.array(rmsds)
    
    ## rmsds centered at the starting and ending data points are impossible
    ## to compute because there are no points before or after, respectively.
    ## To mantain a 1:1 correspondance zeros will be placed instead.
    #rmsds = np.insert(rmsds, 0,np.zeros( int((wsize-1)/2)) )
    #rmsds = np.insert(rmsds, len(rmsds),np.zeros( int((wsize-1)/2)))
    
    return(rmsds)

def get_unwrapping_dLc_nm(
    deltaX,
    L0,
    F1,
    F2,
    Lp,
    R,
    dLc_range,
    S=None,
    grid_spacing=0.01,
):
    """
    Solve for all dLc values satisfying:

        deltaX = eps(F2) * dLc
               + L0 * (eps(F2) - eps(F1))
               + peeling_rigid_body_contribution(dLc, R, units="nm")
               - 2*R
    """

    eps1 = get_normalized_extension(F1, Lp, S)
    eps2 = get_normalized_extension(F2, Lp, S)

    def predicted_deltaX(dLc):
        return (
            eps2 * dLc
            + L0 * (eps2 - eps1)
            + peeling_rigid_body_contribution(dLc, R, units="nm")
            - 2 * R
        )

    roots = solve_equation(
        predicted_deltaX,
        deltaX,
        x_range=dLc_range,
        grid_spacing=grid_spacing,
    )

    return roots

def detect_non_coop_transiton(
        d,
        F,
        pF,
        wsize=3,
        savgol_window=51,
        rmsd_nsigma=3,
        drmsd_nsigma=2
):
    """
    Detect the beginning of a persistent increase in RMSD.

    The complete RMSD signal is fitted with:

        Before transition: constant baseline
        After transition:  baseline + a*t + b*t**2

    Every possible transition index is tested. The selected index is the one
    that gives the best fit to the complete RMSD signal while also producing
    a positive post-transition derivative.

    No zeros or NaNs are appended.

    Returns
    -------
    rmsd_sd : float
        MAD-based standard deviation of the pre-transition RMSD.

    pass_threshold : tuple
        Indices in d, F, and pF whose RMSD exceeds the baseline threshold.

    potential_in : list
        Transition index in d, F, and pF.
        Empty if no transition is detected.

    rmsds : ndarray
        Sliding-window RMSD signal without padding.

    drmsds : ndarray
        Smoothed derivative of the RMSD signal.
    """

    d = np.asarray(d, dtype=float)
    F = np.asarray(F, dtype=float)
    pF = np.asarray(pF, dtype=float)

    if not (len(d) == len(F) == len(pF)):
        raise ValueError("d, F, and pF must have the same length.")

    if wsize < 1 or wsize % 2 == 0:
        raise ValueError("wsize must be a positive odd integer.")

    # No padding is performed.
    rmsds = np.asarray(
        sliding_progressive_rmse(wsize, F, pF),
        dtype=float
    )

    n = len(rmsds)

    if n < 15:
        raise ValueError("The RMSD signal is too short.")

    # rmsds[i] corresponds to the center of its original force window.
    data_offset = (wsize - 1) // 2

    # Make Savitzky-Golay window valid and odd.
    sg_window = min(savgol_window, n)

    if sg_window % 2 == 0:
        sg_window -= 1

    if sg_window < 5:
        raise ValueError(
            "The RMSD signal is too short for Savitzky-Golay filtering."
        )

    # Same length as rmsds. No padding.
    drmsds = signal.savgol_filter(
        rmsds,
        window_length=sg_window,
        polyorder=2,
        deriv=1
    )

    def mad_stats(values):
        """
        Use the existing mad() function with a standard-deviation fallback.
        """
        median, raw_mad, gaussian_sd = mad(values)

        if not np.isfinite(gaussian_sd) or gaussian_sd == 0:
            gaussian_sd = np.std(values)

        return median, raw_mad, gaussian_sd

    def no_transition_result():
        return None, rmsds, drmsds

    # Prevent extremely short regions at either end.
    # This is determined from the total signal length, not a fixed window.
    min_segment = max(5, int(0.05 * n))

    best_error = np.inf
    best_onset = None

    for onset in range(min_segment, n - min_segment):

        # Complete pre-transition region.
        pre_rmsd = rmsds[:onset]
        baseline, _, baseline_sd = mad_stats(pre_rmsd)

        if not np.isfinite(baseline_sd):
            continue

        # The derivative before transition should represent the baseline noise.
        drmsd_median, _, drmsd_sd = mad_stats(drmsds[:onset])

        if not np.isfinite(drmsd_sd):
            continue

        positive_derivative_threshold = max(
            0.0,
            drmsd_median + drmsd_nsigma * drmsd_sd
        )

        # Use the complete signal after the candidate onset.
        post_rmsd = rmsds[onset:]
        post_drmsd = drmsds[onset:]

        # The post-transition signal must have a significantly positive
        # derivative overall.
        if np.median(post_drmsd) <= positive_derivative_threshold:
            continue

        rmsd_threshold = baseline + rmsd_nsigma * baseline_sd

        # The post-transition RMSD must be globally higher than baseline.
        if np.median(post_rmsd) <= rmsd_threshold:
            continue

        # Fit a curved increase after the onset:
        #
        # rmsd = baseline + a*t + b*t**2
        #
        # The intercept is fixed to the pre-transition baseline, ensuring
        # continuity at the candidate transition.
        t = np.arange(len(post_rmsd), dtype=float)

        design_matrix = np.column_stack((
            t,
            t ** 2
        ))

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            post_rmsd - baseline,
            rcond=None
        )

        linear_coefficient = coefficients[0]
        quadratic_coefficient = coefficients[1]

        # Reject models that decrease after the transition.
        if linear_coefficient < 0 or quadratic_coefficient < 0:
            continue

        predicted_rmsds = np.empty(n, dtype=float)

        predicted_rmsds[:onset] = baseline

        predicted_rmsds[onset:] = (
            baseline
            + design_matrix @ coefficients
        )

        # Evaluate this candidate using the complete signal.
        residuals = rmsds - predicted_rmsds
        fitting_error = np.mean(residuals ** 2)

        if fitting_error < best_error:
            best_error = fitting_error
            best_onset = onset

    if best_onset is None:
        return no_transition_result()

    onset_rmsd_index = int(best_onset)

    # Calculate the final baseline and threshold.
    rmsd_median, _, rmsd_sd = mad_stats(
        rmsds[:onset_rmsd_index]
    )

    if not np.isfinite(rmsd_sd):
        return no_transition_result()

    rmsd_threshold = (
        rmsd_median
        + rmsd_nsigma * rmsd_sd
    )

    transition_data_index = (
        onset_rmsd_index
        + data_offset
    )

    return (transition_data_index,
        rmsds,
        drmsds)

def split_trappos_modes (time,
                         trappos, 
                         threshold = 0.001,
                         min_run_len = 5, 
                         cycle_number = 1, 
                         min_velocity_threshold = 9):
    
    
    def _clean_short_runs(labels, min_len=10):
        labels = labels.copy()
        edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]

        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < min_len:
                left = labels[a-1] if a > 0 else labels[b] if b < len(labels) else labels[a]
                labels[a:b] = left

        return labels
    
    dtrapposds = np.diff(trappos)
    
    class_array = np.zeros_like(dtrapposds, dtype=int)
    class_array[dtrapposds > threshold] = 1
    class_array[dtrapposds < -threshold] = -1
    class_array_clean = _clean_short_runs(class_array, min_len=min_run_len)
    
    pause_edges = []
    stretch_edges = []
    relax_edges = []
    
    # transitions between movement / pause / movement
    edges = np.r_[0, np.flatnonzero(class_array_clean[1:] != class_array_clean[:-1]) + 1, len(class_array_clean)]
    
    for a,b in zip(edges[:-1], edges[1:]):
        fit = np.polyfit(time[a:b+1], trappos[a:b+1]*1000,1)
    
        if fit[0] > min_velocity_threshold:
            stretch_edges.append([time[a],time[b]])
        elif fit[0] < -min_velocity_threshold:
            relax_edges.append([time[a],time[b]])
        else:
            pause_edges.append([time[a],time[b]])
    
    return(stretch_edges,relax_edges,pause_edges)
    
def compute_residence_time_hist(trace, bin_width=1.0, dt=1):
    """
    Compute residence time in uniform bins of given width.
    
    Returns:
        bin_edges   : array of bin edges (length N+1)
        bin_centers : array of bin centers (length N)
        tau         : residence time (seconds) per bin (length N)
    """

    # Define bin edges
    xmin, xmax = np.min(trace), np.max(trace)
    bin_edges = np.arange(xmin, xmax + bin_width, bin_width)

    # Assign samples to bins (0-based indices)
    idx = np.digitize(trace, bin_edges) - 1

    # Count samples per bin
    counts = np.bincount(idx, minlength=len(bin_edges))

    # Tau corresponds to bins = number of intervals = len(edges)-1
    tau = counts[:-1] * dt         # drop last padded count
    
    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_edges, bin_centers, tau

def pairwise_distribution_force_weighted(
        trace,
        force,
        bins=100,
        distance_range=None
):
    """
    Compute the force-weighted pairwise-distance distribution.

    Each pair (i, j) is weighted by its mean force:

        weight_ij = (force_i + force_j) / 2
    """

    x = np.asarray(trace, dtype=float)
    f = np.asarray(force, dtype=float)

    if len(x) != len(f):
        raise ValueError("trace and force must have the same length.")

    valid = np.isfinite(x) & np.isfinite(f)
    x = x[valid]
    f = f[valid]

    # Pairwise contour-length differences
    D = np.abs(x[:, None] - x[None, :])

    # Mean force associated with each pair
    W = 0.5 * (
        f[:, None] + f[None, :]
    )

    Dflat = D.ravel()
    Wflat = W.ravel()

    hist, edges = np.histogram(
        Dflat,
        bins=bins,
        range=distance_range,
        weights=Wflat,
        density=True
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    return centers, hist, Dflat, Wflat


def pairwise_distribution(trace, bins=100, distance_range=None):
    """
    Compute the exact pairwise-distance distribution for one trace.
    """
    x = np.asarray(trace, dtype=float)
    x = x[np.isfinite(x)]

    D = np.abs(x[:, None] - x[None, :])
    Dflat = D.ravel()

    hist, edges = np.histogram(
        Dflat,
        bins=bins,
        range=distance_range,
        density=True
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, hist, Dflat

def get_fd_bounds(d,f):
    min_dist, max_dist = min(d), max(d)
    min_force, max_force = min(f), max(f)
    return([min_dist, max_dist], [min_force, max_force])
    
def get_normalized_extension (F, Lp, S = None):
    if S != None:
        eps = ini_eWLC(1, F, [Lp,1,S])/1
    else:
        eps = ini_eWLC(1, F, [Lp,1])/1    
    return(eps)

def quick_per_point_Lc(d, F, *params):
    d = np.asarray(d, float)
    F = np.asarray(F, float)
    params = list(params)

    # find the single unknown index
    unknown_idx = None
    for i, par in enumerate(params):
        if (len(par) == 3 and par[1] is None) or (len(par) == 2 and par[1] is None):
            unknown_idx = i
            break
    if unknown_idx is None:
        raise ValueError("Provide exactly one element with Lc=None as the unknown.")

    # build eps_i(F) and sum known contributions
    x_known = np.zeros_like(d, dtype=float)
    eps_u = None
    for i, par in enumerate(params):
        if len(par) == 3:
            Lp_i, Lc_i, S_i = par
            eps_i = get_normalized_extension(F,Lp_i, S=S_i)
        elif len(par) == 2:
            Lp_i, Lc_i = par
            eps_i = get_normalized_extension(F,Lp_i)
        else:
            raise ValueError("Each element must be (Lp, Lc) or (Lp, Lc, S).")

        if i == unknown_idx:
            eps_u = np.asarray(eps_i, float)
        else:
            x_known += float(Lc_i) * np.asarray(eps_i, float)

    if eps_u is None:
        raise RuntimeError("Internal: did not capture unknown element’s epsilon.")

    Lc_unknown = (d - x_known) / eps_u
    return Lc_unknown
    
def find_ruptureidx(f_filt, threshold=0.5, distance=5, window=5):
    """
    Identify peaks in force signals that are followed by a drop > `threshold`.

    Args:
        force_filt (np.ndarray or pd.Series): Filtered force signals.
        threshold (float, optional): drop in force to consider a rupture. Default = 1.35.
        distance (int, optional): minimum index distance between peaks in samples. Default = 5.
        window (int, optional): number of points after peak to check drop. Default = 7.

    Returns:
        np.ndarray: array of indices of peaks that match criteria.
    """
    peaks, _ = find_peaks(f_filt, distance=distance)
    rupture_indices = []

    for peak in peaks:
        post_peak = f_filt[peak:peak + window]
        drop = f_filt[peak] - np.min(post_peak)

        if drop > threshold:
            rupture_indices.append(peak)
    return np.array(rupture_indices)

def find_transitions_peaks(indices,
                           flf,
                           force_cutoff=2,
                           threshold=0.5,
                           distance=5,
                           window=5,
                           add_rupture = True):
    mask = flf > force_cutoff

    trim_ind = indices[mask]
    trimflf = flf[mask]

    idxs = find_ruptureidx(trimflf, threshold=threshold, distance=distance, window=window)

    if isinstance(idxs, (list, np.ndarray)) and len(idxs) > 0:
        final_idxs = trim_ind[idxs]

        if add_rupture:
            final_idxs = np.append(final_idxs, indices[-1])

        return np.array(final_idxs)
    else:
        return np.array([indices[-1]])

def find_transitions_hmm (indices, flf, lctrace, force_cutoff = 2,max_states = 10,
                          force_states = False,
                          filtering = False):
    mask = flf > force_cutoff
    HMM = fitting.HMM_fit(lctrace[mask], max_states, forced_states=force_states,
                      filtering=filtering, annotate_transitions=True,
                      orind=indices[mask])
    HMM.fit()
    return(HMM, HMM.transitions_indexes)

def get_monotonic_segments(
    kv_trace,
    segments,
    sign=1,
    inclusive=True,
):
    """
    Split segments wherever kv_trace moves in the wrong direction.

    Parameters
    ----------
    kv_trace : array-like
        Trace to analyze.
    segments : array-like
        Input intervals.
    sign : {1, -1}
        1 finds nondecreasing segments.
        -1 finds nonincreasing segments.
    inclusive : bool
        If True, intervals use [start, end].
        If False, intervals use [start, end).

    Returns
    -------
    ndarray
        Monotonic, non-flat intervals.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")

    monotonic_segments = []

    for seg in segments:
        start, end = int(seg[0]), int(seg[-1])

        # Internally use end-exclusive indexing
        stop = end + 1 if inclusive else end
        x = kv_trace[start:stop]

        if sign == 1:
            breaks = np.where(np.diff(x) < 0)[0]
        else:
            breaks = np.where(np.diff(x) > 0)[0]

        last = start

        for b in breaks:
            cut = start + b + 1
            monotonic_segments.append([last, cut])
            last = cut

        monotonic_segments.append([last, stop])

    # Intervals are currently end-exclusive internally
    monotonic_segments = [
        [start, stop]
        for start, stop in monotonic_segments
        if np.any(np.diff(kv_trace[start:stop]) != 0)
    ]

    monotonic_segments = np.asarray(
        monotonic_segments,
        dtype=int,
    ).reshape(-1, 2)

    # Convert output back to inclusive intervals
    if inclusive:
        monotonic_segments[:, 1] -= 1

    return monotonic_segments
            
def extract_dwells_and_steps(kv):
    states = np.asarray(kv)

    # indices where state changes
    change_idx = np.where(np.diff(states) != 0)[0] + 1

    # start/end of each dwell
    edges = np.concatenate(([0], change_idx, [len(states)]))

    # dwell durations
    durations = np.diff(edges)

    # state of each dwell
    dwell_states = states[edges[:-1]]

    # step sizes between dwells
    step_sizes = np.diff(dwell_states)

    # dwell index ranges (start, end)
    dwell_indices = np.column_stack((edges[:-1], edges[1:]))

    return step_sizes, dwell_states, durations, dwell_indices

def get_dwell_intervals(ideal_trace, inclusive=True, offset=0):
    change_idx = np.where(np.diff(ideal_trace) != 0)[0] + 1

    starts = np.r_[0, change_idx]

    if inclusive:
        ends = np.r_[change_idx - 1, len(ideal_trace) - 1]
    else:
        ends = np.r_[change_idx, len(ideal_trace)]

    intervals = np.column_stack((starts, ends))

    levels = ideal_trace[starts]

    intervals += offset

    return intervals, levels

#%% for organizing to measure noise vs force!

def noise_vs_force_no_bins(lc, F, win=501, clip_q=0.995, spread_estimator = "MAD"):
    dd = np.diff(lc)
    Fm = 0.5*(F[:-1] + F[1:])

    # clip rare big steps so the local MAD reflects plateau noise
    thr = np.quantile(np.abs(dd[np.isfinite(dd)]), clip_q)
    dd = np.clip(dd, -thr, thr)

    if spread_estimator == "MAD":
        estimator_dd = running_mad(dd, win=win)
    
        # MAD->sigma and /sqrt(2) (optional but nice)
        sigma = (estimator_dd / 0.6745) / np.sqrt(2)
    else:
        estimator_dd = running_std(dd, win=win)
    
        # MAD->sigma and /sqrt(2) (optional but nice)
        sigma = estimator_dd  / np.sqrt(2)

    return Fm, sigma

def collect_noise_points_lists(lcs, forces,
                           win=501,
                           clip_q=0.995,
                           spread_estimator="MAD"):
    F_all = []
    s_all = []
    
    for lc, F in zip(lcs, forces):
        Fm, sigma = noise_vs_force_no_bins(
            lc, F,
            win=win,
            clip_q=clip_q,
            spread_estimator=spread_estimator
        )
    
        m = np.isfinite(Fm) & np.isfinite(sigma)
    
        F_all.append(Fm[m])
        s_all.append(sigma[m])
    
    return np.concatenate(F_all), np.concatenate(s_all)

def binned_summary(F, s, force_edges, min_points=2000):

    centers = 0.5*(force_edges[:-1] + force_edges[1:])

    med  = np.full_like(centers, np.nan, float)
    q25  = np.full_like(centers, np.nan, float)
    q75  = np.full_like(centers, np.nan, float)
    n    = np.zeros_like(centers, int)

    bin_idx = np.digitize(F, force_edges) - 1
    ok = (bin_idx >= 0) & (bin_idx < len(centers))

    for b in range(len(centers)):
        sb = s[ok & (bin_idx == b)]
        n[b] = sb.size
        if n[b] < min_points:
            continue
        med[b] = np.median(sb)
        q25[b] = np.quantile(sb, 0.25)
        q75[b] = np.quantile(sb, 0.75)

    return centers, med, q25, q75, n
#%% Main classes

class PutativeLLPsDetector:
    def __init__(
        self,
        d,
        f,
        sampling_rate,
        target_ds_factor_4_LLPs_det,
        pause_position_tolerance,
        WLCmodel,
        mad_thresh=5,
        split_pause_window_size_idx=313,
        min_pause_samples=1250,
        filter_type="bessel"
    ):

        self.d = d
        self.f = f
        self.sampling_rate = sampling_rate
        self.target_ds_factor_4_LLPs_det = target_ds_factor_4_LLPs_det
        self.mad_thresh = mad_thresh
        self.pause_position_tolerance = pause_position_tolerance
        self.split_pause_window_size_idx = split_pause_window_size_idx
        self.WLCmodel = WLCmodel
        self.min_pause_samples = min_pause_samples
        self.filter_type = filter_type

        self.lc_pause_det_filt = None
        self.pause_det_orind = None
        self.pause_centers = None
        self.pause_residence_samples = None
        self.pause_ends = None
        self.pause_starts = None
        self.pause_intervals_idx = None
        self.pause_levels = None
        self.pause_mask = None

        if len(self.d) != len(self.f):
            raise ValueError("d and f must have the same length")

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0")

        if self.target_ds_factor_4_LLPs_det < 1:
            raise ValueError("target_ds_factor_4_LLPs_det must be >= 1")

        if self.pause_position_tolerance <= 0:
            raise ValueError("pause_position_tolerance must be > 0")

        if self.split_pause_window_size_idx <= 0:
            raise ValueError("split_pause_window_size_idx must be > 0")

        if self.min_pause_samples < 0:
            raise ValueError("min_pause_samples must be >= 0")

        self.ori_idx = np.arange(len(self.d))


    def downsample_4_LLPs_det(self):

        if 2 * self.target_ds_factor_4_LLPs_det >= len(self.d):
            raise ValueError("Trace is too short for the requested filtering")

        self.lc_pause_det_filt = downsample_lc_trace(
            self.d,
            self.f,
            self.target_ds_factor_4_LLPs_det,
            self.sampling_rate,
            self.WLCmodel,
            filter_type=self.filter_type,
            order=3,
            cutoff_factor=1,
            decimate=False
        )

        # Trim edges
        self.lc_pause_det_filt = self.lc_pause_det_filt[
            self.target_ds_factor_4_LLPs_det:
            -self.target_ds_factor_4_LLPs_det
        ]

        self.pause_det_orind = self.ori_idx[
            self.target_ds_factor_4_LLPs_det:
            -self.target_ds_factor_4_LLPs_det
        ]


    def find_pause_candidates(self):

        # dt=1 means tau is measured in samples
        bin_edges, bin_centers, tau = compute_residence_time_hist(
            self.lc_pause_det_filt,
            bin_width=self.pause_position_tolerance,
            dt=1
        )

        med_tau, mad_tau, _ = mad(tau)

        threshold = med_tau + self.mad_thresh * mad_tau

        pause_bins_mask = tau > threshold

        self.pause_centers = bin_centers[pause_bins_mask]
        self.pause_residence_samples = tau[pause_bins_mask]

        # Map spatial pause bins onto samples
        idx = np.digitize(
            self.lc_pause_det_filt,
            bin_edges
        ) - 1

        idx = np.clip(
            idx,
            0,
            len(pause_bins_mask) - 1
        )

        self.pause_mask = pause_bins_mask[idx]

        # Find candidate pause intervals
        changes = np.diff(
            np.r_[False, self.pause_mask, False].astype(int)
        )

        self.pause_starts = np.flatnonzero(changes == 1)
        self.pause_ends = np.flatnonzero(changes == -1)


    def correct_pause_fragmentation(self):

        if len(self.pause_starts) > 1:

            merged_starts = [self.pause_starts[0]]
            merged_ends = []

            current_end = self.pause_ends[0]

            current_median = np.median(
                self.lc_pause_det_filt[
                    self.pause_starts[0]:self.pause_ends[0]
                ]
            )

            for start, end in zip(
                self.pause_starts[1:],
                self.pause_ends[1:]
            ):

                next_median = np.median(
                    self.lc_pause_det_filt[start:end]
                )

                if abs(
                    next_median - current_median
                ) <= self.pause_position_tolerance:

                    current_end = end

                    current_median = np.median(
                        self.lc_pause_det_filt[
                            merged_starts[-1]:current_end
                        ]
                    )

                else:

                    merged_ends.append(current_end)
                    merged_starts.append(start)

                    current_end = end
                    current_median = next_median

            merged_ends.append(current_end)

            self.pause_starts = np.asarray(merged_starts)
            self.pause_ends = np.asarray(merged_ends)

            self._rebuild_pause_mask()


    def correct_pause_overestimation(self):

        for start, end in zip(
            self.pause_starts,
            self.pause_ends
        ):

            if end - start < 2 * self.split_pause_window_size_idx + 1:
                continue

            for i in range(
                start + self.split_pause_window_size_idx,
                end - self.split_pause_window_size_idx
            ):

                left_median = np.median(
                    self.lc_pause_det_filt[
                        i - self.split_pause_window_size_idx:i
                    ]
                )

                right_median = np.median(
                    self.lc_pause_det_filt[
                        i:i + self.split_pause_window_size_idx
                    ]
                )

                if abs(
                    right_median - left_median
                ) > self.pause_position_tolerance:

                    self.pause_mask[i] = False

        # Get intervals after splitting
        changes = np.diff(
            np.r_[False, self.pause_mask, False].astype(int)
        )

        self.pause_starts = np.flatnonzero(changes == 1)
        self.pause_ends = np.flatnonzero(changes == -1)

        self._rebuild_pause_mask()


    def remove_short_pauses(self):

        if len(self.pause_starts) > 0:

            pause_lengths = (
                self.pause_ends - self.pause_starts
            )

            keep = pause_lengths >= self.min_pause_samples

            self.pause_starts = self.pause_starts[keep]
            self.pause_ends = self.pause_ends[keep]

            self._rebuild_pause_mask()


    def build_pause_intervals(self):

        # Original indexes
        # start and end are both inclusive
        self.pause_intervals_idx = np.column_stack((
            self.pause_det_orind[self.pause_starts],
            self.pause_det_orind[self.pause_ends - 1]
        ))


    def get_pause_levels(self):

        self.pause_levels = np.array([
            np.mean(self.lc_pause_det_filt[start:end])
            for start, end in zip(
                self.pause_starts,
                self.pause_ends
            )
        ])


    def run(self):

        self.downsample_4_LLPs_det()
        self.find_pause_candidates()
        self.correct_pause_fragmentation()
        self.correct_pause_overestimation()
        self.remove_short_pauses()
        self.build_pause_intervals()
        self.get_pause_levels()


    def _rebuild_pause_mask(self):

        self.pause_mask = np.zeros(
            len(self.lc_pause_det_filt),
            dtype=bool
        )

        for start, end in zip(
            self.pause_starts,
            self.pause_ends
        ):
            self.pause_mask[start:end] = True

    
class FindTransitionPaths:
    # find shortest path
    def __init__(self, trappos, force, trans_idxs, origin, destination, minpoints, reverse = False):
        self.trappos = trappos
        self.force = force
        self.trans_idxs = np.array(trans_idxs)
        self.minpoints = minpoints
        self.origin =origin
        self.destination = destination

        self.edges = None
        self.graph = None
        self.paths = None
        self.start_trans_position = None
        self.draft_rupt_force = None
        self.total_extension = 1
        self.reverse = reverse

    def test_valid_edge(self, current_trans_idx, force_off = 1, forward = True):
        current_trans_force = self.force[current_trans_idx]
        lbF = current_trans_force - 1
        ubF = current_trans_force + 1
        trans_idxs = self.trans_idxs
        current_trans_idx_pos = np.where(trans_idxs == current_trans_idx)[0][0]
        c = current_trans_idx_pos

        if forward:
            for next_trans_idx in trans_idxs[current_trans_idx_pos+1:len(trans_idxs)]:
                mirr_slice = self.trappos[trans_idxs[c]:next_trans_idx+1].reshape(-1,1)
                force_slice = self.force[trans_idxs[c]:next_trans_idx+1].reshape(-1,1)
                end_segment = np.hstack(  (mirr_slice, force_slice) )
                minF_indx_slice = np.argmin(end_segment[:,1])
                end_segment = end_segment[minF_indx_slice:len(end_segment)]

                tmp_mask = (end_segment[:,1] > lbF) & (end_segment[:,1] < ubF)
                if len(end_segment[tmp_mask]) > self.minpoints:
                    self.edges.append((current_trans_idx, next_trans_idx))
                c = c + 1

        else:
            rev_trans_idxs = trans_idxs[::-1]
            current_trans_idx_pos = np.where(rev_trans_idxs == current_trans_idx)[0][0]
            c = current_trans_idx_pos


            for prev_trans_idx in rev_trans_idxs[current_trans_idx_pos+1:len(rev_trans_idxs)]:

                mirr_slice = self.trappos[prev_trans_idx:rev_trans_idxs[c]+1].reshape(-1,1)
                force_slice = self.force[prev_trans_idx:rev_trans_idxs[c]+1].reshape(-1,1)
                end_segment = np.hstack(  (mirr_slice, force_slice) )
                minF_indx_slice = np.argmin(end_segment[:,1])
                end_segment = end_segment[minF_indx_slice:len(end_segment)]

                tmp_mask = (end_segment[:,1] > lbF) & (end_segment[:,1] < ubF)
                if len(end_segment[tmp_mask]) > self.minpoints:
                    self.edges.append((current_trans_idx, prev_trans_idx))
                c = c + 1

    def find_graph_edges(self):

        self.edges = []

        for current_trans_idx in self.trans_idxs:
            # forward scan
            self.test_valid_edge(current_trans_idx, force_off = 1, forward = True)

            if self.reverse:
                # reverse scan
                self.test_valid_edge(current_trans_idx, force_off = 1, forward = False)

    def build_graph(self):
        # Create an empty directed graph
        self.graph = nx.DiGraph()

        # Add nodes
        self.graph.add_nodes_from(self.trans_idxs)

        # Add edges
        self.graph.add_edges_from(self.edges)

    def find_paths(self):
        all_paths = list(nx.all_simple_paths(self.graph, source=self.origin, target=self.destination))
        self.paths = sorted(all_paths, key = len)

        if len(self.paths) == 0:
            #print("Imposible to calculate extension to the end.")
            self.total_extension = 0

    def run (self):
        stdout = self._test_valid_direction()

        if stdout:
            self.find_graph_edges()
            self.build_graph()
            self.find_paths()
        else:
            return

    def _test_valid_direction(self):
        if self.origin in self.trans_idxs:
            if self.destination in self.trans_idxs:
                return (True)
            else:
                return(False)
        else:
            return(False)

class FindKVTrace:
    def __init__(self, lctrace, PF = 5, min_step_size = 0.4, min_dwell_samples = 3):
        self.lctrace = lctrace
        self.PF = PF
        self.KV_trace = None
        self.min_step_size = min_step_size
        self.min_dwell_samples = min_dwell_samples
    
    def calculate_KV_trace(self):
        
        KVobj = CB_GaussStep(self.lctrace, PF = self.PF)
        KVobj.multi_step_finding()
        self.KV_trace = KVobj.fit
    
    def remove_short_kv_dwells(self):
        KV_trace = np.asarray(self.KV_trace, dtype=float).copy()

        while True:

            change_idx = np.where(np.diff(KV_trace) != 0)[0] + 1

            starts = np.r_[0, change_idx]
            ends = np.r_[change_idx, len(KV_trace)]
            levels = KV_trace[starts]

            dwell_lengths = ends - starts

            # Ignore first plateau because it cannot be merged backward
            short = np.where(dwell_lengths[1:] < self.min_dwell_samples)[0]

            if len(short) == 0:
                break

            i = short[0] + 1

            self._merge_plateau_backward(KV_trace,
                                         starts,
                                         ends,
                                         levels,
                                         i,
                                         )

        self.KV_trace = KV_trace
        
    def remove_small_kv_steps(self):
        KV_trace = np.asarray(self.KV_trace, dtype=float).copy()
        
        while True:

            change_idx = np.where(np.diff(KV_trace) != 0)[0] + 1

            starts = np.r_[0, change_idx]
            ends = np.r_[change_idx, len(KV_trace)]
            levels = KV_trace[starts]

            if len(levels) < 2:
                break

            steps = np.diff(levels)

            small = np.where(np.abs(steps) < self.min_step_size)[0]

            if len(small) == 0:
                break

            # step i goes from plateau i to plateau i+1
            i = small[0] + 1

            self._merge_plateau_backward(KV_trace,
                                         starts,
                                         ends,
                                         levels,
                                         i,
                                         )

        self.KV_trace = KV_trace
    
    def remove_large_kv_steps(self, max_step_size):
        KV_trace = np.asarray(self.KV_trace, dtype=float).copy()
    
        while True:
    
            change_idx = np.where(np.diff(KV_trace) != 0)[0] + 1
    
            starts = np.r_[0, change_idx]
            ends = np.r_[change_idx, len(KV_trace)]
            levels = KV_trace[starts]
    
            if len(levels) < 2:
                break
    
            steps = np.diff(levels)
    
            large = np.where(np.abs(steps) > max_step_size)[0]
    
            if len(large) == 0:
                break
    
            # step i goes from plateau i to plateau i+1
            i = large[0] + 1
    
            # If this is the last plateau, it can only merge backward
            if i == len(levels) - 1:
    
                self._merge_plateau_backward(
                    KV_trace,
                    starts,
                    ends,
                    levels,
                    i
                )
    
            else:
    
                distance_to_previous = abs(levels[i] - levels[i - 1])
                distance_to_next = abs(levels[i] - levels[i + 1])
    
                if distance_to_previous <= distance_to_next:
    
                    self._merge_plateau_backward(
                        KV_trace,
                        starts,
                        ends,
                        levels,
                        i
                    )
    
                else:
    
                    self._merge_plateau_forward(
                        KV_trace,
                        starts,
                        ends,
                        levels,
                        i
                    )
    
        self.KV_trace = KV_trace
    
    def run(self):
        self.calculate_KV_trace()
        self.remove_short_kv_dwells()
        self.remove_small_kv_steps()

    def _merge_plateau_backward(self, trace, starts, ends, levels, i):
        """
        Merge plateau i into the previous plateau.
        """
        trace[starts[i]:ends[i]] = levels[i - 1]

class ReconstructIdealTrace:
    def __init__(self, 
                 pause_intervals,
                 pause_segments,
                 KV_intervals,
                 KV_segments,
                 KV_segments_dwell_intervals,
                 step_size_tolerance = 0.4):
        
        self.pause_intervals = pause_intervals
        self.pause_segments = pause_segments
        self.KV_intervals = KV_intervals
        self.KV_segments = KV_segments
        self.KV_segments_dwell_intervals = KV_segments_dwell_intervals
        self.step_size_tolerance = step_size_tolerance
        
        self.mapping_table = None
        self.trace_len = None
        self.ideal_trace = None
    
    def get_mapping_table (self):
        mapping_table = []
        
        for i, (start, end) in enumerate(self.pause_intervals):
            mapping_table.append((start,end, "pause", i))
        
        for i, (start, end) in enumerate(self.KV_intervals):
            mapping_table.append((start,end, "non_pause", i))
        
        mapping_table.sort(key=lambda x: x[0])
        
        self.mapping_table = mapping_table
        self.trace_len = mapping_table[-1][1] + 1
    
    def stitch_segments (self):
        self.ideal_trace = np.zeros(self.trace_len)
        for start, end, label, i in self.mapping_table:
            if label == "pause":
                self.ideal_trace[start:end+1] =  self.pause_segments[i]
            else:
                self.ideal_trace[start:end+1] = self.KV_segments[i]
    
    def fix_missmatches (self):
        c = 0
        while c < len(self.mapping_table) -1:
            start_current, end_current, label_current, i_current = self.mapping_table[c]
            start_future, end_future, label_future, i_future = self.mapping_table[c+1]
            
            end_current_level = self.ideal_trace[end_current]
            start_future_level = self.ideal_trace[start_future]
            
            delta = abs(end_current_level - start_future_level)
            
            if delta < self.step_size_tolerance:
                if label_current == "pause":
                    interval_to_correct = self.KV_segments_dwell_intervals[i_future][0]
                    self.ideal_trace[interval_to_correct[0]:interval_to_correct[-1]+1] = end_current_level

                else:
                    interval_to_correct = self.KV_segments_dwell_intervals[i_current][-1]
                    self.ideal_trace[interval_to_correct[0]:interval_to_correct[1] + 1] = start_future_level
                

            c += 1
    
    def run(self):
        self.get_mapping_table()
        self.stitch_segments()
        self.fix_missmatches()
        
class ArmFinder:
    def __init__(self, trappos, force, trans_idxs, trans_params):
        self.trappos = trappos
        self.force = force
        self.trans_idxs = np.array(trans_idxs)
        self.trans_params = trans_params
        self.idx_map = {idx: i for i, idx in enumerate(self.trans_idxs)}

    def _slice(self, start, end):
        mirror_slice = self.trappos[start:end].reshape(-1, 1)
        force_slice = self.force[start:end].reshape(-1, 1)
        return np.hstack((mirror_slice, force_slice))

    def find_start_leg(self, trans_idx):
        current_idx_pos = self.idx_map[trans_idx]
        prev_idx = 0 if current_idx_pos == 0 else self.trans_idxs[current_idx_pos - 1]
        start = prev_idx + self.trans_params["padding"]
        end = trans_idx + 1

        start_leg = self._slice(start, end)

        if len(start_leg) > self.trans_params["nfit"]:
            leg = start_leg[-self.trans_params["nfit"]:]
            fit = np.polyfit(leg[:, 0], leg[:, 1], 1)
            f1 = np.polyval(fit, leg[-1, 0])
            x1 = (f1 - fit[1]) / fit[0]
            return leg, fit, f1, x1

        elif len(start_leg) > self.trans_params["navg"]:
            leg = start_leg[-self.trans_params["navg"]:]
            x1 = np.mean(leg[:, 0])
            f1 = np.mean(leg[:, 1])
            return leg, np.nan, f1, x1

        else:
            return np.nan, np.nan, np.nan, np.nan

    def find_end_leg(self, end_trans_idx, transF):
        pos = self.idx_map[end_trans_idx]
        prev_idx = 0 if pos == 0 else self.trans_idxs[pos - 1]
        start = prev_idx + self.trans_params["padding"]
        end = end_trans_idx + 1

        end_leg = self._slice(start, end)

        if len(end_leg) == 0:
            return np.nan, np.nan, np.nan

        minF_idx = np.argmin(end_leg[:, 1])
        end_leg = end_leg[minF_idx:]
        mask = (end_leg[:, 1] > transF - self.trans_params["FFitRange"]) & \
               (end_leg[:, 1] < transF + self.trans_params["FFitRange"])
        end_leg = end_leg[mask]

        if len(end_leg) > self.trans_params["nfit"]:
            tmpfit = np.polyfit(end_leg[:, 0], end_leg[:, 1], 1)
            fit_vals = np.polyval(tmpfit, end_leg[:, 0])
            cidx = np.argmin(np.abs(fit_vals - transF))

            half_nfit = int(self.trans_params["nfit"] // 2)
            lb = max(cidx - half_nfit, 0)
            ub = cidx + 1 + half_nfit
            end_leg = end_leg[lb:ub]

            fit = np.polyfit(end_leg[:, 0], end_leg[:, 1], 1)
            x2 = (transF - fit[1]) / fit[0]
            return end_leg, fit, x2

        elif len(end_leg) > self.trans_params["navg"]:
            cidx = np.argmin(np.abs(end_leg[:, 1] - transF))
            if np.abs(end_leg[cidx, 1] - transF) < self.trans_params["FAvgRange"]:
                half_navg = int(self.trans_params["navg"] // 2)
                lb = max(cidx - half_navg, 0)
                ub = cidx + 1 + half_navg
                end_leg = end_leg[lb:ub]

                x2 = np.mean(end_leg[:, 0])
                return end_leg, np.nan, x2

        return np.nan, np.nan, np.nan

class FindTransExtensions:
    def __init__(self, fec_id, trappos, force, trans_idxs,
                 trans_params = {"nfit" : 50, "navg" : 5, "padding": 5, "FAvgRange" : 0.1, "FFitRange" : 0.75,
                               "Lp": 50, "S": None},
                 debug = True):
        self.trappos = trappos
        self.force = force
        self.trans_idxs = np.array(trans_idxs)
        self.trans_params = trans_params
        self.fec_id = fec_id
        self.debug = debug
        self.total_direct_path = (self.trans_idxs[0], self.trans_idxs[-1])
        self.trans_fitter = ArmFinder(trappos, force, trans_idxs, trans_params)

        self.all_trans_info = {}
        self.rel_stts = []

# forward is not necessary, just add a condition to test is dest is higher than ori.
# also need to add a option to skip test and just give the extension if negative
    def find_path_extension(self, path, forward = True, sanity_test = True, bypass_sl = False):
        ori = path[0] if forward else path[-1]
        dest = path[-1] if forward else path[0]
        
        if bypass_sl:
            sl, sf, rF, ext1 = np.nan, np.nan, self.force[ori], self.trappos[ori]
        else:
            sl, sf, rF, ext1 = self.trans_fitter.find_start_leg(ori)

        if np.isnan(ext1):
            if np.isnan(rF):
                return(np.nan, np.nan, np.nan, np.nan, np.nan)
            else:
                return(rF, np.nan, np.nan, np.nan, np.nan)
        else:
            el, ef, ext2 = self.trans_fitter.find_end_leg(dest, rF)

            if not np.isnan(ext2):
                minext = min(el[:,0])
                maxext = max(el[:,0])

                if forward:
                    if sanity_test:
                        if ext1 > ext2 or (ext2 > maxext) or (ext2 < minext):
                            original_nfit = self.trans_params["nfit"]
                            self.trans_params["nfit"] = 60000
                            el, ef, ext2 = self.trans_fitter.find_end_leg(dest, rF)
                            self.trans_params["nfit"] = original_nfit
                    dExt = ext2-ext1

                else:
                    if sanity_test:
                        if ext1 < ext2 or (ext2 > maxext) or (ext2 < minext):
                            original_nfit = self.trans_params["nfit"]
                            self.trans_params["nfit"] = 60000
                            el, ef, ext2 = self.trans_fitter.find_end_leg(dest, rF)
                            self.trans_params["nfit"] = original_nfit
                    dExt = ext1-ext2
                
                eps = get_normalized_extension(rF, self.trans_params["Lp"], S = self.trans_params["S"])
                dLc = dExt/eps

                if not np.isnan(ext2):
                    if self.debug:
                        if bypass_sl:
                            plt.scatter(self.trappos[ori],self.force[ori], color = "blue", zorder = 2, s = 1)
                        else:
                            plt.scatter(sl[:,0],sl[:,1], color = "blue", zorder = 2, s = 1)
                       
                        plt.scatter(el[:,0],el[:,1], color = "red", zorder = 3, s = 1)

                        try:
                            plt.scatter(sl[:,0], np.polyval(sf, sl[:,0]), color = "black",s = 0.1,zorder = 4)
                        except:
                            pass

                        try:
                            plt.scatter(el[:,0], np.polyval(ef, el[:,0]), color = "black",s = 0.1,zorder = 4)
                        except:
                            pass

                        plt.plot(self.trappos, self.force, label = self.fec_id)
                        plt.axhline(rF)
                        plt.axvline(ext1, color = "blue",alpha = 0.4)
                        plt.axvline(ext2, color = "red",alpha=0.4)

                        plt.legend()
                        plt.show()

                return(rF, ext1, ext2, dExt, dLc)

            else:
                return(rF, ext1, np.nan, np.nan, np.nan)

    def find_forward_tdLc (self):
        rF, ext1, ext2, dExt, dLc = self.find_path_extension(self.total_direct_path, forward = True)
        trans_info = {}
        trans_info["rupture_force"] = rF
        trans_info["x1"] = ext1
        trans_info["x2"] = ext2
        trans_info["dExt"] = dExt
        trans_info["dLc"] = dLc

        self.all_trans_info[self.total_direct_path] = trans_info

    def find_reverse_tdLc (self):
        rF, ext1, ext2, dExt, dLc = self.find_path_extension(self.total_direct_path, forward = False)
        self.all_trans_info[self.total_direct_path]["dLc"] = dLc

    def find_all_forward_direct_path_extensions(self):
        for c, i in enumerate(self.trans_idxs[ 0:len(self.trans_idxs) -1] ):
            path = (i, self.trans_idxs[c+1])
            if path != self.total_direct_path:
                rF, ext1, ext2, dExt, dLc = self.find_path_extension(path, forward = True)
                #print(rF)
                trans_info = {}
                trans_info["rupture_force"] = rF
                trans_info["x1"] = ext1
                trans_info["x2"] = ext2
                trans_info["dExt"] = dExt
                trans_info["dLc"] = dLc

                self.all_trans_info[path] = trans_info

    def find_fixing_paths(self, gap_path):
        gap_path_source = gap_path[0]
        gap_path_destination = gap_path[-1]
        gap_path_destination_pos = np.where(self.trans_idxs == gap_path_destination)[0][0]
        searchable_indexes = self.trans_idxs[gap_path_destination_pos+1:len(self.trans_idxs)]

        potential_fixing_paths = []
        fixing_paths = []

        for i in searchable_indexes:

            trans_paths_obj = FindTransitionPaths(self.trappos,
                                                  self.force,
                                                  self.trans_idxs,
                                                  gap_path_destination,
                                                  i,
                                                  self.trans_params["navg"])
            trans_paths_obj.run()
            possible = trans_paths_obj.total_extension

            if possible:
                fixing_path = trans_paths_obj.paths
                potential_fixing_paths.append(fixing_path)

        potential_fixing_paths = sum(potential_fixing_paths, [])
        potential_fixing_paths = sorted(potential_fixing_paths, key = len)

        # test fixing path
        for i in potential_fixing_paths:
            trans_paths_obj = FindTransitionPaths(self.trappos,
                                                  self.force,
                                                  self.trans_idxs,
                                                  gap_path_source,
                                                  i[-1],
                                                  self.trans_params["navg"])
            trans_paths_obj.run()

            if trans_paths_obj.total_extension:
                fixing_paths.append(i)

        #fixing_paths = sum(fixing_paths,[])
        fixing_paths = sorted(fixing_paths, key = len)

        return(potential_fixing_paths,fixing_paths)

    def fill_gap_with_tdLc(self,gap_path):
        grandtotalLc = self.all_trans_info[self.total_direct_path]["dLc"]

        cum_dLc = 0
        for i in self.all_trans_info.keys():
            if i != self.total_direct_path:
                if i != gap_path:
                    cum_dLc = cum_dLc + self.all_trans_info[i]["dLc"]

        self.all_trans_info[gap_path]["dLc"] = grandtotalLc - cum_dLc

    def get_fixing_path(self, path):

        path = tuple(path)
        dLc = self.all_trans_info.get(path,{}).get("dLc")

        if dLc == None:
            rF, ext1, ext2, dExt, dLc = self.find_path_extension(path, forward = True)

        return(dLc)

    def fill_with_indirect_paths(self, gap_path):
        pot_fix_paths, fixing_paths = self.find_fixing_paths(gap_path)
        success = False

        for fp in fixing_paths:
            if len(fp) > 1:
                fp_route = [[fp[i], fp[i+1]] for i in range(len(fp) - 1)]
                dLc1 = 0

                for step in fp_route:
                    step_dLc = self.get_fixing_path(step)

                    dLc1 = dLc1 + step_dLc

            else:
                dLc1 = self.get_fixing_path(fp)

            if np.isnan(dLc1):
                pass
            else:
                # a --> c
                rF, ext1, ext2, dExt, dLc2 = self.find_path_extension((gap_path[0],fp[-1]), forward = True)
                if np.isnan(dLc2):
                    pass
                else:
                    self.all_trans_info[gap_path]["dLc"] = dLc2 - dLc1
                    success = True
                    break

        return(success)

    def fill_gaps(self):
        # detect gaps
        gap_paths = []
        grandtotalLc = self.all_trans_info[self.total_direct_path]["dLc"]

        for i in self.all_trans_info.keys():
            if i != self.total_direct_path:
                if np.isnan(self.all_trans_info[i]["dLc"]):
                    gap_paths.append(i)

        # if gap path = 1 AND grantotal is not empty then try using grandtotal
        if (len(gap_paths) == 1) and (not np.isnan(grandtotalLc)):
            self.fill_gap_with_tdLc(gap_paths[0])

        # Try to do indirect paths
        else:
            for i in gap_paths:
                success = self.fill_with_indirect_paths(i)
                if not success:
                # do reverse
                    rF, ext1, ext2, dExt, dLc = self.find_path_extension(i, forward = False)
                    self.all_trans_info[i]["dLc"] = dLc

                # if that fails then do indirect path
                #if np.isnan(dLc):
                 #   self.fill_with_indirect_paths(i)


    def get_rel_stts(self):

        rel_states = [0]
        cum_Lc = 0

        if len(self.all_trans_info.keys()) == 1:
            dLc = self.all_trans_info[self.total_direct_path]["dLc"]
            cum_Lc = cum_Lc + dLc
            rel_states.append(cum_Lc)

        else:
            for path in self.all_trans_info.keys():
                if path != self.total_direct_path:
                    dLc = self.all_trans_info[path]["dLc"]
                    cum_Lc = cum_Lc + dLc
                    rel_states.append(cum_Lc)

        self.rel_stts = rel_states

    def run(self):
        # try to find tdLc
        self.find_forward_tdLc()
        tdLc = self.all_trans_info[self.total_direct_path]["dLc"]

        # if empty try reverse
        if np.isnan(tdLc):
            self.find_reverse_tdLc()

        # find size of all transitions
        self.find_all_forward_direct_path_extensions()

        # try filling gaps
        self.fill_gaps()

        # get stts
        self.get_rel_stts()

class PiecewiseLinearSegmentation:
    def __init__(self, time, bp, max_changepoints=10, pen=3):
        self.time = np.asarray(time, dtype=float)
        self.bp = np.asarray(bp, dtype=float)

        if self.time.shape != self.bp.shape:
            raise ValueError("time and bp must have the same shape.")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError(
                "time must be strictly increasing; "
                "repeated or decreasing timestamps are not allowed."
            )

        self.N = len(self.time)
        self.max_changepoints = int(max_changepoints)
        self.pen = float(pen)

        # To be filled by fit()
        self.segments = None
        self.slopes = None
        self.durations = None
        self.lengths = None
        self.slope_class = None

    def fit(self):
        """
        Run ruptures PELT with a linear model (bp ~ time).
        Returns self.
        """

        # ---- 1. Build 2D input for ruptures: [y, X] = [bp, time] ----
        # First column = observed variable (bp), second column = covariate (time)
        signal = np.column_stack([self.bp, self.time])  # shape (N, 2)

        # ---- 2. Change-point detection ----
        algo = rpt.Pelt(model="linear").fit(signal)
        # You can tune `pen`; I kept it as a parameter
        change_idx = algo.predict(pen=self.pen)

        # ruptures returns the last index == N; we don't want that as a "changepoint"
        if change_idx and change_idx[-1] == self.N:
            change_idx = change_idx[:-1]

        # limit number of changepoints
        if len(change_idx) > self.max_changepoints:
            change_idx = change_idx[: self.max_changepoints]

        # boundaries in index space
        boundaries = [0] + change_idx + [self.N]
        segments = [(boundaries[i], boundaries[i + 1])
                    for i in range(len(boundaries) - 1)]
        self.segments = segments

        # ---- 3. Compute segment stats (like the MATLAB code) ----
        slopes = []
        durations = []
        lengths = []

        for (start, end) in segments:
            # Note: segments are [start, end) in Python
            t0 = self.time[start]
            t1 = self.time[end - 1]
            y0 = self.bp[start]
            y1 = self.bp[end - 1]

            duration = t1 - t0
            length = y1 - y0

            if duration <= 0:
                slope = 0.0
            else:
                slope = length / duration

            durations.append(duration)
            lengths.append(length)
            slopes.append(slope)

        self.slopes = np.asarray(slopes)
        self.durations = np.asarray(durations)
        self.lengths = np.asarray(lengths)

        # ---- 4. Slope classes, matching the MATLAB rules ----
        # slopeClass = 2  iff slope > 10
        # slopeClass = 1  iff (1 <= slope <= 10) AND (length > 100)
        # slopeClass = 0  otherwise
        slope_class = np.zeros(len(self.slopes), dtype=int)
        slope_class[self.slopes > 10.0] = 2

        mask1 = (
            (self.slopes >= 1.0)
            & (self.slopes <= 10.0)
            & (self.lengths > 100.0)
        )
        slope_class[mask1] = 1

        self.slope_class = slope_class
        return self

    def summary(self):
        return {
            "segments": self.segments,
            "slopes": self.slopes,
            "durations": self.durations,
            "lengths": self.lengths,
            "slope_class": self.slope_class,
        }


class EditTransitions:
    def __init__(self, fec, distance, force, HFd, HFf, trans_idxs, plot_bounds,
                 parameters, var_component=0, lctrace=None, curve_type="S",
                 states = None, WLCpars = [], non_coop = False):

        # normalize external inputs
        self.dlf = np.asarray(distance)
        self.flf = np.asarray(force)
        self.trans_idxs = np.asarray(trans_idxs, dtype=int)   # initial normalization
        self.fec = fec
        self.lctrace = None if lctrace is None else np.asarray(lctrace)
        self.HFd = np.asarray(HFd)
        self.HFf = np.asarray(HFf)
        self.plot_bounds = plot_bounds
        self.var_component = var_component
        self.parameters = parameters
        self.curve_type = curve_type
        self.states = states
        self.WLCpars = WLCpars
        self.user_input = None
        self.non_coop = non_coop

    # ---------- helpers ----------
    def _order_indices(self, arr):
        """Sort indices asc for 'S' and desc for 'R' using NumPy only."""
        arr = np.sort(arr)   # safe: arr always ndarray
        if self.curve_type == "R":
            arr = arr[::-1]
        return arr

    def _unique_concat(self, base, new):
        """Concatenate unique indices while preserving order in `new`."""
        mask = ~np.isin(new, base)
        if np.any(mask):
            return np.concatenate([base, new[mask]])
        return base

    # ---------- main interactions ----------
    def find_transition_in_slice(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            ind_slice, dlf_slice, flf_slice, lctrace_trim = self._slicer()
    
            # Esc/cancel from interactive selector
            if ind_slice is None:
                break
    
            if ind_slice.size > 0:
                plt.show()
                if self.parameters["rupture_force_detection"] == "peaks":
                    threshold = hand.is_valid_value("Enter a threshold value: ", float, min_value=0, max_value=None)
                    distance = hand.is_valid_value("Enter a distance drop value: ", int, min_value=1, max_value=None)
                    window = hand.is_valid_value("Enter a window value: ", int, min_value=1, max_value=None)
    
                    nidxs = find_transitions_peaks(ind_slice, flf_slice,
                                                   force_cutoff=0,
                                                   threshold=threshold,
                                                   distance=distance,
                                                   window=window,
                                                   add_rupture=False)
                    print(nidxs)
                else:
                    fixed_steps = hand.is_valid_value("Enter the number of steps: ", int, min_value=1)
                    _, nidxs = find_transitions_hmm(ind_slice, flf_slice, lctrace_trim,
                                                    force_cutoff=0,
                                                    max_states=fixed_steps,
                                                    force_states=True)
    
                if not isinstance(nidxs, (list, np.ndarray)) or len(nidxs) == 0:
                    print("No transitions found.")
                    user_input = hand.option_handler("Quit (q) or retry (r)? ", valid_values=["q", "r"])
                    if user_input == "q":
                        break
                    else:
                        continue
    
                nidxs = np.asarray(nidxs, dtype=int)   # normalize just once
                preview = self._unique_concat(self.trans_idxs, nidxs)
                self._plot_fec(preview, None)
    
                user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                                 valid_values=["y", "n", "q"])
    
                if user_input == "y":
                    self.trans_idxs = self._unique_concat(self.trans_idxs, nidxs)
                    self.trans_idxs = self._order_indices(self.trans_idxs)
                    break
                elif user_input == "q":
                    break
            else:
                print("Slice is empty. Define new window.")

    def find_non_cooperative_transition (self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            state_index = hand.is_valid_value(f"Enter the reference state (0-{len(self.states)-1}): ",
                                                    int,
                                                    min_value = 0,
                                                    max_value = len(self.states)-1)
            
            upper_trans_index = self.trans_idxs[state_index]
            
            if state_index == 0:
                lower_trans_index = 0
            else:
                lower_trans_index = self.trans_idxs[state_index-1]
            
            dslice = self.dlf[lower_trans_index:upper_trans_index+1]
            fslice = self.flf[lower_trans_index:upper_trans_index+1]
            
            ref_state = copy.deepcopy(self.WLCpars)
            ref_state[self.var_component][1] = self.states[state_index]

            try:
                pf = ini_eWLC(0, dslice, *ref_state)
            except:
                print("Empty slice.")
                user_input = hand.option_handler(
                    "Try again (y) or quit without applying changes (q)? ",
                    valid_values=["y", "q"]
                )
                
                if user_input == "q":
                    break
            
                continue
                
            
            transition_raw, _, _ = detect_non_coop_transiton(dslice, fslice, pf)
            
            if transition_raw is None:
                print("No non-cooperative transition was detected.")
            
                user_input = hand.option_handler(
                    "Try again (y) or quit without applying changes (q)? ",
                    valid_values=["y", "q"]
                )
            
                if user_input == "q":
                    break
            
                continue
            
            # transition_raw is relative to dslice.
            # Convert it to an index in the complete self.dlf/self.flf arrays.
            transition_idx = lower_trans_index + int(transition_raw)   
            
            nidxs = np.asarray([transition_idx], dtype=int)   # normalize just once
            preview = self._unique_concat(self.trans_idxs, nidxs)
            self._plot_fec(preview, None)

            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = self._unique_concat(self.trans_idxs, nidxs)
                self.trans_idxs = self._order_indices(self.trans_idxs)
                break
            elif user_input == "q":
                break

    def reset(self):
        while True:
            if self.curve_type == "S":
                flf = self.flf
                ind = np.arange(len(flf), dtype=int)
            elif self.curve_type == "R":
                flf = self.flf[::-1]
                ind = np.arange(len(flf), dtype=int)[::-1]
            else:
                return "Invalid curve type."

            if self.parameters["rupture_force_detection"] == "peaks":
                nidxs = find_transitions_peaks(ind, flf,
                                               force_cutoff=self.parameters.get("force_cutoff", 1),
                                               threshold=self.parameters.get("threshold", 0.5),
                                               distance=self.parameters.get("distance", 5),
                                               window=self.parameters.get("window", 5),
                                               add_rupture=False)
            else:
                _, nidxs = find_transitions_hmm(ind, flf, self.lctrace,
                                                force_cutoff=self.parameters.get("force_cutoff", 2),
                                                max_states=self.parameters.get("max_states", 10),
                                                force_states=False,
                                                filtering=True)

            nidxs = np.asarray(nidxs, dtype=int)   # normalize here
            self._plot_fec(nidxs,None)
            user_input = hand.option_handler("Satisfied (y) or quit loop without applying changes (q)? ",
                                             valid_values=["y", "q"])

            if user_input == "y":
                self.trans_idxs = nidxs
                break
            elif user_input == "q":
                break

    def remove_transition(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            max_valid = len(self.trans_idxs) - 1
            indexes_to_remove = hand.is_valid_indexes(
                f"Enter the index(es) of the transition to remove (0-{max_valid}) separated by space: ",
                max_valid)

            ntransitions_indexes = np.delete(self.trans_idxs, indexes_to_remove)
            self._plot_fec(ntransitions_indexes, None)

            user_input = hand.option_handler("Keep removed states (y/n) or quit removing loop without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = ntransitions_indexes
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue removing (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break

    def modify_force_index(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            transition_index_index = hand.is_valid_value(
                f"Enter the position of the transition index to modify (0-{len(self.trans_idxs)-1}): ",
                int, min_value=0, max_value=len(self.trans_idxs)-1)

            ntransition_index = hand.is_valid_value(
                f"Enter the new transition index (0-{len(self.dlf)-1}): ",
                int, min_value=0, max_value=len(self.dlf)-1)

            ntransitions_indexes = self.trans_idxs.copy()
            ntransitions_indexes[transition_index_index] = int(ntransition_index)

            self._plot_fec(ntransitions_indexes,None)
            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = ntransitions_indexes
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break

    def manually_add_transition(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            ntransition_index = hand.is_valid_value(
                f"Enter the new transition index (0-{len(self.dlf)-1}): ",
                int, min_value=0, max_value=len(self.dlf)-1)

            preview = np.append(self.trans_idxs, int(ntransition_index))
            self._plot_fec(preview, None)

            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = np.append(self.trans_idxs, int(ntransition_index))
                self.trans_idxs = self._order_indices(self.trans_idxs)
                break
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break

    def run(self):
        print(self.trans_idxs)
        self._plot_fec(self.trans_idxs, self.states)
        
        if self.non_coop:
            user_input = hand.option_handler(
                "Calculate transitions in slice (cs), calculate non-coop transition (cn),add transition manually (add), remove transition (rm), "
                "edit a transition (ed), reset (res), pass (p), or return to previous (r)? ",
                valid_values=["cs","cn", "add", "rm", "ed", "res", "p", "r"]
            )
        else:
            user_input = hand.option_handler(
                "Calculate transitions in slice (cs), add transition manually (add), remove transition (rm), "
                "edit a transition (ed), reset (res), pass (p), or return to previous (r)? ",
                valid_values=["cs", "add", "rm", "ed", "res", "p", "r"]
            )
        
        if user_input == "cs":
            self.find_transition_in_slice()
        elif user_input == "cn":
            self.find_non_cooperative_transition()
        elif user_input == "add":
            self.manually_add_transition()
        elif user_input == "rm":
            self.remove_transition()
        elif user_input == "ed":
            self.modify_force_index()
        elif user_input == "res":
            self.reset()
        elif user_input == "p":
            return "c"
        elif user_input == "r":
            return "r"

        self.user_input = hand.option_handler(
            "Keep working on the same fec (kw), return to previous (r) or continue to the next (c)? ",
            valid_values=["kw", "r", "c"]
        )
        return self.user_input

    def _define_slice_bounds_interactive(self):
        """
        Draw a rectangle around the region to use as the slice.
    
        The drawn box is used to choose data points. The returned bounds are
        refined to the actual min/max distance and force of the selected data.
    
        Controls:
            Left click + drag: draw/select box
            Adjust box normally
            Enter: accept current box
            Esc: cancel
    
        Returns:
            success, fd_bounds
        """
    

    
        previous_backend = plt.get_backend()
    
        selected = {"bounds": None}
        selector = None
        fig = None
        fd_bounds = None
    
        try:
            # Switch to Qt only when slice selection is requested.
            try:
                plt.switch_backend("QtAgg")
            except Exception:
                try:
                    plt.switch_backend("Qt5Agg")
                except Exception as e:
                    print("Could not switch to Qt backend for interactive slice selection.")
                    print(f"Current backend: {previous_backend}")
                    print(f"Error: {e}")
                    return False, None
    
            fig, ax = plt.subplots()
    
            # Full LF trace: same base color as your normal plot
            ax.plot(self.dlf, self.flf, lw=1.5, color="black")
    
            # Current transition points, preserving the original indexes.
            # No labels. No different color scheme.
            valid_trans = self.trans_idxs[
                (self.trans_idxs >= 0) &
                (self.trans_idxs < len(self.dlf))
            ]
    
            if valid_trans.size > 0:
                cmap = plt.get_cmap("tab20")
            
                for j, i in enumerate(valid_trans):
                    color = cmap(j % 20)
            
                    ax.scatter(
                        self.dlf[i],
                        self.flf[i],
                        s=20,
                        color=color,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=1
                    )
    
            # Selected part of trace, updated after box selection
            selected_trace, = ax.plot([], [], lw=1.5, color="red")
    
            ax.set_xlabel("Distance")
            ax.set_ylabel("Force")
            ax.set_title(
                "Draw a box around the slice region.\n"
                "Red trace = selected data points.\n"
                "Press Enter to accept. Press Esc to cancel."
            )
    
            try:
                ax.set_xlim(self.plot_bounds[0])
                ax.set_ylim(self.plot_bounds[1])
            except Exception:
                pass
    
            def onselect(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
    
                if x1 is None or x2 is None or y1 is None or y2 is None:
                    selected["bounds"] = None
                    selected_trace.set_data([], [])
                    fig.canvas.draw_idle()
                    return
    
                min_distance, max_distance = sorted([x1, x2])
                min_force, max_force = sorted([y1, y2])
    
                raw_bounds = [
                    [min_distance, max_distance],
                    [min_force, max_force]
                ]
    
                d = np.asarray(self.dlf)
                f = np.asarray(self.flf)
    
                raw_mask = (
                    (d >= min_distance) &
                    (d <= max_distance) &
                    (f >= min_force) &
                    (f <= max_force)
                )
    
                n_selected = np.sum(raw_mask)
    
                if n_selected == 0:
                    selected["bounds"] = None
                    selected_trace.set_data([], [])
                    print(f"Raw selected bounds: {raw_bounds}")
                    print("No data points inside selected box.")
                    fig.canvas.draw_idle()
                    return
    
                selected_d = d[raw_mask]
                selected_f = f[raw_mask]
    
                refined_bounds = [
                    [float(np.min(selected_d)), float(np.max(selected_d))],
                    [float(np.min(selected_f)), float(np.max(selected_f))]
                ]
    
                selected["bounds"] = refined_bounds
                selected_trace.set_data(selected_d, selected_f)
    
                print(f"Raw selected bounds: {raw_bounds}")
                print(f"Refined data bounds: {refined_bounds}")
                print(f"Selected points: {n_selected}")
    
                fig.canvas.draw_idle()
    
            def on_key(event):
                if event.key in ["enter", "return"]:
                    plt.close(fig)
    
                elif event.key == "escape":
                    selected["bounds"] = None
                    plt.close(fig)
    
            try:
                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True,
                    props={
                        "facecolor": "none",
                        "edgecolor": "black",
                        "linewidth": 1.0,
                        "alpha": 0.9
                    },
                    handle_props={
                        "marker": "s",
                        "markersize": 2,
                        "markeredgewidth": 0.8,
                        "markerfacecolor": "white",
                        "markeredgecolor": "black"
                    },
                    grab_range=5
                )
            except TypeError:
                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True
                )
    
            fig.canvas.mpl_connect("key_press_event", on_key)
    
            plt.show(block=True)
    
            fd_bounds = selected["bounds"]
    
        finally:
            try:
                if selector is not None:
                    selector.set_active(False)
            except Exception:
                pass
    
            try:
                if fig is not None:
                    plt.close(fig)
            except Exception:
                pass
    
            try:
                plt.switch_backend(previous_backend)
            except Exception:
                try:
                    from IPython import get_ipython
                    ip = get_ipython()
                    if ip is not None:
                        ip.run_line_magic("matplotlib", "inline")
                except Exception:
                    pass
    
        if fd_bounds is None:
            print("Interactive slice selection cancelled.")
            return False, None
    
        valid_dist_range = hand.is_valid_range(fd_bounds[0])
        valid_force_range = hand.is_valid_range(fd_bounds[1])
        success = False
    
        if all([valid_dist_range, valid_force_range]):
            success = True
            print(f"Selected slice bounds: {fd_bounds}")
    
        return (success, fd_bounds)

    def _slicer(self):
        success, slice_bounds = self._define_slice_bounds_interactive()

        if not success or slice_bounds is None:
            return None, None, None, None

        mask = (
            (self.dlf >= slice_bounds[0][0]) & (self.dlf <= slice_bounds[0][1]) &
            (self.flf >= slice_bounds[1][0]) & (self.flf <= slice_bounds[1][1])
        )

        ind = np.arange(len(self.flf), dtype=int)
        ind_slice = ind[mask]
        dlf_slice = self.dlf[mask]
        flf_slice = self.flf[mask]

        if self.parameters["rupture_force_detection"] == "hmm":
            lctrace_trim = self.lctrace[mask]
        else:
            lctrace_trim = None

        return ind_slice, dlf_slice, flf_slice, lctrace_trim

    def _plot_fec(self, transitions_indexes, states):
        transitions_indexes = np.asarray(transitions_indexes, dtype=int)  # normalize external arg
        fplt.plot_individual_fec(self.dlf,
                                 self.flf,
                                 self.fec,
                                 "black",
                                 self.plot_bounds[0],
                                 self.plot_bounds[1],
                                 "",
                                 *self.WLCpars,
                                 var_component = self.var_component,
                                 distances_hf=self.HFd,
                                 forces_hf=self.HFf,
                                 scatter=False,
                                 annot=False,
                                 savefig=False,
                                 rupture_index=False,
                                 lcstates=states,
                                 states_end_indexes=transitions_indexes,
                                 lc_trajectory=False)

class EditStates:
    def __init__(self, 
                 fec, 
                 distance, 
                 force, 
                 HFd, 
                 HFf, 
                 trans_idxs,
                 transition_finding_pars, 
                 states,
                 basal_lc,
                 discard, 
                 plot_bounds, 
                 *WLCpars, 
                 var_lc_component = 0, 
                 lc_trace = None,
                 curve_type = "S"):
        
        self.dlf = distance
        self.flf = force
        self.trans_idxs = trans_idxs
        self.fec = fec
        self.HFd = HFd
        self.HFf = HFf
        self.plot_bounds = plot_bounds
        self.var_lc_component = var_lc_component
        self.transition_finding_pars = transition_finding_pars
        self.states = states
        self.basal_lc = basal_lc
        self.curve_type = curve_type
        self.dLcs_sum = states - states[0]
        self.dLcs = [b - a for a, b in zip(self.states, self.states[1:])]
        self.dLcs.append(0)
        self.WLCpars =WLCpars
        self.lc_trace = lc_trace
        self.fixed_parameter_mask = None
        self.auto_trans_obj = None

    def edit_transitions(self):
        self.auto_trans_obj  = EditTransitions(self.fec, 
                                          self.dlf, 
                                          self.flf, 
                                          self.HFd, 
                                          self.HFf, 
                                          self.trans_idxs, 
                                          self.plot_bounds,
                                          self.transition_finding_pars, 
                                          var_component=self.var_lc_component, 
                                          lctrace=self.lc_trace, 
                                          curve_type=self.curve_type,
                                          states = self.states,
                                          WLCpars=self.WLCpars, 
                                          non_coop = True)
        self.auto_trans_obj.run()
        self._update_trans_and_states()
        
    def modify_state_by_fit(self):
        while True:
            print("Current states are " + str(self.states))
            
            state_index = hand.is_valid_value(f"Enter the position of state to modify (0-{len(self.states)-1}): ",
                                                    int,
                                                    min_value = 0,
                                                    max_value = len(self.trans_idxs)-1)
            
            upper_trans_index = self.trans_idxs[state_index]
            
            if state_index == 0:
                lower_trans_index = 0
            else:
                lower_trans_index = self.trans_idxs[state_index-1]
            

            raw_slice_points = len(self.dlf[lower_trans_index:upper_trans_index+1])
            
            lower_padding = hand.is_valid_value("Enter lower padding: ",
                                                int,
                                                min_value = 0,
                                                max_value = raw_slice_points)

            dfit = self.dlf[lower_trans_index+lower_padding:upper_trans_index+1]
            ffit = self.flf[lower_trans_index+lower_padding:upper_trans_index+1]
            
            self._gen_mask()
            
            guesses = copy.deepcopy(self.WLCpars)
            user_inp = hand.is_valid_value("Enter an Lc guess (um): ", float, min_value=None, max_value=None)
            
            guesses[self.var_lc_component][1] = user_inp
            
            fit_constructor = fitting.eWLC_fit(dfit, 
                                               ffit,
                                               guesses,
                                               fitting_bounds = None, 
                                               fixed_parameter_mask = self.fixed_parameter_mask,
                                               parameters_bounds = None,
                                               distance_offset = False,
                                               force_offset = False,
                                               distance_offset_bounds = None,
                                               force_offset_bounds = None,
                                               global_fit = False,
                                               invert=True,
                                               inversion_method="fast",)
            try:
                fit_constructor.fit()
            except:
                print("Fitting failed. Try decreasing padding and using a better Lc guess.")
                user_input = hand.option_handler("Try again (y/n)? ",
                                                valid_values=["y", "n"])
                if user_input == "n":
                    break
                else:
                    continue

            nstate = fit_constructor.fitobj[f"component_{self.var_lc_component}/Lc"].value
            
            nstates = self.states.copy()
            ndLcs_sum = self.dLcs_sum.copy()
            nstates[state_index] = nstate
            ndLcs = [b - a for a, b in zip(nstates, nstates[1:])]
            ndLcs.append(0)
            

            ndLcs_sum = nstates - nstates[0]


            self._plot_fec(self.trans_idxs,nstates)

            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                            valid_values=["y", "n", "q"])

            if user_input == "y":
                self.states = nstates
                self.dLcs_sum = ndLcs_sum
                self.dLcs = ndLcs

                
            elif user_input == "q":
                break
            
            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break
                        
    def modify_state_lc_manually(self):

        while True:
            print("Current states are " + str(self.states))

            state_index = hand.is_valid_value(f"Enter the position of state to modify (0-{len(self.states)-1}): ",
                                                    int,
                                                    min_value = 0,
                                                    max_value = len(self.trans_idxs)-1)

            nstate = hand.is_valid_value("Enter the new state lc: ",
                                               float,
                                               min_value = None,
                                               max_value = None)
            nstates = self.states.copy()
            ndLcs_sum = self.dLcs_sum.copy()
            nstates[state_index] = nstate
            ndLcs = [b - a for a, b in zip(nstates, nstates[1:])]
            ndLcs.append(0)
            
            ndLcs_sum = nstates - nstates[0]


            try:
                self._plot_fec(self.trans_idxs,nstates)
            except:
                print("Invalid state used. Did you put 0 by mistake?")
                user_input = hand.option_handler(
                    "Try again (y) or quit without applying changes (q)? ",
                    valid_values=["y", "q"]
                )
                
                if user_input == "q":
                    break
            
                continue   
                
            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                            valid_values=["y", "n", "q"])

            if user_input == "y":
                self.states = nstates
                self.dLcs_sum = ndLcs_sum
                self.dLcs = ndLcs
                
            elif user_input == "q":
                break
            
            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break
        
    def run(self):
        print(self.states)
        self._plot_fec(self.trans_idxs,self.states)
        user_input = hand.option_handler("Edit transitions (et), Edit states by fit (ft) , manually edit state (es), pass (p), or return to previous (r)? ",
                                          valid_values=["et","ft", "es","p","r"])

        if user_input == "et":
            self.edit_transitions()
            return(self.auto_trans_obj.user_input)
        elif user_input == "es":
            self.modify_state_lc_manually()
        elif user_input == "ft":
            self.modify_state_by_fit()
        elif user_input == "r":
            return("r")
        elif user_input == "p":
            return("c")

        user_input = hand.option_handler("Keep working on the same fec (kw), return to previous (r) or continue to the next (c)? ",
                                          valid_values=["kw", "r", "c"])
        return(user_input)
    
    def _update_trans_and_states(self):
        old_trans = np.asarray(self.trans_idxs, dtype=int)
        new_trans = np.asarray(self.auto_trans_obj.trans_idxs, dtype=int)
        old_states = np.asarray(self.states, dtype=float)
    
        from difflib import SequenceMatcher
    
        matcher = SequenceMatcher(
            a=old_trans.tolist(),
            b=new_trans.tolist(),
            autojunk=False
        )
    
        new_states = []
    
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                new_states.extend(old_states[i1:i2])
    
            elif tag == "delete":
                # Corresponding transitions and states were removed.
                continue
    
            elif tag == "insert":
                # Newly detected transitions do not yet have fitted states.
                new_states.extend([np.nan] * (j2 - j1))
    
            elif tag == "replace":
                old_count = i2 - i1
                new_count = j2 - j1
                shared_count = min(old_count, new_count)
    
                # Preserve states for transitions that were merely repositioned.
                new_states.extend(old_states[i1:i1 + shared_count])
    
                # Add placeholders if replacement produced additional transitions.
                new_states.extend([np.nan] * (new_count - shared_count))
    
        self.trans_idxs = new_trans
        self.states = np.asarray(new_states, dtype=float)
    
        # Keep all derived state arrays synchronized.
        self.dLcs_sum = self.states - self.states[0]

        self.dLcs = np.diff(self.states).tolist()
        self.dLcs.append(0)
    
    def _gen_mask(self):
        fixed_parameter_mask = []
        for c, pars in enumerate(self.WLCpars):
            bool_component_c = []
            for i in pars:
                bool_component_c.append(True)
            if c == self.var_lc_component:
                bool_component_c[1] = False
            fixed_parameter_mask.append(bool_component_c)
        self.fixed_parameter_mask = fixed_parameter_mask
    
    def _plot_fec(self, transition_indexes,states):
        # Set the color based on whether it's discarded or not
        #fec_color = "black" if self.discard == 0 else "red"
        
        fplt.plot_individual_fec(self.dlf,
                                 self.flf,
                                 self.fec,
                                 "black",
                                 self.plot_bounds[0],
                                 self.plot_bounds[1],
                                 "",
                                 *self.WLCpars,
                                 var_component = self.var_lc_component,
                                 distances_hf=self.HFd,
                                 forces_hf=self.HFf,
                                 scatter=False,
                                 annot=False,
                                 savefig=False,
                                 rupture_index=False,
                                 lcstates=states,
                                 states_end_indexes=transition_indexes,
                                 lc_trajectory=False)


class EditMotorTrace:
    def __init__(
        self,
        time,
        trace,
        trace_avg,
        ideal_trace,
        xlim,
        ylim,
        xtickspace,
        ytickspace,
        trace_id,
        step_size_tol=0.4,
        min_dwell_samples=5,
        KV_PF=4,
        pause_intervals=None,
    ):

        self.time = np.asarray(time)
        self.trace = np.asarray(trace)
        self.trace_avg = np.asarray(trace_avg)
        self.ideal_trace = np.asarray(ideal_trace)

        self.xlim = xlim
        self.ylim = ylim
        self.xtickspace = xtickspace
        self.ytickspace = ytickspace
        self.trace_id = trace_id

        self.step_size_tol = step_size_tol
        self.min_dwell_samples = min_dwell_samples
        self.KV_PF = KV_PF

        if pause_intervals is None:
            self.pause_intervals = None
        else:
            self.pause_intervals = np.asarray(
                pause_intervals,
                dtype=int
            )


    def trim_trace(self):

        while True:

            bounds, _, _ = self._select_rectangle(
                "Draw a box around the region to DELETE.\n"
                "Press Enter to accept. Press Esc to cancel."
            )

            if bounds is None:
                break

            t1, t2, _, _ = bounds

            mask = ~(
                (self.time >= t1) &
                (self.time <= t2)
            )

            ntime = self.time[mask]
            ntrace = self.trace[mask]
            ntrace_avg = self.trace_avg[mask]
            nideal_trace = self.ideal_trace[mask]

            # Update pause intervals to the new trace indexing
            npause_intervals = self._trim_pause_intervals(mask)

            nxlim = [ntime[0], ntime[-1]]

            fplt.plot_individual_motor_trace(
                ntime,
                ntrace,
                nideal_trace,
                nxlim,
                self.ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=ntrace_avg,
                savefig=False,
                pause_intervals=npause_intervals,
            )

            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )

            if user_input == "y":

                self.time = ntime
                self.trace = ntrace
                self.trace_avg = ntrace_avg
                self.ideal_trace = nideal_trace
                self.pause_intervals = npause_intervals
                self.xlim = nxlim

                break

            elif user_input == "q":
                break


    def replace_dwells(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box over the dwells to replace.\n"
                "The rectangle only selects dwells; complete dwell intervals are used.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            selected_dwells = []

            for i, (start, end) in enumerate(intervals):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print("No dwells selected.")
                continue


            selected_raw_points = []

            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                selected_raw_points.append(
                    self.trace[start:end]
                )


            selected_raw_points = np.concatenate(
                selected_raw_points
            )

            mean_level = np.mean(
                selected_raw_points
            )


            nideal_trace = self.ideal_trace.copy()

            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                nideal_trace[start:end] = mean_level


            preview_xlim = [
                max(self.xlim[0], t1 - 3),
                min(self.xlim[1], t2 + 3)
            ]

            preview_ylim = [
                y1 - 10,
                y2 + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )

            ax.plot(
                self.time,
                nideal_trace,
                color="cyan",
                linewidth=0.8,
                label="Proposed ideal trace"
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )


            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep replacing dwells with means (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def merge_dwells(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box around two dwells to merge.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            starts = intervals[:, 0]
            ends = intervals[:, 1]

            selected_dwells = []

            for i, (start, end) in enumerate(intervals):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print(
                    "No dwells selected. Select two."
                )
                continue

            elif len(selected_dwells) == 1:
                print(
                    "Only one dwell selected. Select two."
                )
                continue

            elif len(selected_dwells) > 2:
                print(
                    "More than two dwells selected. Select two."
                )
                continue


            first_dwell = selected_dwells[0]
            second_dwell = selected_dwells[1]


            preview_xlim = [
                max(self.xlim[0], t1 - 3),
                min(self.xlim[1], t2 + 3)
            ]

            preview_ylim = [
                y1 - 10,
                y2 + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )


            for j, dwell_i in enumerate(
                selected_dwells
            ):

                start = starts[dwell_i]
                end = ends[dwell_i]

                label = (
                    "Selected dwells"
                    if j == 0
                    else None
                )

                ax.plot(
                    self.time[start:end],
                    self.ideal_trace[start:end],
                    color="cyan",
                    linewidth=2,
                    label=label
                )


            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Merge dwells by keeping earlier dwell level (e), "
                "keeping later dwell level (l), or quit (q)? ",
                valid_values=["e", "l", "q"]
            )

            if user_input == "q":
                break


            nideal_trace = self.ideal_trace.copy()


            if user_input == "l":

                self._merge_plateau_forward(
                    nideal_trace,
                    starts,
                    ends,
                    levels,
                    first_dwell
                )


            elif user_input == "e":

                self._merge_plateau_backward(
                    nideal_trace,
                    starts,
                    ends,
                    levels,
                    second_dwell
                )


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )

            ax.plot(
                self.time,
                nideal_trace,
                color="cyan",
                linewidth=0.8,
                label="Proposed ideal trace"
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )

            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep merging dwells (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def refit_segment(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box over the dwell(s) to refit.\n"
                "Complete dwell intervals will be refitted using the raw data.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            selected_dwells = []

            for i, (start, end) in enumerate(
                intervals
            ):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print("No dwells selected.")
                continue


            while True:

                PF_input = input(
                    f"Penalty factor for KV refit "
                    f"[press Enter for default {self.KV_PF}]: "
                ).strip()

                if PF_input == "":
                    PF = self.KV_PF
                    break

                try:
                    PF = float(PF_input)
                    break

                except ValueError:
                    print(
                        "Penalty factor must be a number."
                    )


            print(
                f"Using KV penalty factor: {PF}"
            )


            nideal_trace = self.ideal_trace.copy()


            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                lcseg = self.trace[start:end]

                KVobj = FindKVTrace(
                    lcseg,
                    PF=PF,
                    min_step_size=self.step_size_tol,
                    min_dwell_samples=self.min_dwell_samples
                )

                KVobj.run()

                nideal_trace[start:end] = (
                    KVobj.KV_trace
                )


            first_start = (
                intervals[selected_dwells[0]][0]
            )

            last_end = (
                intervals[selected_dwells[-1]][1]
            )

            preview_xlim = [
                max(
                    self.xlim[0],
                    self.time[first_start] - 3
                ),
                min(
                    self.xlim[1],
                    self.time[last_end - 1] + 3
                )
            ]


            selected_trace = (
                self.trace[first_start:last_end]
            )

            selected_original = (
                self.ideal_trace[first_start:last_end]
            )

            selected_proposed = (
                nideal_trace[first_start:last_end]
            )

            preview_ylim = [
                min(
                    np.min(selected_trace),
                    np.min(selected_original),
                    np.min(selected_proposed)
                ) - 10,

                max(
                    np.max(selected_trace),
                    np.max(selected_original),
                    np.max(selected_proposed)
                ) + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )


            for j, dwell_i in enumerate(
                selected_dwells
            ):

                start, end = intervals[dwell_i]

                label = (
                    "Proposed KV refit"
                    if j == 0
                    else None
                )

                ax.plot(
                    self.time[start:end],
                    nideal_trace[start:end],
                    color="cyan",
                    linewidth=1.2,
                    label=label
                )


            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )


            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep refitting segments (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def run(self):

        self._plot_trace()

        user_input = hand.option_handler(
            "Trim trace (tr), replace dwells with means (rd), "
            "merge dwells (md), refit segment (rf), "
            "discard trace (d), pass (p), or return to previous (r)? ",
            valid_values=[
                "tr",
                "rd",
                "md",
                "rf",
                "d",
                "p",
                "r"
            ]
        )

        if user_input == "tr":
            self.trim_trace()

        elif user_input == "rd":
            self.replace_dwells()

        elif user_input == "md":
            self.merge_dwells()

        elif user_input == "rf":
            self.refit_segment()

        elif user_input == "d":
            return("d")

        elif user_input == "r":
            return("r")

        elif user_input == "p":
            return("c")


        user_input = hand.option_handler(
            "Keep working on the same trace (kw), discard trace (d), "
            "return to previous (r), or continue to the next (c)? ",
            valid_values=["kw", "d", "r", "c"]
        )

        return(user_input)


    def _merge_plateau_backward(
        self,
        trace,
        starts,
        ends,
        levels,
        i
    ):

        trace[
            starts[i]:ends[i]
        ] = levels[i - 1]


    def _merge_plateau_forward(
        self,
        trace,
        starts,
        ends,
        levels,
        i
    ):

        trace[
            starts[i]:ends[i]
        ] = levels[i + 1]


    def _trim_pause_intervals(self, mask):

        if self.pause_intervals is None:
            return None

        if len(self.pause_intervals) == 0:
            return np.empty(
                (0, 2),
                dtype=int
            )


        # Map old indices to the new indices after trimming.
        old_to_new = np.full(
            len(mask),
            -1,
            dtype=int
        )

        old_to_new[mask] = np.arange(
            np.sum(mask)
        )


        new_intervals = []

        for start, end in self.pause_intervals:

            # Original indices belonging to this pause.
            pause_indices = np.arange(
                start,
                end + 1
            )

            # Keep only pause samples that survived trimming.
            surviving = pause_indices[
                mask[pause_indices]
            ]

            if len(surviving) == 0:
                continue


            # If trimming cut through the middle of a pause,
            # split the remaining parts into separate intervals.
            split_points = (
                np.where(
                    np.diff(surviving) > 1
                )[0] + 1
            )

            surviving_groups = np.split(
                surviving,
                split_points
            )


            for group in surviving_groups:

                if len(group) == 0:
                    continue

                new_start = old_to_new[
                    group[0]
                ]

                new_end = old_to_new[
                    group[-1]
                ]

                new_intervals.append(
                    [new_start, new_end]
                )


        if len(new_intervals) == 0:
            return np.empty(
                (0, 2),
                dtype=int
            )

        return np.asarray(
            new_intervals,
            dtype=int
        )


    def _select_rectangle(
        self,
        title,
        xlim=None,
        ylim=None
    ):

        from matplotlib.ticker import (
            MultipleLocator,
            NullFormatter
        )

        previous_backend = plt.get_backend()

        selected = {
            "bounds": None,
            "xlim": None,
            "ylim": None
        }

        selector = None
        fig = None

        if xlim is None:
            xlim = self.xlim

        if ylim is None:
            ylim = self.ylim


        try:

            try:
                plt.switch_backend("QtAgg")

            except Exception:

                try:
                    plt.switch_backend(
                        "Qt5Agg"
                    )

                except Exception as e:

                    print(
                        "Could not switch to Qt backend."
                    )
                    print(
                        f"Current backend: {previous_backend}"
                    )
                    print(
                        f"Error: {e}"
                    )

                    return (
                        None,
                        list(xlim),
                        list(ylim)
                    )


            fig, ax = (
                fplt.plot_individual_motor_trace(
                    self.time,
                    self.trace,
                    self.ideal_trace,
                    xlim,
                    ylim,
                    self.xtickspace,
                    self.ytickspace,
                    self.trace_id,
                    lctrace_averaged=self.trace_avg,
                    savefig=False,
                    show=False,
                    pause_intervals=self.pause_intervals,
                )
            )

            ax.set_title(title)


            ax.xaxis.set_minor_locator(
                MultipleLocator(3)
            )

            ax.yaxis.set_minor_locator(
                MultipleLocator(3)
            )

            ax.xaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.yaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.tick_params(
                axis="both",
                which="minor",
                length=3
            )


            def onselect(
                eclick,
                erelease
            ):

                x1 = eclick.xdata
                x2 = erelease.xdata
                y1 = eclick.ydata
                y2 = erelease.ydata

                if (
                    x1 is None or
                    x2 is None or
                    y1 is None or
                    y2 is None
                ):

                    selected["bounds"] = None
                    return


                t1, t2 = sorted(
                    [x1, x2]
                )

                ymin, ymax = sorted(
                    [y1, y2]
                )

                selected["bounds"] = (
                    t1,
                    t2,
                    ymin,
                    ymax
                )


            def on_key(event):

                if event.key in [
                    "enter",
                    "return"
                ]:

                    selected["xlim"] = list(
                        ax.get_xlim()
                    )

                    selected["ylim"] = list(
                        ax.get_ylim()
                    )

                    plt.close(fig)


                elif event.key == "escape":

                    selected["bounds"] = None

                    selected["xlim"] = list(
                        ax.get_xlim()
                    )

                    selected["ylim"] = list(
                        ax.get_ylim()
                    )

                    plt.close(fig)


                elif event.key == "r":

                    ax.set_xlim(
                        self.xlim
                    )

                    ax.set_ylim(
                        self.ylim
                    )

                    selected["xlim"] = list(
                        self.xlim
                    )

                    selected["ylim"] = list(
                        self.ylim
                    )

                    fig.canvas.draw_idle()


            try:

                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True,
                    props={
                        "facecolor": "none",
                        "edgecolor": "black",
                        "linewidth": 1.0,
                        "alpha": 0.9,
                    },
                    handle_props={
                        "marker": "s",
                        "markersize": 2,
                        "markeredgewidth": 0.8,
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                    },
                    grab_range=5,
                )

            except TypeError:

                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True,
                )


            fig.canvas.mpl_connect(
                "key_press_event",
                on_key
            )

            plt.show(block=True)


            bounds = selected["bounds"]

            if selected["xlim"] is None:

                selected["xlim"] = list(
                    ax.get_xlim()
                )

            if selected["ylim"] is None:

                selected["ylim"] = list(
                    ax.get_ylim()
                )


        finally:

            try:

                if selector is not None:
                    selector.set_active(False)

            except Exception:
                pass


            try:

                if fig is not None:
                    plt.close(fig)

            except Exception:
                pass


            try:
                plt.switch_backend(
                    previous_backend
                )

            except Exception:

                try:

                    from IPython import get_ipython

                    ip = get_ipython()

                    if ip is not None:

                        ip.run_line_magic(
                            "matplotlib",
                            "inline"
                        )

                except Exception:
                    pass


        return(
            bounds,
            selected["xlim"],
            selected["ylim"]
        )


    def _plot_trace(self):

        fplt.plot_individual_motor_trace(
            self.time,
            self.trace,
            self.ideal_trace,
            self.xlim,
            self.ylim,
            self.xtickspace,
            self.ytickspace,
            self.trace_id,
            lctrace_averaged=self.trace_avg,
            savefig=False,
            pause_intervals=self.pause_intervals,
        )



class EditMotorTrace_old:
    def __init__(
        self,
        time,
        trace,
        trace_avg,
        ideal_trace,
        xlim,
        ylim,
        xtickspace,
        ytickspace,
        trace_id,
        step_size_tol=0.4,
        min_dwell_samples=5,
        KV_PF=4,
        pause_intervals=None,
    ):

        self.time = np.asarray(time)
        self.trace = np.asarray(trace)
        self.trace_avg = np.asarray(trace_avg)
        self.ideal_trace = np.asarray(ideal_trace)

        self.xlim = xlim
        self.ylim = ylim
        self.xtickspace = xtickspace
        self.ytickspace = ytickspace
        self.trace_id = trace_id

        self.step_size_tol = step_size_tol
        self.min_dwell_samples = min_dwell_samples
        self.KV_PF = KV_PF

        if pause_intervals is None:
            self.pause_intervals = None
        else:
            self.pause_intervals = np.asarray(
                pause_intervals,
                dtype=int
            )


    def trim_trace(self):

        while True:

            bounds, _, _ = self._select_rectangle(
                "Draw a box around the region to DELETE.\n"
                "Press Enter to accept. Press Esc to cancel."
            )

            if bounds is None:
                break

            t1, t2, _, _ = bounds

            mask = ~(
                (self.time >= t1) &
                (self.time <= t2)
            )

            ntime = self.time[mask]
            ntrace = self.trace[mask]
            ntrace_avg = self.trace_avg[mask]
            nideal_trace = self.ideal_trace[mask]

            # Update pause intervals to the new trace indexing
            npause_intervals = self._trim_pause_intervals(mask)

            nxlim = [ntime[0], ntime[-1]]

            fplt.plot_individual_motor_trace(
                ntime,
                ntrace,
                nideal_trace,
                nxlim,
                self.ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=ntrace_avg,
                savefig=False,
                pause_intervals=npause_intervals,
            )

            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )

            if user_input == "y":

                self.time = ntime
                self.trace = ntrace
                self.trace_avg = ntrace_avg
                self.ideal_trace = nideal_trace
                self.pause_intervals = npause_intervals
                self.xlim = nxlim

                break

            elif user_input == "q":
                break


    def replace_dwells(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box over the dwells to replace.\n"
                "The rectangle only selects dwells; complete dwell intervals are used.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            selected_dwells = []

            for i, (start, end) in enumerate(intervals):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print("No dwells selected.")
                continue


            selected_raw_points = []

            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                selected_raw_points.append(
                    self.trace[start:end]
                )


            selected_raw_points = np.concatenate(
                selected_raw_points
            )

            mean_level = np.mean(
                selected_raw_points
            )


            nideal_trace = self.ideal_trace.copy()

            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                nideal_trace[start:end] = mean_level


            preview_xlim = [
                max(self.xlim[0], t1 - 3),
                min(self.xlim[1], t2 + 3)
            ]

            preview_ylim = [
                y1 - 10,
                y2 + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )

            ax.plot(
                self.time,
                nideal_trace,
                color="cyan",
                linewidth=0.8,
                label="Proposed ideal trace"
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )


            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep replacing dwells with means (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def merge_dwells(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box around two dwells to merge.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            starts = intervals[:, 0]
            ends = intervals[:, 1]

            selected_dwells = []

            for i, (start, end) in enumerate(intervals):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print(
                    "No dwells selected. Select two."
                )
                continue

            elif len(selected_dwells) == 1:
                print(
                    "Only one dwell selected. Select two."
                )
                continue

            elif len(selected_dwells) > 2:
                print(
                    "More than two dwells selected. Select two."
                )
                continue


            first_dwell = selected_dwells[0]
            second_dwell = selected_dwells[1]


            preview_xlim = [
                max(self.xlim[0], t1 - 3),
                min(self.xlim[1], t2 + 3)
            ]

            preview_ylim = [
                y1 - 10,
                y2 + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )


            for j, dwell_i in enumerate(
                selected_dwells
            ):

                start = starts[dwell_i]
                end = ends[dwell_i]

                label = (
                    "Selected dwells"
                    if j == 0
                    else None
                )

                ax.plot(
                    self.time[start:end],
                    self.ideal_trace[start:end],
                    color="cyan",
                    linewidth=2,
                    label=label
                )


            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Merge dwells by keeping earlier dwell level (e), "
                "keeping later dwell level (l), or quit (q)? ",
                valid_values=["e", "l", "q"]
            )

            if user_input == "q":
                break


            nideal_trace = self.ideal_trace.copy()


            if user_input == "l":

                self._merge_plateau_forward(
                    nideal_trace,
                    starts,
                    ends,
                    levels,
                    first_dwell
                )


            elif user_input == "e":

                self._merge_plateau_backward(
                    nideal_trace,
                    starts,
                    ends,
                    levels,
                    second_dwell
                )


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )

            ax.plot(
                self.time,
                nideal_trace,
                color="cyan",
                linewidth=0.8,
                label="Proposed ideal trace"
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )

            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep merging dwells (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def refit_segment(self):

        selection_xlim = list(self.xlim)
        selection_ylim = list(self.ylim)

        while True:

            bounds, selection_xlim, selection_ylim = self._select_rectangle(
                "Draw a box over the dwell(s) to refit.\n"
                "Complete dwell intervals will be refitted using the raw data.\n"
                "Press Enter to accept. Press Esc to cancel.\n"
                "Press r to reset zoom.",
                xlim=selection_xlim,
                ylim=selection_ylim
            )

            if bounds is None:
                break

            t1, t2, y1, y2 = bounds

            intervals, levels = get_dwell_intervals(
                self.ideal_trace,
                inclusive=False
            )

            selected_dwells = []

            for i, (start, end) in enumerate(
                intervals
            ):

                dwell_start_time = self.time[start]
                dwell_end_time = self.time[end - 1]

                overlaps_selection = (
                    dwell_end_time >= t1 and
                    dwell_start_time <= t2
                )

                if overlaps_selection:
                    selected_dwells.append(i)


            if len(selected_dwells) == 0:
                print("No dwells selected.")
                continue


            while True:

                PF_input = input(
                    f"Penalty factor for KV refit "
                    f"[press Enter for default {self.KV_PF}]: "
                ).strip()

                if PF_input == "":
                    PF = self.KV_PF
                    break

                try:
                    PF = float(PF_input)
                    break

                except ValueError:
                    print(
                        "Penalty factor must be a number."
                    )


            print(
                f"Using KV penalty factor: {PF}"
            )


            nideal_trace = self.ideal_trace.copy()


            for dwell_i in selected_dwells:

                start, end = intervals[dwell_i]

                lcseg = self.trace[start:end]

                KVobj = FindKVTrace(
                    lcseg,
                    PF=PF,
                    min_step_size=self.step_size_tol,
                    min_dwell_samples=self.min_dwell_samples
                )

                KVobj.run()

                nideal_trace[start:end] = (
                    KVobj.KV_trace
                )


            first_start = (
                intervals[selected_dwells[0]][0]
            )

            last_end = (
                intervals[selected_dwells[-1]][1]
            )

            preview_xlim = [
                max(
                    self.xlim[0],
                    self.time[first_start] - 3
                ),
                min(
                    self.xlim[1],
                    self.time[last_end - 1] + 3
                )
            ]


            selected_trace = (
                self.trace[first_start:last_end]
            )

            selected_original = (
                self.ideal_trace[first_start:last_end]
            )

            selected_proposed = (
                nideal_trace[first_start:last_end]
            )

            preview_ylim = [
                min(
                    np.min(selected_trace),
                    np.min(selected_original),
                    np.min(selected_proposed)
                ) - 10,

                max(
                    np.max(selected_trace),
                    np.max(selected_original),
                    np.max(selected_proposed)
                ) + 10
            ]


            fig, ax = fplt.plot_individual_motor_trace(
                self.time,
                self.trace,
                self.ideal_trace,
                preview_xlim,
                preview_ylim,
                self.xtickspace,
                self.ytickspace,
                self.trace_id,
                lctrace_averaged=self.trace_avg,
                savefig=False,
                show=False,
                pause_intervals=self.pause_intervals,
            )

            ax.lines[-1].set_label(
                "Original ideal trace"
            )


            for j, dwell_i in enumerate(
                selected_dwells
            ):

                start, end = intervals[dwell_i]

                label = (
                    "Proposed KV refit"
                    if j == 0
                    else None
                )

                ax.plot(
                    self.time[start:end],
                    nideal_trace[start:end],
                    color="cyan",
                    linewidth=1.2,
                    label=label
                )


            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3
            )

            plt.show(block=True)


            user_input = hand.option_handler(
                "Satisfied (y/n) or quit without applying changes (q)? ",
                valid_values=["y", "n", "q"]
            )


            if user_input == "y":

                self.ideal_trace = nideal_trace

                user_input = hand.option_handler(
                    "Keep refitting segments (y/n)? ",
                    valid_values=["y", "n"]
                )

                if user_input == "y":
                    continue

                elif user_input == "n":
                    break


            elif user_input == "q":
                break


    def run(self):

        self._plot_trace()

        user_input = hand.option_handler(
            "Trim trace (tr), replace dwells with means (rd), "
            "merge dwells (md), refit segment (rf), "
            "pass (p), or return to previous (r)? ",
            valid_values=[
                "tr",
                "rd",
                "md",
                "rf",
                "p",
                "r"
            ]
        )

        if user_input == "tr":
            self.trim_trace()

        elif user_input == "rd":
            self.replace_dwells()

        elif user_input == "md":
            self.merge_dwells()

        elif user_input == "rf":
            self.refit_segment()

        elif user_input == "r":
            return("r")

        elif user_input == "p":
            return("c")


        user_input = hand.option_handler(
            "Keep working on the same trace (kw), return to previous (r) "
            "or continue to the next (c)? ",
            valid_values=["kw", "r", "c"]
        )

        return(user_input)


    def _merge_plateau_backward(
        self,
        trace,
        starts,
        ends,
        levels,
        i
    ):

        trace[
            starts[i]:ends[i]
        ] = levels[i - 1]


    def _merge_plateau_forward(
        self,
        trace,
        starts,
        ends,
        levels,
        i
    ):

        trace[
            starts[i]:ends[i]
        ] = levels[i + 1]


    def _trim_pause_intervals(self, mask):

        if self.pause_intervals is None:
            return None

        if len(self.pause_intervals) == 0:
            return np.empty(
                (0, 2),
                dtype=int
            )


        # Map old indices to the new indices after trimming.
        old_to_new = np.full(
            len(mask),
            -1,
            dtype=int
        )

        old_to_new[mask] = np.arange(
            np.sum(mask)
        )


        new_intervals = []

        for start, end in self.pause_intervals:

            # Original indices belonging to this pause.
            pause_indices = np.arange(
                start,
                end + 1
            )

            # Keep only pause samples that survived trimming.
            surviving = pause_indices[
                mask[pause_indices]
            ]

            if len(surviving) == 0:
                continue


            # If trimming cut through the middle of a pause,
            # split the remaining parts into separate intervals.
            split_points = (
                np.where(
                    np.diff(surviving) > 1
                )[0] + 1
            )

            surviving_groups = np.split(
                surviving,
                split_points
            )


            for group in surviving_groups:

                if len(group) == 0:
                    continue

                new_start = old_to_new[
                    group[0]
                ]

                new_end = old_to_new[
                    group[-1]
                ]

                new_intervals.append(
                    [new_start, new_end]
                )


        if len(new_intervals) == 0:
            return np.empty(
                (0, 2),
                dtype=int
            )

        return np.asarray(
            new_intervals,
            dtype=int
        )


    def _select_rectangle(
        self,
        title,
        xlim=None,
        ylim=None
    ):

        from matplotlib.ticker import (
            MultipleLocator,
            NullFormatter
        )

        previous_backend = plt.get_backend()

        selected = {
            "bounds": None,
            "xlim": None,
            "ylim": None
        }

        selector = None
        fig = None

        if xlim is None:
            xlim = self.xlim

        if ylim is None:
            ylim = self.ylim


        try:

            try:
                plt.switch_backend("QtAgg")

            except Exception:

                try:
                    plt.switch_backend(
                        "Qt5Agg"
                    )

                except Exception as e:

                    print(
                        "Could not switch to Qt backend."
                    )
                    print(
                        f"Current backend: {previous_backend}"
                    )
                    print(
                        f"Error: {e}"
                    )

                    return (
                        None,
                        list(xlim),
                        list(ylim)
                    )


            fig, ax = (
                fplt.plot_individual_motor_trace(
                    self.time,
                    self.trace,
                    self.ideal_trace,
                    xlim,
                    ylim,
                    self.xtickspace,
                    self.ytickspace,
                    self.trace_id,
                    lctrace_averaged=self.trace_avg,
                    savefig=False,
                    show=False,
                    pause_intervals=self.pause_intervals,
                )
            )

            ax.set_title(title)


            ax.xaxis.set_minor_locator(
                MultipleLocator(3)
            )

            ax.yaxis.set_minor_locator(
                MultipleLocator(3)
            )

            ax.xaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.yaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.tick_params(
                axis="both",
                which="minor",
                length=3
            )


            def onselect(
                eclick,
                erelease
            ):

                x1 = eclick.xdata
                x2 = erelease.xdata
                y1 = eclick.ydata
                y2 = erelease.ydata

                if (
                    x1 is None or
                    x2 is None or
                    y1 is None or
                    y2 is None
                ):

                    selected["bounds"] = None
                    return


                t1, t2 = sorted(
                    [x1, x2]
                )

                ymin, ymax = sorted(
                    [y1, y2]
                )

                selected["bounds"] = (
                    t1,
                    t2,
                    ymin,
                    ymax
                )


            def on_key(event):

                if event.key in [
                    "enter",
                    "return"
                ]:

                    selected["xlim"] = list(
                        ax.get_xlim()
                    )

                    selected["ylim"] = list(
                        ax.get_ylim()
                    )

                    plt.close(fig)


                elif event.key == "escape":

                    selected["bounds"] = None

                    selected["xlim"] = list(
                        ax.get_xlim()
                    )

                    selected["ylim"] = list(
                        ax.get_ylim()
                    )

                    plt.close(fig)


                elif event.key == "r":

                    ax.set_xlim(
                        self.xlim
                    )

                    ax.set_ylim(
                        self.ylim
                    )

                    selected["xlim"] = list(
                        self.xlim
                    )

                    selected["ylim"] = list(
                        self.ylim
                    )

                    fig.canvas.draw_idle()


            try:

                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True,
                    props={
                        "facecolor": "none",
                        "edgecolor": "black",
                        "linewidth": 1.0,
                        "alpha": 0.9,
                    },
                    handle_props={
                        "marker": "s",
                        "markersize": 2,
                        "markeredgewidth": 0.8,
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                    },
                    grab_range=5,
                )

            except TypeError:

                selector = RectangleSelector(
                    ax,
                    onselect,
                    useblit=True,
                    button=[1],
                    minspanx=0,
                    minspany=0,
                    spancoords="data",
                    interactive=True,
                )


            fig.canvas.mpl_connect(
                "key_press_event",
                on_key
            )

            plt.show(block=True)


            bounds = selected["bounds"]

            if selected["xlim"] is None:

                selected["xlim"] = list(
                    ax.get_xlim()
                )

            if selected["ylim"] is None:

                selected["ylim"] = list(
                    ax.get_ylim()
                )


        finally:

            try:

                if selector is not None:
                    selector.set_active(False)

            except Exception:
                pass


            try:

                if fig is not None:
                    plt.close(fig)

            except Exception:
                pass


            try:
                plt.switch_backend(
                    previous_backend
                )

            except Exception:

                try:

                    from IPython import get_ipython

                    ip = get_ipython()

                    if ip is not None:

                        ip.run_line_magic(
                            "matplotlib",
                            "inline"
                        )

                except Exception:
                    pass


        return(
            bounds,
            selected["xlim"],
            selected["ylim"]
        )


    def _plot_trace(self):

        fplt.plot_individual_motor_trace(
            self.time,
            self.trace,
            self.ideal_trace,
            self.xlim,
            self.ylim,
            self.xtickspace,
            self.ytickspace,
            self.trace_id,
            lctrace_averaged=self.trace_avg,
            savefig=False,
            pause_intervals=self.pause_intervals,
        )
#%% deprecated

def correct_spatial_drift_by_vm_old (fec_h5_file, fec_time, molext, pol_degree = 1):
    k1 = fec_h5_file.force1x.calibration[0].stiffness
    k2 = fec_h5_file.force2x.calibration[0].stiffness
    ds_trappos1x, distance1 = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file.distance1)
    ds_trappos1x, ds_force1x = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file.downsampled_force1x)
    ds_trappos1x, ds_force2x = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file.downsampled_force2x)
    #ds_trappos1x, ds_time = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file.distance1.seconds)
    
    ds_time = distance1.seconds
    ds_trappos1x = ds_trappos1x.data
    distance1 = distance1.data
    ds_force1x = ds_force1x.data
    ds_force2x = ds_force2x.data
    #ds_time = ds_time.data
    vm_diffx = (ds_force2x/k2 - ds_force1x/k1)/1000
    vm_trap_centers_sep = distance1+vm_diffx
    
    spatial_drift = vm_trap_centers_sep - ds_trappos1x
    spatial_drift = spatial_drift - spatial_drift[0]
    fit = np.polyfit(ds_time, spatial_drift,pol_degree)
    corrected_molext = molext + np.polyval(fit, fec_time)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Relative drift error (um)")
    plt.plot(ds_time,spatial_drift)
    plt.plot(ds_time,np.polyval(fit, ds_time))
    
    return(corrected_molext)

def find_start_leg (self, rip_idx):
    current_idx_pos = np.where(self.trans_idxs == rip_idx)[0][0]

    if current_idx_pos == 0:
        previous_rip_idx = 0
    else:
        previous_rip_idx = self.trans_idxs[current_idx_pos-1]

    # leg is at least navg data points after prev rip
    mirror_slice = self.trappos[previous_rip_idx+self.rip_params["padding"]:rip_idx+1].reshape(-1,1)
    force_slice = self.force[previous_rip_idx+self.rip_params["padding"]:rip_idx+1].reshape(-1,1)
    start_leg = np.hstack( (mirror_slice, force_slice) )

    if len(start_leg) > self.rip_params["nfit"]:
        start_leg = start_leg[len(start_leg) - self.rip_params["nfit"]:len(start_leg)]
        start_fit = np.polyfit(start_leg[:,0], start_leg[:,1],1)
        f1 = np.polyval(start_fit, start_leg[-1,0])
        x1 = (f1-start_fit[1])/start_fit[0]
        return(start_leg, start_fit, f1, x1)

    elif len(start_leg) > self.rip_params["navg"]:
        start_leg = start_leg[len(start_leg) - self.rip_params["navg"]:len(start_leg)]
        x1 = np.mean(start_leg[:,0])
        f1 = np.mean(start_leg[:,1])
        return(start_leg, np.nan, f1, x1)
    else:
        return(np.nan,np.nan,np.nan,np.nan)

def find_end_leg(self, end_rip_idx, ripF):
    end_rip_idx_pos = np.where(self.trans_idxs == end_rip_idx)[0][0]

    # sanity check just in case
    if end_rip_idx_pos >= len (self.trans_idxs):
        return(np.nan,np.nan,np.nan) # will be filled with nans
    else:
        if end_rip_idx_pos-1 < 0:
            previous_rip_idx = 0
        else:
            previous_rip_idx = self.trans_idxs[end_rip_idx_pos-1]

        # leg is at least navg data points after prev rip
        mirror_slice = self.trappos[previous_rip_idx+self.rip_params["padding"]:end_rip_idx+1].reshape(-1,1)
        force_slice = self.force[previous_rip_idx+self.rip_params["padding"]:end_rip_idx+1].reshape(-1,1)
        end_leg = np.hstack( (mirror_slice, force_slice) )

        if len(end_leg) == 0:
            return(np.nan,np.nan,np.nan)
        else:
            minF_idx = np.argmin(end_leg[:,1])
            end_leg = end_leg[minF_idx:len(end_leg)]

            ## filter using FFitRange
            mask = (end_leg[:,1] > ripF - self.rip_params["FFitRange"]) &\
                    (end_leg[:,1] < ripF + self.rip_params["FFitRange"])
            end_leg = end_leg[mask]

            if len(end_leg) > self.rip_params["nfit"]:
                tmpendfit = np.polyfit(end_leg[:,0], end_leg[:,1],1)
                tmpfitforces = np.polyval(tmpendfit, end_leg[:,0])
                cindx = np.argmin(abs(tmpfitforces - ripF))
                lb = cindx-int(self.rip_params["nfit"]/2)
                ub = cindx+1+int(self.rip_params["nfit"]/2)

                if lb < 0:
                    lb = 0
                end_leg = end_leg[lb:ub]
                end_fit = np.polyfit(end_leg[:,0], end_leg[:,1],1)
                x2 = (ripF-end_fit[1])/end_fit[0]
                return(end_leg, end_fit, x2)

            elif len(end_leg) > self.rip_params["navg"]:
                cindx = np.argmin(abs(end_leg[:,1] - ripF))
                if abs(end_leg[:,1][cindx] - ripF) < self.rip_params["FAvgRange"]:
                    lb = cindx-int(self.rip_params["navg"]/2)
                    ub = cindx+1+int(self.rip_params["navg"]/2)

                    if lb < 0:
                        lb = 0

                    end_leg = end_leg[lb:ub]
                    x2 = np.mean(end_leg[:,0])
                    return(end_leg, np.nan, x2)
                else:
                    return(np.nan,np.nan,np.nan)

            else:
                return(np.nan,np.nan,np.nan)
   
def recalculate_states_hmm(self):
    while True:
        if not isinstance(self.lc_trace, (list, np.ndarray)):
            print("No valid lc_trace given. Provide a valid lc_trace to allow calculation.")
            break
        else:
            print("Current states are " + str(self.states))
            hmmobj = fitting.HMM_fit(self.lc_trace, len(self.trans_idxs), len(self.trans_idxs), 
                                     filtering = False,
                                     annotate_transitions = False, 
                                     orind = False, 
                                     plotting = False)
            hmmobj.fit_HMM_model()
            
            nstates = hmmobj.states
            print(nstates)
            nstates[0] = self.states[0]
            ndLcs_sum = [x - self.basal_lc for x in nstates]
            self._plot_fec(nstates)
            
            user_input = hand.option_handler("Satisfied (y) or quit without applying changes (q)? ",
                                            valid_values=["y", "q"])

            if user_input == "y":
                self.states = nstates
                self.dLcs_sum = ndLcs_sum
                
            elif user_input == "q":
                break

#%% deprecated

def fill_gaps_original(self):
    # detect gaps
    gap_paths = []
    grandtotalLc = self.all_trans_info[self.total_direct_path]["dLc"]

    for i in self.all_trans_info.keys():
        if i != self.total_direct_path:
            if np.isnan(self.all_trans_info[i]["dLc"]):
                gap_paths.append(i)

    # if gap path = 1 AND grantotal is not empty then try using grandtotal
    if (len(gap_paths) == 1) and (not np.isnan(grandtotalLc)):
        self.fill_gap_with_tdLc(gap_paths[0])

    # Try to do indirect paths
    else:
        for i in gap_paths:
            succ = self.fill_with_indirect_paths(i)

            # if that fails then do reverse as last resource
            if not succ:
                rF, ext1, ext2, dExt, dLc = self.find_path_extension(i, forward = False)
                self.all_trans_info[i]["dLc"] = dLc

class EditTransitions_old:
    def __init__(self, fec, distance, force, HFd, HFf, trans_idxs, plot_bounds,
                 parameters, var_component=0, lctrace=None, curve_type="S",
                 states = None, WLCpars = []):

        # normalize external inputs
        self.dlf = np.asarray(distance)
        self.flf = np.asarray(force)
        self.trans_idxs = np.asarray(trans_idxs, dtype=int)   # initial normalization
        self.fec = fec
        self.lctrace = None if lctrace is None else np.asarray(lctrace)
        self.HFd = np.asarray(HFd)
        self.HFf = np.asarray(HFf)
        self.plot_bounds = plot_bounds
        self.var_component = var_component
        self.parameters = parameters
        self.curve_type = curve_type
        self.states = states
        self.WLCpars = WLCpars
        self.user_input = None

    # ---------- helpers ----------
    def _order_indices(self, arr):
        """Sort indices asc for 'S' and desc for 'R' using NumPy only."""
        arr = np.sort(arr)   # safe: arr always ndarray
        if self.curve_type == "R":
            arr = arr[::-1]
        return arr

    def _unique_concat(self, base, new):
        """Concatenate unique indices while preserving order in `new`."""
        mask = ~np.isin(new, base)
        if np.any(mask):
            return np.concatenate([base, new[mask]])
        return base

    # ---------- main interactions ----------
    def find_transition_in_slice(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            ind_slice, dlf_slice, flf_slice, lctrace_trim = self._slicer()

            if ind_slice is None:
                break
            
            if ind_slice.size > 0:
                plt.show()
                if self.parameters["rupture_force_detection"] == "peaks":
                    threshold = hand.is_valid_value("Enter a threshold value: ", float, min_value=0, max_value=None)
                    distance = hand.is_valid_value("Enter a distance drop value: ", int, min_value=1, max_value=None)
                    window = hand.is_valid_value("Enter a window value: ", int, min_value=1, max_value=None)

                    nidxs = find_transitions_peaks(ind_slice, flf_slice,
                                                   force_cutoff=0,
                                                   threshold=threshold,
                                                   distance=distance,
                                                   window=window,
                                                   add_rupture=False)
                    print(nidxs)
                else:
                    fixed_steps = hand.is_valid_value("Enter the number of steps: ", int, min_value=1)
                    _, nidxs = find_transitions_hmm(ind_slice, flf_slice, lctrace_trim,
                                                    force_cutoff=0,
                                                    max_states=fixed_steps,
                                                    force_states=True)

                if not isinstance(nidxs, (list, np.ndarray)) or len(nidxs) == 0:
                    print("No transitions found.")
                    user_input = hand.option_handler("Quit (q) or retry (r)? ", valid_values=["q", "r"])
                    if user_input == "q":
                        break
                    else:
                        continue

                nidxs = np.asarray(nidxs, dtype=int)   # normalize just once
                preview = self._unique_concat(self.trans_idxs, nidxs)
                self._plot_fec(preview, None)

                user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                                 valid_values=["y", "n", "q"])

                if user_input == "y":
                    self.trans_idxs = self._unique_concat(self.trans_idxs, nidxs)
                    self.trans_idxs = self._order_indices(self.trans_idxs)
                    break
                elif user_input == "q":
                    break
            else:
                print("Slice is empty. Define new window.")

    def reset(self):
        while True:
            if self.curve_type == "S":
                flf = self.flf
                ind = np.arange(len(flf), dtype=int)
            elif self.curve_type == "R":
                flf = self.flf[::-1]
                ind = np.arange(len(flf), dtype=int)[::-1]
            else:
                return "Invalid curve type."

            if self.parameters["rupture_force_detection"] == "peaks":
                nidxs = find_transitions_peaks(ind, flf,
                                               force_cutoff=self.parameters.get("force_cutoff", 1),
                                               threshold=self.parameters.get("threshold", 0.5),
                                               distance=self.parameters.get("distance", 5),
                                               window=self.parameters.get("window", 5),
                                               add_rupture=False)
            else:
                _, nidxs = find_transitions_hmm(ind, flf, self.lctrace,
                                                force_cutoff=self.parameters.get("force_cutoff", 2),
                                                max_states=self.parameters.get("max_states", 10),
                                                force_states=False,
                                                filtering=True)

            nidxs = np.asarray(nidxs, dtype=int)   # normalize here
            self._plot_fec(nidxs,None)
            user_input = hand.option_handler("Satisfied (y) or quit loop without applying changes (q)? ",
                                             valid_values=["y", "q"])

            if user_input == "y":
                self.trans_idxs = nidxs
                break
            elif user_input == "q":
                break

    def remove_transition(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            max_valid = len(self.trans_idxs) - 1
            indexes_to_remove = hand.is_valid_indexes(
                f"Enter the index(es) of the transition to remove (0-{max_valid}) separated by space: ",
                max_valid)

            ntransitions_indexes = np.delete(self.trans_idxs, indexes_to_remove)
            self._plot_fec(ntransitions_indexes, None)

            user_input = hand.option_handler("Keep removed states (y/n) or quit removing loop without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = ntransitions_indexes
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue removing (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break

    def modify_force_index(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            transition_index_index = hand.is_valid_value(
                f"Enter the position of the transition index to modify (0-{len(self.trans_idxs)-1}): ",
                int, min_value=0, max_value=len(self.trans_idxs)-1)

            ntransition_index = hand.is_valid_value(
                f"Enter the new transition index (0-{len(self.dlf)-1}): ",
                int, min_value=0, max_value=len(self.dlf)-1)

            ntransitions_indexes = self.trans_idxs.copy()
            ntransitions_indexes[transition_index_index] = int(ntransition_index)

            self._plot_fec(ntransitions_indexes,None)
            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = ntransitions_indexes
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break

    def manually_add_transition(self):
        while True:
            print("Current transition indexes are " + np.array2string(self.trans_idxs))
            ntransition_index = hand.is_valid_value(
                f"Enter the new transition index (0-{len(self.dlf)-1}): ",
                int, min_value=0, max_value=len(self.dlf)-1)

            preview = np.append(self.trans_idxs, int(ntransition_index))
            self._plot_fec(preview, None)

            user_input = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_input == "y":
                self.trans_idxs = np.append(self.trans_idxs, int(ntransition_index))
                self.trans_idxs = self._order_indices(self.trans_idxs)
                break
            elif user_input == "q":
                break

            user_input = hand.option_handler("Continue modifying (y/n)? ", valid_values=["y", "n"])
            if user_input == "n":
                break


    def run(self):
        print(self.trans_idxs)
        self._plot_fec(self.trans_idxs, self.states)
        user_input = hand.option_handler(
            "Calculate transitions in slice (cs), add transition manually (add), remove transition (rm), "
            "edit a transition (ed), reset (res), pass (p), or return to previous (r)? ",
            valid_values=["cs", "add", "rm", "ed", "res", "p", "r"]
        )

        if user_input == "cs":
            self.find_transition_in_slice()
        elif user_input == "add":
            self.manually_add_transition()
        elif user_input == "rm":
            self.remove_transition()
        elif user_input == "ed":
            self.modify_force_index()
        elif user_input == "res":
            self.reset()
        elif user_input == "p":
            return "c"
        elif user_input == "r":
            return "r"

        self.user_input = hand.option_handler(
            "Keep working on the same fec (kw), return to previous (r) or continue to the next (c)? ",
            valid_values=["kw", "r", "c"]
        )
        return self.user_input

    def _slicer(self):
        slice_bounds = hand.define_fd_bounds(self.dlf, self.flf)
        mask = (
            (self.dlf > slice_bounds[0][0]) & (self.dlf < slice_bounds[0][1]) &
            (self.flf > slice_bounds[1][0]) & (self.flf < slice_bounds[1][1])
        )

        ind = np.arange(len(self.flf), dtype=int)
        ind_slice = ind[mask]
        dlf_slice = self.dlf[mask]
        flf_slice = self.flf[mask]

        if self.parameters["rupture_force_detection"] == "hmm":
            lctrace_trim = self.lctrace[mask]
        else:
            lctrace_trim = None
        return ind_slice, dlf_slice, flf_slice, lctrace_trim

    def _plot_fec(self, transitions_indexes, states):
        transitions_indexes = np.asarray(transitions_indexes, dtype=int)  # normalize external arg
        fplt.plot_individual_fec(self.dlf,
                                 self.flf,
                                 self.fec,
                                 "black",
                                 self.plot_bounds[0],
                                 self.plot_bounds[1],
                                 "",
                                 *self.WLCpars,
                                 var_component = self.var_component,
                                 distances_hf=self.HFd,
                                 forces_hf=self.HFf,
                                 scatter=False,
                                 annot=False,
                                 savefig=False,
                                 rupture_index=False,
                                 lcstates=states,
                                 states_end_indexes=transitions_indexes,
                                 lc_trajectory=False)
