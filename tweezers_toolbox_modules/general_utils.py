import numpy as np
from scipy.signal import welch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import brentq
from lumicks.pylake.fitting.model import InverseModel
from lumicks.pylake.fitting.detail.derivative_manipulation import (
    invert_function as pylake_invert_function,
)
from scipy.integrate import cumulative_trapezoid

#%% basic math functions

def empirical_cdf(values):
    """
    Calculate the empirical CDF of any numerical variable.

    Returns
    -------
    x : ndarray
        Sorted unique observed values.
    cdf : ndarray
        Fraction of observations less than or equal to each x.
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No valid observations were provided.")

    x, counts = np.unique(values, return_counts=True)
    cdf = np.cumsum(counts) / values.size

    return x, cdf

def clip_signal(x, clip_q):

    center = np.median(
        x
    )

    threshold = np.quantile(
        np.abs(
            x - center
        ),
        clip_q
    )

    x = np.clip(
        x,
        center - threshold,
        center + threshold
    )
    return(x)
        
def inverse_transform_sample(x, pdf, n_samples=1, seed=None):
    """
    Generate random samples from an arbitrary PDF using
    inverse transform sampling.

    Parameters
    ----------
    x : array_like
        Grid over which the PDF is defined.

    pdf : array_like
        Probability density evaluated at x.

    n_samples : int
        Number of random samples.

    seed : int or None
        Random seed.

    Returns
    -------
    samples : ndarray
        Random samples drawn from the distribution.
    """

    x = np.asarray(x)
    pdf = np.asarray(pdf)

    # Construct CDF by numerical integration
    cdf = cumulative_trapezoid(pdf, x, initial=0)

    # Normalize so CDF goes from 0 -> 1
    cdf /= cdf[-1]

    # Uniform random numbers
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n_samples)

    # Numerically evaluate inverse CDF
    samples = np.interp(u, cdf, x)

    return samples

def fast_invert_function(
    distances,
    initial,
    f_min,
    f_max,
    model_function,
    derivative_function,
    tol=1e-8,
    max_iter=50,
):
    """
    Invert a monotonic model using vectorized safeguarded Newton iterations.

    All target distances are solved simultaneously. Points that cannot be
    solved safely are passed to Pylake's original inversion implementation.
    """
    distances = np.asarray(distances, dtype=float)
    scalar_input = distances.ndim == 0
    original_shape = distances.shape
    targets = distances.reshape(-1)

    if targets.size == 0:
        return np.empty(original_shape, dtype=float)

    # Newton requires the analytical derivative.
    if derivative_function is None:
        return pylake_invert_function(
            distances,
            initial,
            f_min,
            f_max,
            model_function,
            derivative_function,
            tol,
        )

    result = np.full(targets.size, np.nan, dtype=float)

    def evaluate(function, values):
        values = np.asarray(values, dtype=float)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
            under="ignore",
        ):
            output = np.asarray(function(values), dtype=float)

        if output.shape != values.shape:
            output = np.broadcast_to(output, values.shape)

        return np.asarray(output, dtype=float)

    # Avoid evaluating exactly at zero, where WLC derivatives may become
    # singular or numerically unstable.
    lower_bound = float(f_min)

    if not np.isfinite(lower_bound):
        return pylake_invert_function(
            distances,
            initial,
            f_min,
            f_max,
            model_function,
            derivative_function,
            tol,
        )

    lower_evaluation = max(
        lower_bound,
        1e-12,
    )

    if np.isfinite(f_max) and lower_evaluation >= f_max:
        return pylake_invert_function(
            distances,
            initial,
            f_min,
            f_max,
            model_function,
            derivative_function,
            tol,
        )

    lower = np.full(targets.size, lower_evaluation)
    lower_model = evaluate(model_function, lower)

    target_tolerance = tol * np.maximum(1.0, np.abs(targets))

    valid = np.isfinite(targets) & np.isfinite(lower_model)

    # If the target is at or below the model at the lower force bound, the
    # bounded solution is the lower bound.
    at_lower_bound = (
        valid
        & (targets <= lower_model + target_tolerance)
    )

    result[at_lower_bound] = lower_bound

    requires_solution = valid & ~at_lower_bound

    # Find an upper force bracket. Unlike Pylake, we do this for every data
    # point simultaneously.
    starting_force = max(
        float(initial),
        lower_evaluation * 2.0,
        1e-6,
    )

    if np.isfinite(f_max):
        starting_force = min(starting_force, float(f_max))

    upper = np.full(targets.size, starting_force)
    upper_model = evaluate(model_function, upper)

    for _ in range(40):
        needs_expansion = (
            requires_solution
            & (
                ~np.isfinite(upper_model)
                | (upper_model < targets)
            )
        )

        if not np.any(needs_expansion):
            break

        proposed_upper = np.where(
            upper > 0.0,
            upper * 2.0,
            1.0,
        )

        if np.isfinite(f_max):
            proposed_upper = np.minimum(
                proposed_upper,
                float(f_max),
            )

        expandable = needs_expansion & (proposed_upper > upper)

        if not np.any(expandable):
            break

        upper[expandable] = proposed_upper[expandable]
        upper_model = evaluate(model_function, upper)

    bracketed = (
        requires_solution
        & np.isfinite(upper_model)
        & (upper_model >= targets)
        & (upper > lower)
    )

    indices = np.flatnonzero(bracketed)

    if indices.size:
        target = targets[indices]
        lo = lower[indices].copy()
        hi = upper[indices].copy()
        y_lo = lower_model[indices]
        y_hi = upper_model[indices]

        # Interpolate within each bracket to obtain a better starting value.
        denominator = y_hi - y_lo

        fraction = np.divide(
            target - y_lo,
            denominator,
            out=np.full(target.size, 0.5),
            where=np.isfinite(denominator) & (denominator > 0.0),
        )

        fraction = np.clip(fraction, 0.0, 1.0)
        force = lo + fraction * (hi - lo)

        for _ in range(max_iter):
            model_value = evaluate(model_function, force)
            derivative = evaluate(derivative_function, force)
            residual = model_value - target

            residual_converged = (
                np.isfinite(residual)
                & (
                    np.abs(residual)
                    <= tol * np.maximum(1.0, np.abs(target))
                )
            )

            interval_converged = (
                np.abs(hi - lo)
                <= tol * np.maximum(1.0, np.abs(force))
            )

            converged = residual_converged | interval_converged

            if np.any(converged):
                result[indices[converged]] = force[converged]

            remaining = ~converged

            if not np.any(remaining):
                indices = np.empty(0, dtype=int)
                break

            indices = indices[remaining]
            target = target[remaining]
            lo = lo[remaining]
            hi = hi[remaining]
            force = force[remaining]
            derivative = derivative[remaining]
            residual = residual[remaining]

            # Maintain the brackets.
            below_target = residual < 0.0
            lo[below_target] = force[below_target]
            hi[~below_target] = force[~below_target]

            with np.errstate(divide="ignore", invalid="ignore"):
                newton_force = force - residual / derivative

            midpoint = 0.5 * (lo + hi)

            unsafe_newton = (
                ~np.isfinite(newton_force)
                | ~np.isfinite(derivative)
                | (derivative <= 0.0)
                | (newton_force <= lo)
                | (newton_force >= hi)
            )

            # Newton when safe; bisection otherwise.
            force = np.where(
                unsafe_newton,
                midpoint,
                newton_force,
            )

    # Use the original implementation only for unresolved points.
    unresolved = np.flatnonzero(~np.isfinite(result))

    for index in unresolved:
        result[index] = pylake_invert_function(
            float(targets[index]),
            initial,
            f_min,
            f_max,
            model_function,
            derivative_function,
            tol,
        )

    result = result.reshape(original_shape)

    if scalar_input:
        return float(result)

    return result

def solve_equation(
    lhs,
    rhs,
    x_range,
    grid_spacing=0.01,
):
    """
    Solve

        lhs(x) = rhs

    for all roots within x_range.

    lhs and rhs can be either functions or constants.
    """

    x_min, x_max = x_range

    if x_min >= x_max:
        raise ValueError("x_range must be (minimum, maximum).")

    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be greater than zero.")

    # Convert constants into functions
    if not callable(lhs):
        lhs_value = lhs
        lhs = lambda x: lhs_value + 0*x

    if not callable(rhs):
        rhs_value = rhs
        rhs = lambda x: rhs_value + 0*x

    # The solver creates the residual itself
    def residual(x):
        return lhs(x) - rhs(x)

    grid = np.arange(
        x_min,
        x_max + grid_spacing,
        grid_spacing,
    )

    grid = grid[grid <= x_max]

    if grid[-1] < x_max:
        grid = np.append(grid, x_max)

    values = residual(grid)
    roots = []

    # Roots exactly on grid points
    exact_root_indices = np.flatnonzero(
        np.isclose(values, 0.0, atol=1e-12, rtol=0.0)
    )

    roots.extend(grid[exact_root_indices])

    # Sign changes
    crossing_indices = np.flatnonzero(
        values[:-1] * values[1:] < 0
    )

    for i in crossing_indices:
        root = brentq(
            residual,
            grid[i],
            grid[i + 1],
        )

        roots.append(root)

    if not roots:
        return np.array([], dtype=float)

    roots = np.sort(np.asarray(roots))

    # Remove duplicates
    unique_roots = [roots[0]]

    for root in roots[1:]:
        if not np.isclose(
            root,
            unique_roots[-1],
            atol=1e-8,
            rtol=0.0,
        ):
            unique_roots.append(root)

    return np.asarray(unique_roots)

def generate_random_sample_gaussian (N, seed):
    pass

def get_KbT (T):
    kB = 0.01380649 # pN nm /K
    return kB*T

def get_complement_intervals(segments, trace_len, segment_inclusive=True):

    if segment_inclusive:
        # segments are [start, end], both inclusive
        starts = np.concatenate(([0], segments[:, 1] + 1))
        ends = np.concatenate((segments[:, 0] - 1, [trace_len - 1]))

        complement = np.column_stack((starts, ends))
        complement = complement[starts <= ends]

    else:
        # segments are [start, end)
        starts = np.concatenate(([0], segments[:, 1]))
        ends = np.concatenate((segments[:, 0], [trace_len]))

        complement = np.column_stack((starts, ends))
        complement = complement[starts < ends]

    return complement

def intervals_from_binary_labels(labels):
    labels = np.asarray(labels, dtype=bool)

    edges = np.diff(np.r_[False, labels, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    return list(zip(starts, ends))

def compute_signal_power_spectrum_welch(signal, sampling_frequency, nperseg=4096):
    f, P = welch(signal, fs=sampling_frequency, nperseg=nperseg)
    PdB = 10*np.log10(P)
    return(f, P, PdB)

def compute_power_gain (Pin, Pout, indB = False):
    if indB:
        power_gain = Pout-Pin
    else:
        power_gain = 10*np.log10(Pout/Pin)

    return(power_gain)

def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    gaussian_sd = 1.4826 * mad
    
    return med,mad, gaussian_sd

def running_mad(x, win):
    x = np.asarray(x)
    n = len(x)
    out = np.full(n, np.nan, float)
    half = win // 2
    for i in range(n):
        lo = max(0, i-half)
        hi = min(n, i+half+1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < max(10, win//5):
            continue
        m = np.median(seg)
        out[i] = np.median(np.abs(seg - m))
    return out

def running_std(x, win, ddof=1):
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan, float)
    half = win // 2

    for i in range(n):
        lo = max(0, i-half)
        hi = min(n, i+half+1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < max(10, win//5):
            continue
        out[i] = np.std(seg, ddof=ddof)
    return out


def tukey_inner_fence(data_array):
    '''
    Description
    -----------
    Computes the lower and upper bounds of Tukey's inner fence for outliers.
    Values outside the fence are considered potential outliers.


    Parameters
    ----------
    data_array : a numpy array of values

    Returns
    -------
    lower_bound : Lower bound of the fence
    upper_bound : Upper bound of the fence

    '''
    ## Computing the first and third quartiles of the data.
    quartile_1, quartile_3 = np.percentile(data_array, [25, 75])

    ## Calcultating the interquartile range.
    iqr = quartile_3 - quartile_1

    ## Computing the lower bound
    lower_bound = quartile_1 - (iqr * 1.5)

    ## Computing the upper bound
    upper_bound = quartile_3 + (iqr * 1.5)

    ## Return bounds
    return (quartile_1,lower_bound,upper_bound)

def chord_len(arclen,R=4.76):
    c = 2*R*np.sin(arclen/(2*R))
    return(c)

def get_hf_index(x, downsample_factor):
    hf_index = x*downsample_factor + downsample_factor
    return(hf_index)

def get_lf_index(x, downsample_factor):
    lf_index = int(x/downsample_factor) - downsample_factor
    return(lf_index)

def dbscan_on_scaled(
    X_scaled,
    *,
    min_samples=3,
    eps=None,           # if None -> auto from k-distance + KneeLocator
    plot_knee=True
):
    """
    Run DBSCAN on **already scaled** data.

    Parameters
    ----------
    X_scaled : array-like, shape (N, D)
        Scaled feature matrix (e.g. output of scale_columns(...)[1]).
    min_samples : int, default=3
        DBSCAN min_samples parameter (also used as k for k-distance).
    eps : float or None, default=None
        If None, choose eps from the k-distance curve using KneeLocator.
        If not None, use this value directly and skip knee detection.
    plot_knee : bool, default=True
        If True and eps is None, plot the k-distance curve and detected knee.

    Returns
    -------
    labels : ndarray, shape (N,)
        DBSCAN cluster labels (-1 = noise).
    eps_used : float
        Epsilon value actually used by DBSCAN.
    k_dist : ndarray
        Sorted k-distance values (only meaningful when eps is None).
    knee_index : int or None
        Index in k_dist used to choose eps (None if eps was provided).
    """

    X_scaled = np.asarray(X_scaled)
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(-1, 1)

    n_samples = X_scaled.shape[0]

    # --------------------------------------------------
    # If eps is given: skip k-distance / knee, just run DBSCAN
    # --------------------------------------------------
    if eps is not None:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        return labels, float(eps), None, None

    # --------------------------------------------------
    # Auto eps from k-distance + KneeLocator
    # --------------------------------------------------
    # k = min_samples for k-distance
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    # distance to k-th nearest neighbor
    k_dist = np.sort(distances[:, -1])

    # try KneeLocator
    knee_index = None
    try:
        ks = np.arange(n_samples)
        kneedle = KneeLocator(
            ks,
            k_dist,
            S=1.0,
            curve="convex",
            direction="increasing"
        )
        knee_index = kneedle.knee
    except Exception:
        knee_index = None

    # fallback: max curvature if KneeLocator fails or returns None
    if knee_index is None:
        d1 = np.gradient(k_dist)
        d2 = np.gradient(d1)
        knee_index = int(np.argmax(d2))

    eps_used = float(k_dist[knee_index])

    if plot_knee:
        plt.figure(figsize=(4, 3))
        plt.plot(k_dist, lw=1.5)
        plt.axvline(knee_index, linestyle='--')
        plt.axhline(eps_used, linestyle='--')
        plt.xlabel('Points (sorted)')
        plt.ylabel(f'{min_samples}-NN distance')
        plt.title(f'k-distance (eps ≈ {eps_used:.3f})')
        plt.tight_layout()
        plt.show()

    # run DBSCAN with chosen eps
    db = DBSCAN(eps=eps_used, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    return labels, eps_used, k_dist, knee_index

def kmeans_on_scaled(
    X_scaled,
    *,
    k_min=1,
    k_max=8,
    k_force=None,       # if not None → force this k and skip elbow
    plot_elbow=True,
    random_state=0
):
    """
    Run K-means on **already scaled** data.

    Parameters
    ----------
    X_scaled : array-like, shape (N, D)
        Scaled feature matrix (e.g. output of scale_columns(...)[1]).
    k_min : int, default=1
        Minimum number of clusters to try (used only if k_force is None).
    k_max : int, default=8
        Maximum number of clusters to try (used only if k_force is None).
    k_force : int or None, default=None
        If not None, force this number of clusters and skip elbow/KneeLocator.
    plot_elbow : bool, default=True
        If True and k_force is None, plot inertia (WCSS) vs k and mark chosen k.
    random_state : int, default=0
        Random state for KMeans reproducibility.

    Returns
    -------
    labels : ndarray, shape (N,)
        Cluster labels (0 ... k_opt-1).
    k_opt : int
        Chosen number of clusters (k_force if specified).
    inertias : dict
        Mapping {k: inertia}. If k_force is not None, dict has a single entry.
    centers : ndarray, shape (k_opt, D)
        Cluster centers in scaled feature space.
    """

    X_scaled = np.asarray(X_scaled)
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(-1, 1)

    n_samples = X_scaled.shape[0]

    # sanity on k range
    if k_min < 1:
        k_min = 1
    if k_max > n_samples:
        k_max = n_samples

    # --------------------------------------------------
    # MODE 1: force k, skip elbow
    # --------------------------------------------------
    if k_force is not None:
        k_opt = int(k_force)
        if k_opt < 1 or k_opt > n_samples:
            raise ValueError(f"k_force must be between 1 and {n_samples}, got {k_opt}")

        km = KMeans(
            n_clusters=k_opt,
            n_init=20,
            random_state=random_state
        )
        km.fit(X_scaled)
        labels = km.labels_
        centers = km.cluster_centers_
        inertias_dict = {k_opt: float(km.inertia_)}

        # no elbow plot in forced mode
        return labels, k_opt, inertias_dict, centers

    # --------------------------------------------------
    # MODE 2: use elbow/KneeLocator over [k_min, k_max]
    # --------------------------------------------------
    ks = np.arange(k_min, k_max + 1)
    inertias = []
    models = {}

    for k in ks:
        km = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state
        )
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        models[k] = km

    inertias = np.array(inertias)

    # ----- find elbow with KneeLocator -----
    k_opt = None
    try:
        kneedle = KneeLocator(
            ks,
            inertias,
            S=1.0,
            curve="convex",
            direction="decreasing"
        )
        k_opt = kneedle.elbow if kneedle.elbow is not None else kneedle.knee
    except Exception:
        k_opt = None

    # ----- fallback: max curvature on inertia curve -----
    if k_opt is None:
        d1 = np.gradient(inertias)
        d2 = np.gradient(d1)
        idx = int(np.argmax(d2))
        k_opt = int(ks[idx])

    # safety: ensure k_opt in range
    if k_opt < k_min or k_opt > k_max:
        k_opt = max(k_min, min(2, k_max))

    best_model = models[k_opt]
    labels = best_model.labels_
    centers = best_model.cluster_centers_

    # ----- optional elbow plot -----
    if plot_elbow:
        plt.figure(figsize=(4, 3))
        plt.plot(ks, inertias, 'o-', lw=2)
        plt.axvline(k_opt, linestyle='--')
        plt.title(f'Elbow method (k* = {k_opt})')
        plt.xlabel('k')
        plt.ylabel('Inertia (WCSS)')
        plt.tight_layout()
        plt.show()

    inertias_dict = {int(k): float(i) for k, i in zip(ks, inertias)}

    return labels, int(k_opt), inertias_dict, centers
#%% 
class FastInverseModel(InverseModel):
    """
    Pylake-compatible inverse model using vectorized inversion.

    The most recent inversion is cached because Pylake commonly calculates
    the model and analytical Jacobian consecutively using the same data and
    parameter values.
    """

    def __init__(
        self,
        model,
        independent_min=0.0,
        independent_max=np.inf,
        interpolate=False,
    ):
        super().__init__(
            model,
            independent_min=independent_min,
            independent_max=independent_max,
            interpolate=interpolate,
        )

        self._cached_independent = None
        self._cached_parameters = None
        self._cached_result = None

    def _cache_matches(self, independent, param_vector):
        if self._cached_result is None:
            return False

        return (
            np.array_equal(
                np.asarray(independent),
                self._cached_independent,
            )
            and np.array_equal(
                np.asarray(param_vector),
                self._cached_parameters,
            )
        )

    @staticmethod
    def _copy_result(result):
        if np.ndim(result) == 0:
            return float(result)

        return np.array(result, copy=True)

    def _raw_call(self, independent, param_vector):
        if self._cache_matches(independent, param_vector):
            return self._copy_result(self._cached_result)

        if self.interpolate:
            result = super()._raw_call(
                independent,
                param_vector,
            )
        else:
            result = fast_invert_function(
                distances=independent,
                initial=1.0,
                f_min=self.independent_min,
                f_max=self.independent_max,
                model_function=lambda trial: self.model._raw_call(
                    trial,
                    param_vector,
                ),
                derivative_function=lambda trial: self.model.derivative(
                    trial,
                    param_vector,
                ),
                tol=1e-8,
            )

        self._cached_independent = np.array(
            independent,
            copy=True,
        )
        self._cached_parameters = np.array(
            param_vector,
            copy=True,
        )
        self._cached_result = self._copy_result(result)

        return self._copy_result(result)