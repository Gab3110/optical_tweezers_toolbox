from sfHMM.motor import sfHMM1Motor
import numpy as np
import sys
import matplotlib.pyplot as plt
import lumicks.pylake as lk
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from sfHMM import sfHMM1
from statsmodels.tools.numdiff import approx_hess
from scipy.optimize import minimize

#%%
sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.models import GenWLCModel, get_loading_rate, bell_rate_model, CHS_rate_model
from tweezers_toolbox_modules.models import rupture_force_pdf_cdf_ccdf
from tweezers_toolbox_modules.general_utils import mad, get_KbT
from tweezers_toolbox_modules.general_utils import FastInverseModel

#%% DEFINING MODULE LEVEL FUNCTIONS

def robust_slope_huber(x, y, *, c=1.345, max_iter=50, tol=1e-8, min_pts=20, return_details=False):
    """
    Robust slope (and intercept) via IRLS with Huber loss.
    Returns m, b (y ≈ m*x + b).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]
    n = x.size
    if n < min_pts:
        if return_details:
            return np.nan, np.nan, {"ok": False, "n": n}
        return np.nan, np.nan

    # Initial OLS
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [m, b]
    m, b = beta

    # IRLS
    for _ in range(max_iter):
        r = y - (m * x + b)

        _,_,s = mad(r)
        if not np.isfinite(s) or s == 0:
            # fall back: if residual scale is degenerate, we're basically done
            break

        u = r / (s * c)
        w = np.ones_like(u)
        big = np.abs(u) > 1
        w[big] = 1 / np.abs(u[big])  # Huber weights

        # Weighted least squares
        # Solve (X^T W X) beta = X^T W y
        WX0 = w * X[:, 0]
        WX1 = w * X[:, 1]
        XtWX = np.array([
            [np.dot(X[:, 0], WX0), np.dot(X[:, 0], WX1)],
            [np.dot(X[:, 1], WX0), np.dot(X[:, 1], WX1)],
        ])
        XtWy = np.array([
            np.dot(WX0, y),
            np.dot(WX1, y),
        ])

        beta_new = np.linalg.solve(XtWX, XtWy)
        m_new, b_new = float(beta_new[0]), float(beta_new[1])

        if max(abs(m_new - m), abs(b_new - b)) < tol * max(1.0, abs(m), abs(b)):
            m, b = m_new, b_new
            break
        m, b = m_new, b_new

    if return_details:
        r = y - (m * x + b)
        _,s = mad(r)
        # recompute final weights for diagnostics
        if np.isfinite(s) and s > 0:
            u = r / (s * c)
            w = np.ones_like(u)
            big = np.abs(u) > 1
            w[big] = 1 / np.abs(u[big])
        else:
            w = np.ones_like(y)
        return m, b, {"ok": True, "n": n, "mad_resid": float(s) if np.isfinite(s) else np.nan, "weights": w}
    return [m, b]


def test_homoc_res (des_mat, residuals):
    from statsmodels.stats.diagnostic import het_breuschpagan

    p = het_breuschpagan(residuals, des_mat)[1]

    if p < 0.05:
        return (1)
    else:
        return(0)

def create_design_matrix (expvar, pol_degree):

    expvar = expvar[:,np.newaxis]
    polynomial_features = PolynomialFeatures(degree=pol_degree)
    expvar = polynomial_features.fit_transform(expvar)

    return(expvar)

def OLS_fit (expvar, respvar, pol_degree):

    expvar = create_design_matrix (expvar, pol_degree)
    respvar = respvar[:,np.newaxis]

    model = sm.OLS(respvar, expvar)
    fit = model.fit()
    influence = fit.get_influence()

    #pred_resp = fit.predict(expvar)
    stud_res = influence.resid_studentized_internal

    return(fit, stud_res)

def robust_polyfit(x, y, deg=1, epsilon=1.35):
    """
    Robust polynomial fitting using HuberRegressor.
    Returns coefficients in the same order as np.polyfit.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),
                          HuberRegressor(epsilon=epsilon))
    model.fit(x, y)

    huber = model.named_steps['huberregressor']
    coefs = huber.coef_          # ascending order (x, x^2, ...)
    intercept = huber.intercept_

    # Rebuild to np.polyfit style: highest power first
    coeffs = np.concatenate(([coefs[i] for i in range(deg-1, -1, -1)], [intercept]))

    return coeffs

#%% DEFINING FITTING CLASSES

class eWLC_fit:
    def __init__(self, 
                 distance, 
                 force, 
                 guesses,
                 fitting_bounds = None,
                 fixed_parameter_mask = None,
                 parameters_bounds = None,
                 distance_offset = False,
                 force_offset = False,
                 distance_offset_bounds = None,
                 force_offset_bounds = None,
                 global_fit = False, 
                 invert = True,
                 inversion_method = "pylake"):
        
        self.distance = distance
        self.force = force
        self.fitting_bounds = fitting_bounds
        self.guesses = guesses
        self.fixed_parameter_mask = fixed_parameter_mask
        self.parameters_bounds = parameters_bounds
        self.distance_offset = distance_offset
        self.force_offset = force_offset
        self.distance_offset_bounds = distance_offset_bounds
        self.force_offset_bounds = force_offset_bounds
        self.global_fit = global_fit
        self.invert = invert
        self.inversion_method = inversion_method
        
        self.dfit = None
        self.ffit = None
        self.WLCmodel = None
        self.fitobj = None

    def prepare_data (self):
        if self.fitting_bounds:
            if not self.global_fit:
                mask = (self.distance > self.fitting_bounds[0][0]) & (self.distance < self.fitting_bounds[0][1]) & (self.force > self.fitting_bounds[1][0]) & (self.force < self.fitting_bounds[1][1])
                self.dfit = self.distance[mask]
                self.ffit = self.force[mask]
            else:
                self.dfit = []
                self.ffit = []
    
                for d,f in zip(self.distance, self.force):
                    mask = (d > self.fitting_bounds[0][0]) & (d < self.fitting_bounds[0][1]) & (f > self.fitting_bounds[1][0]) & (f < self.fitting_bounds[1][1])
    
                    self.dfit.append(d[mask])
                    self.ffit.append(f[mask])

        else:
            self.dfit = self.distance
            self.ffit = self.force
        

    def create_fitobj(self):
        self.WLCmodel = GenWLCModel(*self.guesses,
                                    fixed_parameter_mask = self.fixed_parameter_mask ,
                                    distance=self.dfit, 
                                    force=self.ffit,
                                    distance_offset=self.distance_offset, 
                                    force_offset=self.force_offset,
                                    eWLC_approx="marko", 
                                    global_fit_obj=self.global_fit)
        self.WLCmodel.gen_model()

        if self.invert:
            if self.inversion_method == "pylake":
                self.WLCmodel.model = self.WLCmodel.model.invert()
        
            elif self.inversion_method == "fast":
                self.WLCmodel.model = FastInverseModel(
                    self.WLCmodel.model
                )
        
            else:
                raise ValueError(
                    "inversion_method must be 'pylake' or 'fast'"
                )
        self.WLCmodel.gen_fitobj()

    def check_parameters_bounds(self):
        """Validate self.parameters_bounds against self.guesses with positivity checks."""
        pb = self.parameters_bounds
        if pb is None:
            pass  # nothing to check

        if len(pb) != len(self.guesses):
            raise ValueError(
                f"parameters_bounds has {len(pb)} components, but guesses has {len(self.guesses)}"
            )

        for ci, (g_comp, b_comp) in enumerate(zip(self.guesses, pb)):
            if len(b_comp) != len(g_comp):
                raise ValueError(
                    f"Component {ci}: bounds length {len(b_comp)} != guesses length {len(g_comp)}"
                )

            for pi, (g_val, (lb, ub)) in enumerate(zip(g_comp, b_comp)):
                # type check
                for name, v in (("lb", lb), ("ub", ub), ("guess", g_val)):
                    if not isinstance(v, (int, float)):
                        raise ValueError(
                            f"Component {ci}, param {pi}: {name} must be int or float, got {type(v)}"
                        )
                    if v <= 0:
                        raise ValueError(
                            f"Component {ci}, param {pi}: {name} must be positive, got {v}"
                        )

                # ordering
                if lb >= ub:
                    raise ValueError(
                        f"Component {ci}, param {pi}: require lb < ub (got {lb}, {ub})"
                    )

                # guess inside bounds
                if not (lb <= g_val <= ub):
                    raise ValueError(
                        f"Component {ci}, param {pi}: guess {g_val} not within [{lb}, {ub}]"
                    )
                    
    def set_fit_bounds (self):
        # Define Lp, Lc, St bounds
        for c, bounds in enumerate(self.parameters_bounds):
            if len(bounds) == 2:
                self.WLCmodel.fitobj[f"component_{c}/Lp"].lower_bound = bounds[0][0]
                self.WLCmodel.fitobj[f"component_{c}/Lp"].upper_bound = bounds[0][1]
                self.WLCmodel.fitobj[f"component_{c}/Lc"].lower_bound = bounds[1][0]
                self.WLCmodel.fitobj[f"component_{c}/Lc"].upper_bound = bounds[1][1]

            elif len(bounds) == 3:
                self.WLCmodel.fitobj[f"component_{c}/Lp"].lower_bound = bounds[0][0]
                self.WLCmodel.fitobj[f"component_{c}/Lp"].upper_bound = bounds[0][1]
                self.WLCmodel.fitobj[f"component_{c}/Lc"].lower_bound = bounds[1][0]
                self.WLCmodel.fitobj[f"component_{c}/Lc"].upper_bound = bounds[1][1]
                self.WLCmodel.fitobj[f"component_{c}/St"].lower_bound = bounds[2][0]
                self.WLCmodel.fitobj[f"component_{c}/St"].upper_bound = bounds[2][1]
            else:
                raise ValueError("Please provide valid parameter lists.")
        
        if self.force_offset:
            if self.force_offset_bounds:
                self.WLCmodel.fitobj["component_0/f_offset"].lower_bound = self.force_offset_bounds[0]
                self.WLCmodel.fitobj["component_0/f_offset"].upper_bound = self.force_offset_bounds[1]
                
                if self.global_fit:
                    for i in range(len(self.dfit)):
                        if i != 0:
                            self.WLCmodel.fitobj[f"component_0/f_offset_{i}"].lower_bound = self.force_offset_bounds[0]
                            self.WLCmodel.fitobj[f"component_0/f_offset_{i}"].upper_bound = self.force_offset_bounds[1]
                            
        if self.distance_offset:
            if self.distance_offset_bounds:
                self.WLCmodel.fitobj["component_0/d_offset"].lower_bound = self.distance_offset_bounds[0]
                self.WLCmodel.fitobj["component_0/d_offset"].upper_bound = self.distance_offset_bounds[1]
                
                if self.global_fit:
                    for i in range(len(self.dfit)):
                        if i != 0:
                            self.WLCmodel.fitobj[f"component_0/d_offset_{i}"].lower_bound = self.distance_offset_bounds[0]
                            self.WLCmodel.fitobj[f"component_0/d_offset_{i}"].upper_bound = self.distance_offset_bounds[1]
                    
    def fit(self):
        self.prepare_data()
        self.create_fitobj()
        if self.parameters_bounds:
            self.check_parameters_bounds()
            self.set_fit_bounds()
        self.WLCmodel.fitobj.fit()
        self.fitobj = self.WLCmodel.fitobj

class PerPointLc:
    def __init__(self, d, f, *WLCp, var_lc_component=0,
                 lc_bounds = [0.001, 2], distance_offset = False,
                 force_offset = False, approx = "marko"):
        self.d = d
        self.f = f
        self.WLCp = WLCp
        self.lc_bounds = lc_bounds
        self.trace = None
        self.WLCmodel = None
        self.var_lc_component = var_lc_component
        self.distance_offset = distance_offset
        self.force_offset = force_offset
        self.approx = approx
        self.fixed_parameter_mask = None
    
    def gen_mask(self):
        fixed_parameter_mask = []
        for c, pars in enumerate(self.WLCp):
            bool_component_c = []
            for i in pars:
                bool_component_c.append(True)
            if c == self.var_lc_component:
                bool_component_c[1] = False
            fixed_parameter_mask.append(bool_component_c)
        self.fixed_parameter_mask = fixed_parameter_mask

    def construct_fitobj(self):
        self.WLCmodel = GenWLCModel(*self.WLCp,             
                                    fixed_parameter_mask = self.fixed_parameter_mask,
                                    distance=self.d, 
                                    force=self.f,
                                    distance_offset=False, 
                                    force_offset=True,
                                    eWLC_approx="marko", 
                                    global_fit_obj=False)
        self.WLCmodel.gen_model() # model generated has d as dependent and force as independent
        self.WLCmodel.model = self.WLCmodel.model.invert() # invert so now force is dependent.
        self.WLCmodel.gen_fitobj()
        
        # Define bounds
        self.WLCmodel.fitobj[f"component_{self.var_lc_component}/Lc"].lower_bound = self.lc_bounds[0]
        self.WLCmodel.fitobj[f"component_{self.var_lc_component}/Lc"].upper_bound = self.lc_bounds[1] 
        
    def find_lc_trace(self):
        self.WLCmodel.fitobj.fit()
        # if d as dependent (f independent), force goes before than distance.
        # In the opposite case (when the model is inverted), distance goes first (as tutorial).
        # Inverting the model makes this function really slow! Not inverting and 
        # swapping f and d is much faster. However, I will follow the tutorial. 
        # You can change if you like.
        self.trace = lk.parameter_trace(self.WLCmodel.model, self.WLCmodel.fitobj[""],
                                        f"component_{self.var_lc_component}/Lc", self.d, self.f)
    def run(self):
        self.gen_mask()
        self.construct_fitobj()
        self.find_lc_trace()

## Class to do HMM fit
class HMM_fit:
    def __init__(self, trace, max_states, method="base", forced_states = False, filtering = False,
                 annotate_transitions = False, orind = False, plotting = False):
        self.trace = trace
        self.max_states = max_states
        self.forced_states = forced_states
        self.filtering = filtering
        self.annotate_transitions = annotate_transitions
        self.predicted_trace_model = None
        self.predicted_trace = None
        self.states = None
        self.transitions_indexes = None
        self.orind = orind
        self.plotting = plotting
        self.method = method

    def fit_HMM_model(self):

        #from sfHMM import motor_sampling
        if self.forced_states:
            optimal_state_number = self.max_states
        else:
            optimal_state_number = (1,self.max_states)

        # Find the states using HMM optimized for motors
        if self.method == "base":
            self.predicted_trace_model = sfHMM1(self.trace, krange = optimal_state_number)
        else:
            self.predicted_trace_model = sfHMM1Motor(self.trace, krange = optimal_state_number)

        self.predicted_trace_model.step_finding()
        self.predicted_trace_model.denoising()
        self.predicted_trace_model.gmmfit(n_init=3, method = "bic")        
        self.predicted_trace_model.hmmfit()
        self.predicted_trace = self.predicted_trace_model.viterbi
        self.states = list(set(self.predicted_trace))
        self.states.sort()

    def curate_states(self, min_state_residence = 3):
        self.states = [x for x in self.states if len(np.where(self.predicted_trace == x)[0]) > min_state_residence]
        self.states.sort()

    def find_transitions (self):
        self.transitions_indexes = []

        for i in self.states:
            index = np.where(self.predicted_trace == i)[0]
            index = index[-1] - 1
            self.transitions_indexes.append(index)

        if isinstance(self.orind, np.ndarray):
            or_trans_indexes = []
            for i in self.transitions_indexes:
                or_trans_indexes.append(self.orind[i])

            self.transitions_indexes = or_trans_indexes
            self.transitions_indexes.sort()

    def plot(self):
        plt.plot(self.trace)
        plt.plot(self.predicted_trace)
        plt.show()
        plt.close()

    def fit (self):
        self.fit_HMM_model()

        if self.filtering:
            self.curate_states()

        if self.annotate_transitions:
            self.find_transitions()

        if self.plotting:
            self.plot()

import warnings

# ============================================================
# Initial-guess helpers
# ============================================================

def guess_x_dagger(sigma_F, T=298.15):
    kBT = get_KbT(T)

    return (
        np.pi / np.sqrt(6)
        * kBT / sigma_F
    )


def guess_k0(Fstar, Fdot_at_Fstar, x_dagger, T=298.15):
    kBT = get_KbT(T)

    return (
        Fdot_at_Fstar
        * x_dagger
        / kBT
        * np.exp(
            -Fstar * x_dagger / kBT
        )
    )


def guess_dG_CHS(Fmax, x_dagger, T=298.15, nu=0.5):
    kBT = get_KbT(T)

    dG_min = (
        nu
        * Fmax
        * x_dagger
        / kBT
    )

    return 2 * dG_min


# ============================================================
# Rupture-force MLE
# ============================================================

class FitRuptureForceDist:

    def __init__(
        self,
        rupture_forces,
        trap_keqs,
        pulling_speed,
        eWLC_pars,
        k0_guess=None,
        x_dagger_guess=None,
        dG_dagger_guess=None,
        T=298.15,
        n_grid=10000,
        rate_model="bell",
        force_dependent_loading_rates=True,
        fixed_loading_rate=None
    ):

        self.rupture_forces = np.asarray(
            rupture_forces,
            dtype=float
        )

        self.trap_keqs = trap_keqs
        self.pulling_speed = pulling_speed
        self.eWLC_pars = eWLC_pars

        self.k0_guess = k0_guess
        self.x_dagger_guess = x_dagger_guess
        self.dG_dagger_guess = dG_dagger_guess

        self.T = T
        self.n_grid = n_grid
        self.rate_model = rate_model

        self.force_dependent_loading_rates = (
            force_dependent_loading_rates
        )

        self.fixed_loading_rate = fixed_loading_rate

        self.Fgrid = None
        self.Fdots_F = None

        self.Fstar_guess = None
        self.sigma_F = None
        self.Fdot_at_Fstar = None

        self.result = None

        self.H = None
        self.H_log = None
        self.C = None
        self.errors = None
        self.correlation_matrix = None
        self.error_method = None

        if self.force_dependent_loading_rates:

            if len(self.trap_keqs) != len(self.rupture_forces):

                raise ValueError(
                    "trap_keqs and rupture_forces "
                    "must have the same length"
                )

        else:

            if self.fixed_loading_rate is None:

                raise ValueError(
                    "fixed_loading_rate must be provided when "
                    "force_dependent_loading_rates=False"
                )


    # ========================================================
    # Force grid
    # ========================================================

    def initialize_grid(self):

        self.Fgrid = np.linspace(
            0,
            np.max(self.rupture_forces),
            self.n_grid
        )

    # ========================================================
    # Loading rates
    # ========================================================

    def compute_force_dependent_loading_rates_per_trace(self):

        self.Fdots_F = get_loading_rate(
            self.Fgrid,
            self.pulling_speed,
            self.trap_keqs,
            *self.eWLC_pars
        )


    def get_Fdot_at_force(self, F):

        if not self.force_dependent_loading_rates:
            return self.fixed_loading_rate

        Fdots = [
            np.interp(
                F,
                self.Fgrid,
                Fdot_trace
            )
            for Fdot_trace in self.Fdots_F
        ]

        return np.mean(Fdots)


    # ========================================================
    # Initial guesses
    # ========================================================

    def estimate_force_distribution(self):

        self.Fstar_guess = np.mean(
            self.rupture_forces
        )

        self.sigma_F = np.std(
            self.rupture_forces,
            ddof=1
        )

        self.Fdot_at_Fstar = self.get_Fdot_at_force(
            self.Fstar_guess
        )


    def estimate_x_dagger_guess(self):

        if self.x_dagger_guess is None:

            self.x_dagger_guess = guess_x_dagger(
                self.sigma_F,
                self.T
            )


    def estimate_k0_guess(self):

        if self.k0_guess is None:

            self.k0_guess = guess_k0(
                self.Fstar_guess,
                self.Fdot_at_Fstar,
                self.x_dagger_guess,
                self.T
            )


    def estimate_dG_guess(self):

        if (
            self.rate_model == "CHS"
            and self.dG_dagger_guess is None
        ):

            self.dG_dagger_guess = guess_dG_CHS(
                np.max(self.rupture_forces),
                self.x_dagger_guess,
                self.T
            )


    def print_initial_guesses(self):

        print("\nInitial parameter guesses:")
        print("F* =", self.Fstar_guess, "pN")
        print("sigma_F =", self.sigma_F, "pN")
        print(
            "mean Fdot(F*) =",
            self.Fdot_at_Fstar,
            "pN/s"
        )
        print("k0 =", self.k0_guess, "1/s")
        print(
            "x_dagger =",
            self.x_dagger_guess,
            "nm"
        )

        if self.rate_model == "CHS":

            print(
                "dG_dagger =",
                self.dG_dagger_guess,
                "kBT"
            )


    def estimate_initial_guesses(self):

        self.estimate_force_distribution()
        self.estimate_x_dagger_guess()
        self.estimate_k0_guess()
        self.estimate_dG_guess()
        self.print_initial_guesses()


    # ========================================================
    # Force-dependent rate
    # ========================================================

    def compute_k_F(
        self,
        k0,
        x_dagger,
        dG_dagger=None
    ):

        if self.rate_model == "bell":

            return bell_rate_model(
                self.Fgrid,
                k0,
                x_dagger,
                T=self.T
            )

        elif self.rate_model == "CHS":

            return CHS_rate_model(
                self.Fgrid,
                k0,
                x_dagger,
                dG_dagger,
                T=self.T
            )

        else:

            raise ValueError(
                "Select a valid rate model (bell/CHS)"
            )


    # ========================================================
    # Rupture-force PDF
    # ========================================================

    def compute_p_F(self, kF):

        if self.force_dependent_loading_rates:
            Fdot = self.Fdots_F

        else:
            Fdot = self.fixed_loading_rate

        return rupture_force_pdf_cdf_ccdf(
            self.Fgrid,
            kF,
            Fdot
        )


    # ========================================================
    # Log likelihood
    # ========================================================

    def compute_log_likelihood(
        self,
        k0,
        x_dagger,
        dG_dagger=None
    ):

        kF = self.compute_k_F(
            k0,
            x_dagger,
            dG_dagger
        )

        p_F, _, _ = self.compute_p_F(kF)

        if self.force_dependent_loading_rates:

            p_Frupture = []

            for F_rupture, p_F_trace in zip(
                self.rupture_forces,
                p_F
            ):

                p = np.interp(
                    F_rupture,
                    self.Fgrid,
                    p_F_trace
                )

                p_Frupture.append(p)

            p_Frupture = np.asarray(
                p_Frupture
            )

        else:

            p_Frupture = np.interp(
                self.rupture_forces,
                self.Fgrid,
                p_F
            )

        return np.sum(
            np.log(p_Frupture)
        )


    # ========================================================
    # MLE
    # ========================================================

    def fit(self):

        if self.rate_model == "bell":

            initial_guess = [
                self.k0_guess,
                self.x_dagger_guess
            ]

            self.result = minimize(
                lambda alpha:
                    -self.compute_log_likelihood(
                        k0=alpha[0],
                        x_dagger=alpha[1]
                    ),
                initial_guess,
                method="Nelder-Mead",
                options={
                    "maxiter": 500,
                    "maxfev": 500
                }
            )

        elif self.rate_model == "CHS":

            initial_guess = [
                self.k0_guess,
                self.x_dagger_guess,
                self.dG_dagger_guess
            ]

            self.result = minimize(
                lambda alpha:
                    -self.compute_log_likelihood(
                        k0=alpha[0],
                        x_dagger=alpha[1],
                        dG_dagger=alpha[2]
                    ),
                initial_guess,
                method="Nelder-Mead",
                options={
                    "maxiter": 500,
                    "maxfev": 500
                }
            )

        else:

            raise ValueError(
                "Select a valid rate model (bell/CHS)"
            )

        return self.result


    # ========================================================
    # Ordinary Hessian
    # ========================================================

    def _estimate_hessian(self):

        if self.rate_model == "bell":

            self.H = approx_hess(
                self.result.x,
                lambda alpha:
                    -self.compute_log_likelihood(
                        k0=alpha[0],
                        x_dagger=alpha[1]
                    )
            )

        elif self.rate_model == "CHS":

            self.H = approx_hess(
                self.result.x,
                lambda alpha:
                    -self.compute_log_likelihood(
                        k0=alpha[0],
                        x_dagger=alpha[1],
                        dG_dagger=alpha[2]
                    )
            )

        else:

            raise ValueError(
                "Select a valid rate model (bell/CHS)"
            )


    # ========================================================
    # Log-space Hessian fallback
    # ========================================================

    def _estimate_hessian_log_bell(self):

        k0_fit, x_dagger_fit = self.result.x

        beta_fit = np.array([
            np.log(k0_fit),
            x_dagger_fit
        ])

        def nll(beta):

            return -self.compute_log_likelihood(
                k0=np.exp(beta[0]),
                x_dagger=beta[1]
            )

        self.H_log = approx_hess(
            beta_fit,
            nll
        )

        C_log = np.linalg.inv(
            self.H_log
        )

        # Transform back to [k0, x_dagger]
        J = np.array([
            [k0_fit, 0],
            [0, 1]
        ])

        self.C = (
            J
            @ C_log
            @ J.T
        )


    def _estimate_hessian_log_CHS(self):

        (
            k0_fit,
            x_dagger_fit,
            dG_dagger_fit
        ) = self.result.x

        beta_fit = np.array([
            np.log(k0_fit),
            x_dagger_fit,
            np.log(dG_dagger_fit)
        ])

        def nll(beta):

            return -self.compute_log_likelihood(
                k0=np.exp(beta[0]),
                x_dagger=beta[1],
                dG_dagger=np.exp(beta[2])
            )

        self.H_log = approx_hess(
            beta_fit,
            nll
        )

        C_log = np.linalg.inv(
            self.H_log
        )

        # Transform back to
        # [k0, x_dagger, dG_dagger]
        J = np.array([
            [k0_fit, 0, 0],
            [0, 1, 0],
            [0, 0, dG_dagger_fit]
        ])

        self.C = (
            J
            @ C_log
            @ J.T
        )


    def _estimate_hessian_log(self):

        if self.rate_model == "bell":
            self._estimate_hessian_log_bell()

        elif self.rate_model == "CHS":
            self._estimate_hessian_log_CHS()

        else:

            raise ValueError(
                "Select a valid rate model (bell/CHS)"
            )


    # ========================================================
    # Standard errors
    # ========================================================

    def _errors_are_valid(self):

        return (
            np.all(np.isfinite(self.H))
            and np.all(np.isfinite(self.C))
            and np.all(np.isfinite(self.errors))
        )


    def get_standard_errors(self):

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    RuntimeWarning
                )

                self._estimate_hessian()

                self.C = np.linalg.inv(
                    self.H
                )

                self.errors = np.sqrt(
                    np.diag(self.C)
                )

                if not self._errors_are_valid():

                    raise ValueError(
                        "Ordinary Hessian failed."
                    )

            self.error_method = (
                "ordinary Hessian"
            )

        except Exception:

            print(
                "Ordinary Hessian failed. "
                "Using log-parameter Hessian."
            )

            self._estimate_hessian_log()

            self.errors = np.sqrt(
                np.diag(self.C)
            )

            self.error_method = (
                "log-parameter Hessian"
            )


        si_sj = np.outer(
            self.errors,
            self.errors
        )

        self.correlation_matrix = (
            self.C
            / si_sj
        )

        return (
            self.errors,
            self.correlation_matrix
        )


    # ========================================================
    # Run
    # ========================================================

    def run(self):

        self.initialize_grid()

        if self.force_dependent_loading_rates:
            self.compute_force_dependent_loading_rates_per_trace()

        self.estimate_initial_guesses()
        self.fit()
        self.get_standard_errors()

        return self.result


#%% deprecated

class eWLC_fit_old:
    def __init__(self, distances, forces, fitting_bounds, par1, offsetpar,
                 par2 = [], global_fit = False, fixed_par1 = [False,False,False],
                 fixed_offsetpar = [False,False],  fixed_par2 = []):
        self.distances = distances
        self.forces = forces
        self.fitting_bounds = fitting_bounds
        self.par1 = par1
        self.par2 = par2
        self.global_fit = global_fit
        self.offsetpar = offsetpar
        self.fixed_par1 = fixed_par1
        self.fixed_offsetpar = fixed_offsetpar
        self.fixed_par2 = fixed_par2

        self.dfit = None
        self.ffit = None
        self.model = None
        self.fit_obj = None

    def prepare_data (self):
        if not self.global_fit:
            mask = (self.distances > self.fitting_bounds[0][0]) & (self.distances < self.fitting_bounds[0][1]) & (self.forces > self.fitting_bounds[1][0]) & (self.forces < self.fitting_bounds[1][1])
            self.dfit = self.distances[mask]
            self.ffit = self.forces[mask]
        else:
            self.dfit = []
            self.ffit = []

            for d,f in zip(self.distances, self.forces):
                mask = (d > self.fitting_bounds[0][0]) & (d < self.fitting_bounds[0][1]) & (f > self.fitting_bounds[1][0]) & (f < self.fitting_bounds[1][1])

                self.dfit.append(d[mask])
                self.ffit.append(f[mask])

    def create_fit(self):
        if not self.par2:
            self.model = lk.ewlc_odijk_force("par1").subtract_independent_offset() + lk.force_offset("par1")
        else:
            if len(self.par2) == 2:
                self.model = lk.ewlc_odijk_distance("par1").subtract_independent_offset() + \
                        lk.distance_offset("par1") + \
                        lk.wlc_marko_siggia_distance("par2")
            else:
                self.model = lk.ewlc_odijk_distance("par1").subtract_independent_offset() + \
                        lk.distance_offset("par1") + \
                        lk.ewlc_odijk_distance("par2")

            self.model = self.model.invert()

        self.fit_obj = lk.FdFit(self.model)

    def add_data(self):
        if not self.global_fit:
            self.fit_obj.add_data("", self.ffit, self.dfit)
        else:
            for i, (d, f) in enumerate(zip(self.dfit, self.ffit)):
                if i == 0:
                    self.fit_obj.add_data (f"AdK {i}", f, d)
                else:
                    self.fit_obj.add_data(f"AdK {i}", f, d, params={"par1/f_offset": f"par1/f_offset_{i}",
                                                           "par1/d_offset": f"par1/d_offset_{i}"})

    def set_fit_guesses_n_bounds (self):
        Lp_bounds = self.par1[0]
        Lc_bounds = self.par1[1]
        St_bounds = self.par1[2]
        doffset_bounds = self.offsetpar[0]
        foffset_bounds = self.offsetpar[1]

        # par1 definition
        ## guess
        self.fit_obj["par1/Lp"].value = (Lp_bounds[0] + Lp_bounds[1])/2
        self.fit_obj["par1/Lc"].value = (Lc_bounds[0] + Lc_bounds[1])/2
        self.fit_obj["par1/St"].value = (St_bounds[0] + St_bounds[1])/2
        self.fit_obj["par1/d_offset"].value = (doffset_bounds[0] + doffset_bounds[1])/2
        self.fit_obj["par1/f_offset"].value = (foffset_bounds[0] + foffset_bounds[1])/2

        ## bounds
        ## Defining bounds
        self.fit_obj["par1/Lp"].lower_bound = Lp_bounds[0]
        self.fit_obj["par1/Lp"].upper_bound = Lp_bounds[1]

        self.fit_obj["par1/Lc"].lower_bound = Lc_bounds[0]
        self.fit_obj["par1/Lc"].upper_bound =  Lc_bounds[1]

        self.fit_obj["par1/St"].lower_bound = St_bounds[0]
        self.fit_obj["par1/St"].upper_bound = St_bounds[1]

        self.fit_obj["par1/d_offset"].lower_bound = doffset_bounds[0]
        self.fit_obj["par1/d_offset"].upper_bound = doffset_bounds[1]

        self.fit_obj["par1/f_offset"].lower_bound = foffset_bounds[0]
        self.fit_obj["par1/f_offset"].upper_bound = foffset_bounds[1]

        ## global fit needs to set object for each fec
        if self.global_fit:
            for i in range(len(self.dfit)):
                if i != 0:
                    self.fit_obj["par1/d_offset_" + str(i)].value = (doffset_bounds[0] + doffset_bounds[1])/2
                    self.fit_obj["par1/d_offset_" + str(i)].lower_bound = doffset_bounds[0]
                    self.fit_obj["par1/d_offset_" + str(i)].upper_bound = doffset_bounds[1]

                    self.fit_obj["par1/f_offset_" + str(i)].value = (foffset_bounds[0] + foffset_bounds[1])/2
                    self.fit_obj["par1/f_offset_" + str(i)].lower_bound = foffset_bounds[0]
                    self.fit_obj["par1/f_offset_" + str(i)].upper_bound = foffset_bounds[1]

        #par2 if neccessary
        if self.par2:
            Lp_bounds = self.par2[0]
            Lc_bounds = self.par2[1]
            if len(self.par2) == 3:
                St_bounds = self.par2[2]

            # par2 definition
            ## guess
            self.fit_obj["par2/Lp"].value = (Lp_bounds[0]+Lp_bounds[1])/2
            self.fit_obj["par2/Lc"].value = (Lc_bounds[0]+Lc_bounds[1])/2
            if len(self.par2) == 3:
                self.fit_obj["par2/St"].value = (St_bounds[0]+St_bounds[1])/2

            ##bounds
            self.fit_obj["par2/Lp"].lower_bound = Lp_bounds[0]
            self.fit_obj["par2/Lp"].upper_bound = Lp_bounds[1]

            self.fit_obj["par2/Lc"].lower_bound = Lc_bounds[0]
            self.fit_obj["par2/Lc"].upper_bound = Lc_bounds[1]

            if len(self.par2) == 3:
                self.fit_obj["par2/St"].lower_bound = St_bounds[0]
                self.fit_obj["par2/St"].upper_bound = St_bounds[1]

    def set_fixed_parameters (self):

        #par1
        self.fit_obj["par1/Lp"].fixed = self.fixed_par1[0]
        self.fit_obj["par1/Lc"].fixed = self.fixed_par1[1]
        self.fit_obj["par1/St"].fixed = self.fixed_par1[2]
        self.fit_obj["par1/d_offset"].fixed = self.fixed_offsetpar[0]
        self.fit_obj["par1/f_offset"].fixed = self.fixed_offsetpar[1]

        if self.global_fit:
            for i in range(len(self.dfit)):
                if i != 0:
                    self.fit_obj["par1/d_offset_" + str(i)].fixed = self.fixed_offsetpar[0]
                    self.fit_obj["par1/f_offset_" + str(i)].fixed = self.fixed_offsetpar[1]

        #par2
        if self.par2:
            self.fit_obj["par2/Lp"].fixed = self.fixed_par2[0]
            self.fit_obj["par2/Lc"].fixed = self.fixed_par2[1]
            if len(self.par2) == 3:
                self.fit_obj["par2/St"].fixed = self.fixed_par2[2]

    def fit(self):
        self.prepare_data()
        self.create_fit()
        self.add_data()
        self.set_fit_guesses_n_bounds()
        self.set_fixed_parameters()
        self.fit_obj.fit()

