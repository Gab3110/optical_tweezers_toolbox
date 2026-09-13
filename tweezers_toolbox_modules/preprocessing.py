#%% IMPORT EXTERNAL MODULES

import sys
import numpy as np
from scipy import signal
import lumicks.pylake as lk
import prox_tv
import matplotlib.pyplot as plt
from scipy.signal import bessel,sosfiltfilt,filtfilt, butter, medfilt, savgol_filter
from hampel import hampel

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.fitting import OLS_fit
from tweezers_toolbox_modules.fitting import create_design_matrix

#%% DEFINING MODULE LEVEL FUNCTIONS

def lf_preprocessing (baseline_path, fec_path,
                            min_bead_distance = 0.2):

    # Sanity check
    try:
        baseline = lk.File(baseline_path)
    except:
        return ("No baseline")

    # High frequency
    fec = lk.File (fec_path)

    f2_offset  = np.mean(baseline.force2x[:"0.50s"].data)
    
    # Low frequency

    baseline_d_lf = baseline.distance1.data

    background_force2x_lf = baseline.downsampled_force2x.data - f2_offset

    if min(baseline_d_lf) < min_bead_distance:
        cal_range_lf = [min_bead_distance, max(baseline_d_lf)]
    else:
        cal_range_lf = [min(baseline_d_lf), max(baseline_d_lf)]

    baseline2x_par_lf = np.polyfit(baseline_d_lf, background_force2x_lf, 7)

    fec_time_lf = fec.distance1.seconds
    d_lf = fec.distance1.data
    f_lf = fec.downsampled_force2x.data - f2_offset

    mask = (d_lf > cal_range_lf[0])&(d_lf < cal_range_lf[1])
    fec_time_lf = fec_time_lf[mask]
    d_lf = d_lf[mask]
    f_lf = f_lf[mask]

    fec_corrected_lfforce2x = f_lf - np.polyval(baseline2x_par_lf, d_lf)
    return (d_lf, fec_corrected_lfforce2x)

def get_sampling_freq (file):
    frequency = 1/((file.force2x.timestamps[1] - file.force2x.timestamps[0])*10**-9)
    return(frequency)

def get_downsampling_factor (original_frequency,target_freq):
    downsampling_factor = original_frequency/target_freq
    return(int(downsampling_factor))


def CK_filtfilt (x, downsampling_factor, decimate = True):
    # Apply filter
    xfilt = filtfilt(np.ones(downsampling_factor)/downsampling_factor,1,x)

    #Decimate
    if decimate:
        xfilt = xfilt[0+downsampling_factor:len(xfilt)-downsampling_factor]
        xfilt = xfilt[0:len(xfilt):downsampling_factor]

    return(xfilt)

def savgol_filt (x, downsampling_factor, polyorder = 1, decimate = True):
    xfilt = savgol_filter(x, window_length = downsampling_factor, polyorder=polyorder)
    
    #Decimate
    if decimate:
        xfilt = xfilt[0+downsampling_factor:len(xfilt)-downsampling_factor]
        xfilt = xfilt[0:len(xfilt):downsampling_factor]
    
    return(xfilt)

def bessel_filtfilt(x, downsampling_factor, fs_original, order = 4,
                    decimate = True, axis = -1, cutoff_factor = 0.8):
    # define nyq of the data
    nyq_original = 0.5 * fs_original

    # define new frequency
    fs_new = fs_original / downsampling_factor

    # define new nyq
    nyq_new = fs_new/2

    # Safe cutoff: ~50%
    cutoff = cutoff_factor * nyq_new  # Hz
    Wn = cutoff / nyq_original

    # Design 4-pole Bessel filter
    sos = bessel(N=order, Wn=Wn, btype='low', analog=False, norm='phase', output='sos')

    # Apply zero-phase filtering
    x_filtered = sosfiltfilt(sos, x, axis=axis)

    # Trim edges to remove filtering artifacts
    if decimate:
        if len(x_filtered) > 2 * downsampling_factor:
            x_filtered = x_filtered[downsampling_factor:-downsampling_factor]
        x_filtered = x_filtered[::downsampling_factor]

    return (x_filtered)

def hampel_plus_bessel_filtfilt(x, downsampling_factor, fs_original, order = 4, decimate = True, axis = -1, cutoff_factor = 0.8):
    
    x = hampel(x, window_size=downsampling_factor).filtered_data
    
    # define nyq of the data
    nyq_original = 0.5 * fs_original

    # define new frequency
    fs_new = fs_original / downsampling_factor

    # define new nyq
    nyq_new = fs_new/2

    # Safe cutoff: ~50%
    cutoff = cutoff_factor * nyq_new  # Hz
    Wn = cutoff / nyq_original

    # Design 4-pole Bessel filter
    sos = bessel(N=order, Wn=Wn, btype='low', analog=False, norm='phase', output='sos')

    # Apply zero-phase filtering
    x_filtered = sosfiltfilt(sos, x, axis=axis)

    # Trim edges to remove filtering artifacts
    if decimate:
        if len(x_filtered) > 2 * downsampling_factor:
            x_filtered = x_filtered[downsampling_factor:-downsampling_factor]
        x_filtered = x_filtered[::downsampling_factor]

    return (x_filtered)

def median_plus_bessel (x, downsampling_factor, fs_original, order = 4, decimate = True, axis = -1, cutoff_factor = 0.8):
    # Median filter
    med = medfilt(x, kernel_size = downsampling_factor)
    
    # bessel
    x_filtered = bessel_filtfilt(med, downsampling_factor, fs_original, order, decimate, axis, cutoff_factor)
    
    return(x_filtered)
    
def butter_filtfilt(x, downsampling_factor, fs_original ,order = 4,  decimate = True, axis = -1, cutoff_factor = 0.8):
    # define nyq of the data
    nyq_original = 0.5 * fs_original

    # define new frequency
    fs_new = fs_original / downsampling_factor

    # define new nyq
    nyq_new = fs_new/2

    # Safe cutoff: ~50%
    cutoff = cutoff_factor * nyq_new # Hz
    Wn = cutoff / nyq_original

    # Design 4-pole butter filter
    sos = butter(N=order, Wn=Wn, btype='low', analog=False, output='sos')

    # Apply zero-phase filtering
    x_filtered = sosfiltfilt(sos, x, axis=axis)

    # Trim edges to remove filtering artifacts
    if decimate:
        if len(x_filtered) > 2 * downsampling_factor:
            x_filtered = x_filtered[downsampling_factor:-downsampling_factor]
        x_filtered = x_filtered[::downsampling_factor]

    return (x_filtered)

def find_tether_rupture (forces, basepoints, force_threshold,smooth_factor, curvetype = "stretch"):
    if curvetype == "stretch":
        flipped_f = np.flip(forces)
    else:
        flipped_f = forces

    flipped_f_smooth = signal.filtfilt(np.ones(smooth_factor)/smooth_factor,1,flipped_f)
    #plt.plot(flipped_f_smooth)
    dflipped_f_smooth = np.gradient(flipped_f_smooth)

    dflipped_f_smooth_slice = dflipped_f_smooth[smooth_factor:basepoints+1]

    # dF threshold
    df_threshold = np.mean(dflipped_f_smooth_slice) + 3*np.std(dflipped_f_smooth_slice)

    # Defining regions of marked change in force
    pass_threshold = np.where(dflipped_f_smooth > df_threshold)[0]

    # Defining the rupture index
    rupture_index = 0
    ## First evaluate all points that passed the threshold
    for i in pass_threshold:
        ## If a particular point also passes the second criteria it is selected for
        ## refinement
        if flipped_f_smooth[i] > force_threshold:
            ## Refinement consist in evaluating the difference between the current
            ## point and the next one. If the difference is positive, discard it
            ## and move to the subsequent. Then try again until difference is
            ## negative.
            diff_flipped_f_smooth = flipped_f_smooth[i+1] - flipped_f_smooth[i]
            while diff_flipped_f_smooth > 0:
                i = i+1
                diff_flipped_f_smooth = flipped_f_smooth[i+1] - flipped_f_smooth[i]
            rupture_index = i
            break

    # Once a first-refined point is selected, then transform it to the unflipped
    # index.
    if curvetype == "stretch":
        rupture_index = len(forces) - 1 - rupture_index

    # Final refinement. If there is a point after the temptative rupture index
    # that is higher in force, then take that one as the new rupture.
    rupture_index = np.argmax(forces[rupture_index:len(forces)]) + rupture_index

    return(rupture_index)

def get_raw_trappos_domain (file_path):
    file = lk.File(file_path)
    min_trappos, max_trappos = np.min(file["Trap position"]["1X"].data),np.max(file["Trap position"]["1X"].data)
    return( [min_trappos,max_trappos ] )
    
def calculate_diff_bead_signals (force1x,
                                 force2x,
                                 k1,
                                 k2,
                                 signs = (-1,1)):
    keq = ((1/k1) + (1/k2))**(-1)

    # known as x1
    bead_displacement1 = force1x/k1

    # known as x2
    bead_displacement2 = force2x/k2

    # In our ctrap x1 is to the right, mobile. x2 to the left, inmobile, then
    # diff bead displacement is x2 - x1. Signs[1]=+, signs[0]=-1
    diff_bead_displacement = signs[1]*bead_displacement2 + signs[0]*bead_displacement1

    # diffF
    diffF = keq*diff_bead_displacement

    return (keq, bead_displacement1, bead_displacement2, diff_bead_displacement,diffF)

def calculate_molext (position,
                      diff_bead_displacement,
                      piezo_tracking_model = False,
                      piezocal = False):

    if piezocal:
        if isinstance(piezo_tracking_model, list):
            raw_sep = np.polyval(piezo_tracking_model, position)
        else:
            fec_trappos1x_pos_mat = create_design_matrix (position,len(piezo_tracking_model.params)-1)
            # Predicting raw bead separation
            raw_sep = piezo_tracking_model.predict(fec_trappos1x_pos_mat)

    else:
        raw_sep = position

    diff_bead_displacement_um = (1/1000)*diff_bead_displacement
    mol_ext = raw_sep - diff_bead_displacement_um
    return(raw_sep,mol_ext)

#%% MAIN CLASSES

class TVDenoising():
    def __init__ (self,
                  signal,
                  all_signals_from_batch,
                  w_ratio=2**(1/3),
                  w0 = 1e-1,
                  max_expand=50,
                  tv_drop_target=1e2, 
                  tv_floor_eps=1e-12,
                  plot = True):
        
        self.signal = np.asarray(signal, dtype=float)
        self.all_signals_from_batch = np.asarray(all_signals_from_batch, dtype=float)
        self.w0 = w0
        self.max_expand = max_expand
        self.tv_drop_target = tv_drop_target
        self.tv_floor_eps = tv_floor_eps
        self.w_ratio = w_ratio
        self.plot = plot

        self.RMSDs = None
        self.TVs = None
        self.lambdas = None
        self.denoised_signal = None
        self.ws = None
        self.w_star = None
        self.w_start_idx = None
        self.w_finding_method = None
        self.inflect_point_index_1 = None
        self.inflect_point_index_2 = None
        self.inter_log10 = None

        # NEW (optional debug)
        self.kappa = None

    def tv_denoise(self, y, w):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if w < 0:
            raise ValueError("w must be >= 0")
        if not np.isfinite(y).all():
            raise ValueError("signal contains NaN/inf")

        yhat = prox_tv.tv1_1d(y, w, method="dp")
        return yhat
    
    def auto_w_grid(self):
        w0_min = self.w0
        w0_max = self.w0

        yhat0 = self.tv_denoise(self.all_signals_from_batch, self.w0)
        tv0 = np.sum(np.abs(np.diff(yhat0))) + self.tv_floor_eps

        for _ in range(self.max_expand):
            w0_max *= self.w_ratio**10
            yhat_hi = self.tv_denoise(self.all_signals_from_batch, w0_max)
            tv_hi = np.sum(np.abs(np.diff(yhat_hi))) + self.tv_floor_eps
            if (tv0 / tv_hi) >= self.tv_drop_target:
                break

        N = int(np.floor(np.log(w0_max/w0_min) / np.log(self.w_ratio))) + 1
        self.ws = w0_min * self.w_ratio**np.arange(N)

    def tv_lcurve_points(self):
        if self.signal.ndim != 1:
            raise ValueError("y must be 1D")
        if self.ws is None:
            raise ValueError("self.ws is None. Run auto_w_grid() first.")
        if np.any(self.ws <= 0):
            raise ValueError("All w values must be > 0 for log-log L-curve.")

        self.RMSDs = np.empty_like(self.ws, dtype=float)
        self.TVs   = np.empty_like(self.ws, dtype=float)

        for i, w in enumerate(self.ws):
            yhat = self.tv_denoise(self.all_signals_from_batch, w)
            r = self.all_signals_from_batch - yhat
            self.RMSDs[i] = np.sqrt(np.mean(r * r))
            self.TVs[i]   = np.sum(np.abs(np.diff(yhat)))

    # ============================================================
    # CHANGED: keep SAME NAME, but now pick w_star by max curvature
    # ============================================================
    def pick_lambda(self):
        """
        CHANGED: pick w_star by max distance-to-chord ("triangle method") in log-log space.
        Much less likely to pick the far-right plateau than curvature.
        """
        f = np.asarray(self.RMSDs, float)
        t = np.asarray(self.TVs, float)
    
        mask = np.isfinite(f) & np.isfinite(t) & (f > 0) & (t > 0)
        if mask.sum() < 5:
            raise ValueError("Not enough positive finite L-curve points to pick lambda.")
    
        X = np.log10(f[mask])
        Y = np.log10(t[mask])
        orig = np.flatnonzero(mask)
    
        # line from first to last point in (X,Y)
        x0, y0 = X[0],  Y[0]
        x1, y1 = X[-1], Y[-1]
        vx, vy = (x1 - x0), (y1 - y0)
        denom = np.hypot(vx, vy) + 1e-30
    
        # perpendicular distance from each point to the chord
        d = np.abs(vy*(X - x0) - vx*(Y - y0)) / denom
    
        # avoid endpoints
        d[0] = -np.inf
        d[-1] = -np.inf
    
        idx_local = int(np.argmax(d))
        idx_orig = int(orig[idx_local])
    
        self.w_start_idx = idx_orig
        self.w_star = float(self.ws[idx_orig])
        self.w_finding_method = "max_distance_to_chord"
        self.inflect_point_index_1 = None
        self.inflect_point_index_2 = None
        self.inter_log10 = None

    def plot_L_curve(self):
        plt.figure()
        plt.loglog(self.RMSDs, self.TVs, "o")
        if self.w_start_idx is not None:
            plt.loglog(self.RMSDs[self.w_start_idx], self.TVs[self.w_start_idx], "ro", markersize=10)
        plt.xlabel(r'RMSD')
        plt.ylabel(r'Total variation')
        plt.grid(True, which="both")
        plt.show()
        plt.close()
        
    def get_denoised_signal(self):
        if self.w_star is None:
            raise RuntimeError("w_star was not selected.")
        self.denoised_signal = self.tv_denoise(self.signal, self.w_star)
        
    def denoise(self):
        self.auto_w_grid()
        self.tv_lcurve_points()
        self.pick_lambda()  # same call name, new behavior
        if self.plot:
            self.plot_L_curve()
        self.get_denoised_signal()
    
                  
class VideoMicroscopyRegression():
    def __init__ (self, baseline_path,
                  crange,
                  video_trappos_regression_pol_degree,
                  video_trap_regress_cal = False,
                  bead1_diameter = 0,
                  bead2_diameter = 0):
        self.baseline_path = baseline_path
        self.video_trap_regress_cal = video_trap_regress_cal
        self.video_trappos_regression_pol_degree = video_trappos_regression_pol_degree
        self.bead1_diameter = bead1_diameter
        self.bead2_diameter = bead2_diameter

        self.error_code = 0

        self.crange = crange
        self.ds_trappos1x_trim = None
        self.cdist1_trim = None
        self.pred_dist = None
        self.baseline_h5_file = None
        self.video_trap_regress_obj = None
        self.video_trap_regress_R2 = None
        self.video_trap_regress_slope = None
        self.video_trap_regress_intercept = None
        self.bead_position_1_trim = None
        self.bead_position_2_trim = None

    def loading_baseline(self):
        try:
            self.baseline_h5_file = lk.File(self.baseline_path)
            return True
        except FileNotFoundError:
            self.error_code = 1
            return False

    def video_trappos_regression(self):
        try:
            # For some stupid reason, lumicks has the bead absolute position labelling inverted.
            # For HF signals, the inmobile trap is 2X and the mobile trap is 1X.
            # For bead absolute positions (LF, detected with camera) the inmobile bead is 1X
            # and the mobile is 2X. We need to pay attention if they change this in future bluelake updates
            # and edit accordingly. For now, I am inverting the numbers so everything is consistent.
            # I. e "2" will be fixed in all signals.
            # NOTE: Bead absolute position channels are the coordinates of the bead center. It is not corrected
            # for bead radius, which is fine. You can substract later.
            #ds_trappos1x, cdisttest = self.baseline_h5_file["Trap position"]["1X"].downsampled_like(self.baseline_h5_file.distance1)
            ds_trappos1x, bead_position_1 = self.baseline_h5_file["Trap position"]["1X"].downsampled_like(self.baseline_h5_file["Bead position"]["Bead 2 X"])
            ds_trappos1x, bead_position_2 = self.baseline_h5_file["Trap position"]["1X"].downsampled_like(self.baseline_h5_file["Bead position"]["Bead 1 X"])
            ds_trappos1x = ds_trappos1x.data
            bead_position_1 = bead_position_1.data
            bead_position_2 = bead_position_2.data
            # Based on pylake tutorial (piezo tracking): To convert absolute trap positions 1x
            # to trap 1 and 2 separation, you run a linear regression between the beads 
            #surface-to-surface distance and the mirror position. To obtain the surface-to-sorface
            # distance you can use the precomputed distance1 channel of each trap. However,
            # I like to do it manually so I have control on the bead radiuses. The calculation
            # that distance 1 is idential to bead_position_1 -  bead_position_2 minus
            # the combined radiuses of both beads. 
            # NOTE: bead_position_2 represent the position of trap 2, which should be inmobile.
            # Howver, it is an array (time series) that does vary over time slighly due to 
            # drift or errors in tracking distance. If you want to assume the position
            # is truly constant, you can make bead_position_2 equal to the first item
            # of the original array. This is what bluelake does.
            cdist1 = bead_position_1 -  bead_position_2 - self.bead1_diameter/2 - self.bead2_diameter/2
        except:
            ds_trappos1x, cdist1 = self.baseline_h5_file["Trap position"]["1X"].downsampled_like(self.baseline_h5_file.distance1)
            ds_trappos1x = ds_trappos1x.data
            cdist1 = cdist1.data
            
        mask = (cdist1 != 0) & (ds_trappos1x > self.crange[0]) & (ds_trappos1x < self.crange[1])
        
        self.ds_trappos1x_trim = ds_trappos1x[mask]
        try:
            self.bead_position_1_trim = bead_position_1[mask]
            self.bead_position_2_trim = bead_position_2[mask]
        except:
            pass
        
        self.cdist1_trim = cdist1[mask]
        self.video_trap_regress_obj, res = OLS_fit (self.ds_trappos1x_trim, self.cdist1_trim, self.video_trappos_regression_pol_degree)
        self.pred_dist = self.video_trap_regress_obj.predict(self.video_trap_regress_obj.model.exog)
        self.video_trap_regress_R2 = self.video_trap_regress_obj.rsquared
        self.video_trap_regress_intercept = self.video_trap_regress_obj.params[0]
        self.video_trap_regress_slope =  self.video_trap_regress_obj.params[1]

    def eval_video_trap_regress_cal_qual (self, threshold = 0.99):
        if self.video_trap_regress_R2 < threshold:
            self.error_code = 3
            return False
        else:
            return True

    def run (self):
        self.loading_baseline()
        if self.error_code != 0:
            return
        else:
            self.video_trappos_regression()
            if self.video_trap_regress_cal:
                self.eval_video_trap_regress_cal_qual()

class BaselineModelling():
    def __init__(self,
                 baseline_path,
                 downfact,
                 original_samp_freq,
                 bsl_pol_degree,
                 crange,
                 filter_type = "boxcar"):

        self.baseline_path = baseline_path
        self.downfact = downfact
        self.bsl_pol_degree = bsl_pol_degree
        self.crange = crange
        self.original_samp_freq = original_samp_freq
        self.filter_type = filter_type

        self.baseline_h5_file = None
        self.calpars = None
        self.f1_offset = None
        self.f2_offset = None
        self.bsl1_model = None
        self.bsl2_model = None

        # Downsampled signals
        self.baseline_timeds = None
        self.baseline_trappos1xds = None
        self.baseline_force1xds = None
        self.pred_bsl1ds = None
        self.baseline_corrected_force_1xds = None
        self.res_bsl_f1xds = None
        self.baseline_force2xds = None
        self.pred_bsl2ds = None
        self.baseline_corrected_force_2xds = None
        self.res_bsl_f2xds = None

        # wrap downsampled baseline signals into a dict
        self.baselineds_dict = None
        self.error_code = 0

    def loading_baseline(self):
        try:
            self.baseline_h5_file = lk.File(self.baseline_path)
            return True
        except FileNotFoundError:
            self.error_code = 1
            return False

    def extract_cal_pars(self):
        # trap 1, mobile
        self.calpars = {}
        self.calpars["bead1"] = self.baseline_h5_file.force1x.calibration[0]
        self.calpars["bead2"] = self.baseline_h5_file.force2x.calibration[0]
    
    def downsample_signals(self):
        if self.filter_type == "boxcar":
            self.baseline_trappos1xds = CK_filtfilt (self.baseline_h5_file['Trap position']['1X'].data, self.downfact)
            self.baseline_force1xds = CK_filtfilt(self.baseline_h5_file.force1x.data, self.downfact)
            self.baseline_force2xds = CK_filtfilt(self.baseline_h5_file.force2x.data, self.downfact)
            ts = self.downfact/self.original_samp_freq
            self.baseline_timeds = np.arange(len(self.baseline_trappos1xds)) * ts
            
        elif self.filter_type == "bessel":
            self.baseline_trappos1xds = bessel_filtfilt (self.baseline_h5_file['Trap position']['1X'].data, self.downfact, self.original_samp_freq)
            self.baseline_force1xds = bessel_filtfilt(self.baseline_h5_file.force1x.data, self.downfact, self.original_samp_freq)
            self.baseline_force2xds = bessel_filtfilt(self.baseline_h5_file.force2x.data, self.downfact, self.original_samp_freq)
            ts = self.downfact/self.original_samp_freq
            self.baseline_timeds = np.arange(len(self.baseline_trappos1xds)) * ts

        elif self.filter_type == "butter":
            self.baseline_trappos1xds = butter_filtfilt (self.baseline_h5_file['Trap position']['1X'].data, self.downfact, self.original_samp_freq)
            self.baseline_force1xds = butter_filtfilt(self.baseline_h5_file.force1x.data, self.downfact, self.original_samp_freq)
            self.baseline_force2xds = butter_filtfilt(self.baseline_h5_file.force2x.data, self.downfact, self.original_samp_freq)
            ts = self.downfact/self.original_samp_freq
            self.baseline_timeds = np.arange(len(self.baseline_trappos1xds)) * ts
        else:
            raise ValueError("Invalid filter type. Please select boxcar, bessel or butter.")
    
    def flush_h5_files(self):
        self.fec_h5_file = None
    
    def model_baselines (self, pol_degree = False):
        if not pol_degree:
            pol_degree = self.bsl_pol_degree

        # Get offsets        
        mask = self.baseline_timeds <= 0.25
        self.f1_offset = np.mean(self.baseline_force1xds[mask])
        self.f2_offset = np.mean(self.baseline_force2xds[mask])

        # Correct force offsets
        self.baseline_force1xds = self.baseline_force1xds - self.f1_offset
        self.baseline_force2xds = self.baseline_force2xds - self.f2_offset

        # Trim based on crange
        mask = (self.baseline_trappos1xds >  self.crange[0]) & (self.baseline_trappos1xds <  self.crange[1])
        self.baseline_timeds = self.baseline_timeds[mask]
        self.baseline_trappos1xds = self.baseline_trappos1xds[mask]
        self.baseline_force1xds = self.baseline_force1xds[mask]
        self.baseline_force2xds = self.baseline_force2xds[mask]

        # Fit baselines
        self.bsl1_model, self.res_bsl_f1xds = OLS_fit (self.baseline_trappos1xds,self.baseline_force1xds,
                                                         pol_degree)
        self.bsl2_model, self.res_bsl_f2xds = OLS_fit (self.baseline_trappos1xds,self.baseline_force2xds,
                                                         pol_degree)

        # Predict baseline forces
        self.pred_bsl1ds = self.bsl1_model.predict(self.bsl1_model.model.exog)
        self.pred_bsl2ds = self.bsl2_model.predict(self.bsl2_model.model.exog)

        # correct baseline forces (sanity check, baseline forces after correction
        # should be flat).
        self.baseline_corrected_force_1xds = self.baseline_force1xds - self.pred_bsl1ds
        self.baseline_corrected_force_2xds = self.baseline_force2xds - self.pred_bsl2ds
        
    def wrap_baseline_results_in_dict (self):
        self.baselineds_dict = {}
        self.baselineds_dict["f1_offset"] = self.f1_offset
        self.baselineds_dict["f2_offset"] = self.f2_offset
        self.baselineds_dict["bsl_pol_degree"] = self.bsl_pol_degree
        self.baselineds_dict["coeff_bsl1"] = str(list(self.bsl1_model.params))
        self.baselineds_dict["coeff_bsl2"] = str(list(self.bsl2_model.params))
        self.baselineds_dict["baseline_trappos1xds"] = self.baseline_trappos1xds
        self.baselineds_dict["baseline_force1xds"] = self.baseline_force1xds
        self.baselineds_dict["pred_bsl1ds"] = self.pred_bsl1ds
        self.baselineds_dict["baseline_corrected_force_1xds"] = self.baseline_corrected_force_1xds
        self.baselineds_dict["res_bsl_f1xds"] = self.res_bsl_f1xds
        self.baselineds_dict["baseline_force2xds"] = self.baseline_force2xds
        self.baselineds_dict["pred_bsl2ds"] = self.pred_bsl2ds
        self.baselineds_dict["baseline_corrected_force_2xds"] = self.baseline_corrected_force_2xds
        self.baselineds_dict["res_bsl_f2xds"] = self.res_bsl_f2xds
        self.baselineds_dict["baseline_timeds"] = self.baseline_timeds
    
    def model(self):
        self.loading_baseline()
        if self.error_code != 0:
            return
        else:
            self.extract_cal_pars()
            self.downsample_signals()
            self.flush_h5_files()
            self.model_baselines()
            
            if self.error_code !=0:
                return
            else:
                self.wrap_baseline_results_in_dict()

class PiezoTracking():
    def __init__(self,
                 fec_path,
                 baseline_path,
                 video_trappos_regression_pol_degree,
                 downfact,
                 bsl_pol_degree,
                 crange,
                 options,
                 video_trappos_regression_bsl_mdl_results = None):

        # User defined attributes
        self.fec_path = fec_path
        self.baseline_path = baseline_path
        self.bsl_pthname_current_base = self.baseline_path.replace(".h5","")
        self.bsl_pthname_current_base = self.bsl_pthname_current_base.replace("rawdat/","")
        self.options = options
        self.baseline_path = baseline_path
        self.bsl_pthname_current_base = self.baseline_path.replace(".h5","")
        self.bsl_pthname_current_base = self.bsl_pthname_current_base.replace("rawdat/","")
        self.downfact = downfact
        self.bsl_pol_degree = bsl_pol_degree
        self.video_trappos_regression_pol_degree = video_trappos_regression_pol_degree
        self.filter_type = self.options.get("filter_type", "boxcar")
        self.crange = crange


        # Initialize empty attributes
        self.error_code = 0
        self.video_microscopy = None
        self.baselines = None
        self.samp_freq_hz = None
        self.fec_h5_file = None
        self.baseline_h5_file = None
        
        ## Downsampled signals
        self.fec_timeds = None
        self.fec_trappos1xds = None
        self.fec_cal_trappos1xds = None
        self.fec_force1xds = None
        self.fec_force2xds = None
        self.fec_corrected_force_1xds = None
        self.fec_corrected_force_2xds = None
        self.fec_molextds = None
        self.fec_cal_molextds = None
        self.fec_bead_disp1ds = None
        self.fec_bead_disp2ds = None
        self.fec_diff_bead_dispds = None
        self.fec_diffFds = None

        # wrap fec downsampled signals to a dict
        self.fecds_dict = None

        ### LF data, from camera
        self.fec_time_lf = None
        self.fec_lfdist = None
        self.fec_lfforce2x = None
        self.fec_corrected_lfforce2x = None

        # Populate empty attributes if results are supplied
        self.video_trappos_regression_bsl_mdl_results = video_trappos_regression_bsl_mdl_results

        if self.video_trappos_regression_bsl_mdl_results:
            self._load_precomputed_results()
        
    def loading_baseline(self):
        try:
            self.baseline_h5_file = lk.File(self.baseline_path)
            return True
        except FileNotFoundError:
            self.error_code = 1
            return False
        
    def loading_fec (self):
        try:
            self.fec_h5_file = lk.File(self.fec_path)
            return True
        except FileNotFoundError:
            self.error_code = 2
            return False

    def get_sampling_freq (self):
        # Get original sampling frequency
        self.samp_freq_hz = round(1/(np.diff(self.fec_h5_file['Trap position']['1X'].timestamps[:2])[0]*10**-9))

    def compute_video_trappos_regression(self):
        if not self.video_trappos_regression_bsl_mdl_results:
            if self.options.get("apply_dist_offset", False) or self.options.get("video_trap_regress_cal", False):
                if self.options.get("apply_dist_offset", False):
                    self.video_microscopy = VideoMicroscopyRegression(self.baseline_path,
                                                                      self.crange,
                                                                      self.video_trappos_regression_pol_degree,
                                                                      video_trap_regress_cal = self.options.get("video_trap_regress_cal", False),
                                                                      bead1_diameter= self.options["bead1_diameter"],
                                                                      bead2_diameter= self.options["bead2_diameter"])
                else:
                    self.video_microscopy = VideoMicroscopyRegression(self.baseline_path,
                                                                      self.crange,
                                                                      self.video_trappos_regression_pol_degree,
                                                                      video_trap_regress_cal = self.options.get("video_trap_regress_cal", False))

                self.video_microscopy.run()
                self.error_code = self.video_microscopy.error_code

    def baseline_modelling(self):
        if not self.video_trappos_regression_bsl_mdl_results:
            self.baselines = BaselineModelling (self.baseline_path,
                                                self.downfact,
                                                self.samp_freq_hz,
                                                self.bsl_pol_degree,
                                                self.crange,
                                                self.filter_type)
            self.baselines.model()
            self.error_code = self.baselines.error_code

    def downsample_signals(self):        
        if self.filter_type == "boxcar":
            self.fec_trappos1xds = CK_filtfilt (self.fec_h5_file['Trap position']['1X'].data, self.downfact)
            self.fec_force1xds = CK_filtfilt(self.fec_h5_file.force1x.data, self.downfact)
            self.fec_force2xds = CK_filtfilt(self.fec_h5_file.force2x.data, self.downfact)
            ts = self.downfact/self.samp_freq_hz
            self.fec_timeds = np.arange(len(self.fec_trappos1xds)) * ts
            
        elif self.filter_type == "bessel":
            self.fec_trappos1xds = bessel_filtfilt (self.fec_h5_file['Trap position']['1X'].data, self.downfact, self.samp_freq_hz)
            self.fec_force1xds = bessel_filtfilt(self.fec_h5_file.force1x.data, self.downfact, self.samp_freq_hz)
            self.fec_force2xds = bessel_filtfilt(self.fec_h5_file.force2x.data, self.downfact, self.samp_freq_hz)
            ts = self.downfact/self.samp_freq_hz
            self.fec_timeds = np.arange(len(self.fec_trappos1xds)) * ts

        elif self.filter_type == "butter":
            self.fec_trappos1xds = butter_filtfilt (self.fec_h5_file['Trap position']['1X'].data, self.downfact, self.samp_freq_hz)
            self.fec_force1xds = butter_filtfilt(self.fec_h5_file.force1x.data, self.downfact, self.samp_freq_hz)
            self.fec_force2xds = butter_filtfilt(self.fec_h5_file.force2x.data, self.downfact, self.samp_freq_hz)
            ts = self.downfact/self.samp_freq_hz
            self.fec_timeds = np.arange(len(self.fec_trappos1xds)) * ts
        else:
            raise ValueError("Invalid filter type. Please select boxcar, bessel or butter.")
    
    def flush_h5_files(self):
        self.fec_h5_file = None

    def trimm_n_offset_fec (self):
        # Get trappos1x and apply offsets
        self.fec_force1xds = self.fec_force1xds - self.baselines.f1_offset
        self.fec_force2xds = self.fec_force2xds - self.baselines.f2_offset

        # Defining trimming mask based on piezo calibration range
        mask = (self.fec_trappos1xds > self.crange[0])&(self.fec_trappos1xds < self.crange[1])

        # Trimming fec
        self.fec_trappos1xds = self.fec_trappos1xds[mask]
        self.fec_force1xds = self.fec_force1xds[mask]
        self.fec_force2xds = self.fec_force2xds[mask]
        self.fec_timeds = self.fec_timeds[mask]

    def apply_baseline_correction (self, pol_degree = False):
        if not pol_degree:
            pol_degree = self.baselines.bsl_pol_degree

        # Transforming the independent variable (trappos1x)
        fec_trappos1x_mat = create_design_matrix (self.fec_trappos1xds, pol_degree)

        # Predicting baseline forces in FEC
        bsl1_forces = self.baselines.bsl1_model.predict(fec_trappos1x_mat)
        bsl2_forces = self.baselines.bsl2_model.predict(fec_trappos1x_mat)

        # Substracting baseline
        self.fec_corrected_force_1xds = self.fec_force1xds - bsl1_forces
        self.fec_corrected_force_2xds = self.fec_force2xds - bsl2_forces

    def correct_force_offset(self):
        mask = (self.fec_timeds >= self.fec_timeds[-1] - 0.25)
        mean_offset_f1x = np.mean(self.fec_corrected_force_1xds[mask])
        mean_offset_f2x = np.mean(self.fec_corrected_force_2xds[mask])

        if (abs(mean_offset_f1x) < 3) and (abs(mean_offset_f2x) < 3):
            self.fec_corrected_force_1xds = self.fec_corrected_force_1xds - mean_offset_f1x
            self.fec_corrected_force_2xds = self.fec_corrected_force_2xds - mean_offset_f2x

    def get_diff_signals (self):
        k1 = self.baselines.calpars["bead1"]["kappa (pN/nm)"]
        k2 = self.baselines.calpars["bead2"]["kappa (pN/nm)"]
        keq, self.fec_bead_disp1ds, self.fec_bead_disp2ds, self.fec_diff_bead_dispds, self.fec_diffFds = calculate_diff_bead_signals (self.fec_corrected_force_1xds,
                                                                                                                              self.fec_corrected_force_2xds,
                                                                                                                              k1,
                                                                                                                              k2,
                                                                                                                              signs = (-1,1))
        self.baselines.calpars["keq"] = keq
        
    def calculate_ext(self):
        # if calibration is on
        if self.options.get("video_trap_regress_cal", False):
            # Calibrate trap position with a slope and use it to calculate cal molext
            self.fec_cal_trappos1xds, self.fec_cal_molextds = calculate_molext(self.fec_trappos1xds,
                                                                           self.fec_diff_bead_dispds,
                                                                           self.video_microscopy.video_trap_regress_obj,
                                                                           piezocal = True)
            # Rerun function but without trap pos cal, to get raw molext
            # Here the function will first return its argument self.fec_trappos1x without changes
            _, self.fec_molextds = calculate_molext(self.fec_trappos1xds,
                                                              self.fec_diff_bead_dispds,
                                                              self.video_microscopy.video_trap_regress_obj,
                                                              piezocal = False)

        # If calibration is off
        # calibrated counterparts will be None
        else:
            _, self.fec_molextds  = calculate_molext(self.fec_trappos1xds,
                                                                    self.fec_diff_bead_dispds,
                                                                    piezo_tracking_model = False,
                                                                    piezocal = False)

        # if distance offset is on
        if self.options.get("apply_dist_offset", False):
            # apply rough offsets to the raw trappos1x and fec_molext
            self.fec_trappos1xds = self.fec_trappos1xds + self.video_microscopy.video_trap_regress_obj.params[0]
            self.fec_molextds = self.fec_molextds + self.video_microscopy.video_trap_regress_obj.params[0]

    def wrap_fec_results_in_dict (self):
        self.fecds_dict = {}
        self.fecds_dict["fec_timeds"] = self.fec_timeds
        self.fecds_dict["fec_trappos1xds"] = self.fec_trappos1xds
        self.fecds_dict["fec_cal_trappos1xds"] = self.fec_cal_trappos1xds
        self.fecds_dict["fec_force1xds"] = self.fec_force1xds
        self.fecds_dict["fec_force2xds"] = self.fec_force2xds
        self.fecds_dict["fec_corrected_force_1xds"] = self.fec_corrected_force_1xds
        self.fecds_dict["fec_corrected_force_2xds"] = self.fec_corrected_force_2xds
        self.fecds_dict["fec_molextds"] = self.fec_molextds
        self.fecds_dict["fec_cal_molextds"] = self.fec_cal_molextds
        self.fecds_dict["fec_bead_disp1ds"] = self.fec_bead_disp1ds
        self.fecds_dict["fec_bead_disp2ds"] = self.fec_bead_disp2ds
        self.fecds_dict["fec_diff_bead_dispds"] = self.fec_diff_bead_dispds
        self.fecds_dict["fec_diffFds"] = self.fec_diffFds
            
    def preprocess(self):
        self.loading_fec()
        if self.error_code != 0:
            return
        else:
            self.get_sampling_freq()
            self.compute_video_trappos_regression()

            if self.error_code != 0:
                return
            else:
                self.baseline_modelling()
                if self.error_code !=0:
                    return
                else:
                    self.downsample_signals()
                    self.flush_h5_files()
                    self.trimm_n_offset_fec()
                    self.apply_baseline_correction()

                    if self.options.get("correct_force_offset", False):
                        self.correct_force_offset()

                    self.get_diff_signals()
                    self.calculate_ext()
                    self.wrap_fec_results_in_dict()
                    

    def _load_precomputed_results(self):
        if self.options.get("apply_dist_offset", False) or self.options.get("video_trap_regress_cal", False):
            self.video_microscopy = self.video_trappos_regression_bsl_mdl_results["video_microscopy"]

        self.baselines = self.video_trappos_regression_bsl_mdl_results["baselines"]

#%% # Deprecated

def lf_preprocess (self):
    # Extract, trim and fit a model to low frequency baseline
    baseline_mobile_bead = self.baseline_h5_file["Bead position"]["Bead 2 X"].data
    baseline_inmobile_bead = self.baseline_h5_file["Bead position"]["Bead 1 X"].data
    bsl_lfdist = baseline_mobile_bead - baseline_inmobile_bead
    mask = (bsl_lfdist != 0) & (bsl_lfdist > self.distance_range[0]) & (bsl_lfdist < self.distance_range[1])
    bsl_lfdist = bsl_lfdist[mask]
    background_force2x_lf = (self.baseline_h5_file.downsampled_force2x.data - self.f2_offset)[mask]
    baseline2x_par_lf = np.polyfit(bsl_lfdist, background_force2x_lf, self.bsl_pol_degree_base)

    # Trim and correct low frequency force on fec
    fec_mobile_bead = self.fec_h5_file["Bead position"]["Bead 2 X"].data
    fec_inmobile_bead = self.fec_h5_file["Bead position"]["Bead 1 X"].data
    self.fec_lfdist = fec_mobile_bead -  fec_inmobile_bead
    mask = (self.fec_lfdist != 0) & (self.fec_lfdist > self.distance_range[0]) & (self.fec_lfdist < self.distance_range[1])
    self.fec_lfdist = self.fec_lfdist[mask]

    self.fec_time_lf = self.fec_h5_file.distance1.seconds[mask]
    self.fec_lfforce2x = (self.fec_h5_file.downsampled_force2x.data - self.f2_offset)[mask]
    self.fec_corrected_lfforce2x = self.fec_lfforce2x  - np.polyval(baseline2x_par_lf, self.fec_lfdist)


