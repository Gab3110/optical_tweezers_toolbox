import numpy as np
import matplotlib.pyplot as plt
import copy, sys
#%%

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.preprocessing import calculate_molext, CK_filtfilt

from tweezers_toolbox_modules.models import ini_eWLC
from tweezers_toolbox_modules import fitting
from tweezers_toolbox_modules import analysis as ana
from tweezers_toolbox_modules.general_utils import clip_signal
from tweezers_toolbox_modules.models import get_trap_plus_handles_stiffness_num
from tweezers_toolbox_modules.general_utils import mad

#%%

class FixPosDrift:
    def __init__(self,
                 samp_rate,
                 molext,
                 diffF,
                 smooth_time=1,
                 mad_thresh=5,
                 drift_model_order = 1):

        self.samp_rate = samp_rate
        self.molext = np.asarray(molext)
        self.diffF = np.asarray(diffF)
        self.smooth_time = smooth_time
        self.mad_thresh = mad_thresh
        self.time = np.arange(len(self.diffF)) / self.samp_rate
        self.drift_model_order = drift_model_order
        
        self.drift_rates = None
        self.drift_raw_signal = None
        self.drift_raw_time = None
        self.molext_smooth = None
        self.molext_smooth_trim = None
        self.time_trim = None
        self.dmolext_dt = None
        self.time_dmolext_dt = None
        self.drift = None
        self.corrected_molext = None

    def smooth_molext(self):
        n = int(self.smooth_time * self.samp_rate)
        print(n)
        self.molext_smooth = CK_filtfilt(self.molext,n,decimate=False)
        self.molext_smooth_trim = self.molext_smooth[n:-n]
        self.time_trim = self.time[n:-n]

    def get_dxdt(self):
        self.dmolext_dt = np.diff(self.molext_smooth_trim)*self.samp_rate
        self.time_dmolext_dt = (self.time_trim[:-1] + self.time_trim[1:])/2
        
    def kill_mol_steps_spikes_in_dxdt(self):
        # Robust center and MAD
        med,mad_value,gaussian_sd = mad(self.dmolext_dt)
        
        # Reject molecular transitions / large excursions
        good = np.abs(self.dmolext_dt - med) < self.mad_thresh * mad_value
        
        self.drift_raw_signal = -self.dmolext_dt[good]
        self.drift_raw_time = self.time_dmolext_dt[good]
    
    def estimate_drift_rates(self):
        coeffs = np.polyfit(
            self.drift_raw_time,
            self.drift_raw_signal,
            self.drift_model_order-1
        )
        
        self.drift_rates = np.polyint(coeffs)
    
    def estimate_drift(self):
        self.drift = np.polyval(self.drift_rates,self.time)
        
    def correct_molext(self):
        self.corrected_molext = self.molext + self.drift
        
    def run(self):
        self.smooth_molext()
        self.get_dxdt()
        self.kill_mol_steps_spikes_in_dxdt()
        self.estimate_drift_rates()
        self.estimate_drift()
        self.correct_molext()
    
def estimate_stiffness_error(fec_h5_file, min_force = 5, max_force = 20, plot = False, detrend_mobile_bead = True):
    # beads naming in h5 files lf camera are swapped
    # get downsampled trap position and cropping bead1x signal
    ds_trappos,bead_1x_mobile = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file["Bead position"]["Bead 2 X"])
    ds_trappos = ds_trappos.data
    bead_1x_mobile = bead_1x_mobile.data
    
    #detrending bead1x
    if detrend_mobile_bead:
        detrended_bead_1x_mobile = bead_1x_mobile - ds_trappos
    else:
        detrended_bead_1x_mobile = bead_1x_mobile
    
    # getting bead2x cropped signal
    ds_trappos,bead_2x_fixed = fec_h5_file["Trap position"]["1X"].downsampled_like(fec_h5_file["Bead position"]["Bead 1 X"])
    bead_2x_fixed =  bead_2x_fixed.data

    # get force 1x
    force1x, _ = fec_h5_file.force1x.downsampled_like(fec_h5_file["Bead position"]["Bead 1 X"])
    force1x = force1x.data

    #get force 2x
    force2x, _ = fec_h5_file.force2x.downsampled_like(fec_h5_file["Bead position"]["Bead 1 X"])
    force2x = force2x.data
    
    k1_psd = fec_h5_file.force1x.calibration[0].stiffness
    k2_psd  = fec_h5_file.force2x.calibration[0].stiffness
    
    mask =  (-force1x > min_force) & (force2x > min_force) & (-force1x < max_force) & (force2x < max_force)
    fit_k1 = np.polyfit(detrended_bead_1x_mobile[mask],force1x[mask],1)[0]/1000
    fit_k2 = np.polyfit(bead_2x_fixed[mask],force2x[mask],1)[0]/1000
    
    if plot:
        plt.plot(bead_2x_fixed[mask],force2x[mask])
        plt.plot(bead_2x_fixed[mask],np.polyval(np.polyfit(bead_2x_fixed[mask],force2x[mask],1),bead_2x_fixed[mask]))
        plt.show()
        plt.close()

    
    return(fit_k1/k1_psd, fit_k2/k2_psd)

    
def poly_baseline(y, x=None, degree=3, return_coeffs=False):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if x is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < degree + 1:
        raise ValueError("Not enough finite points to fit requested polynomial degree")

    # scale x for numerical stability
    x0 = x[m].mean()
    sx = x[m].std()
    if sx == 0:
        sx = 1.0
    xz = (x - x0) / sx

    coeffs = np.polyfit(xz[m], y[m], deg=degree)
    baseline = np.polyval(coeffs, xz)
    y_detrended = y - baseline

    if return_coeffs:
        return baseline, y_detrended, coeffs
    return baseline, y_detrended

def correct_trap_instability (ch_to_correct,ref_ch,degree=31):
    a,b = np.polyfit(ch_to_correct, ref_ch,1)
    r = ch_to_correct - (a*ref_ch+b)
    bsl, _ = poly_baseline(
            r,
            x=None,
            degree=degree,
            return_coeffs=False)
    corrected_ch = ch_to_correct-bsl
    return(r,bsl,corrected_ch)

def correct_compliance (trappos1x, 
                        force, 
                        pretrans_idx, 
                        trans_idx, 
                        state,
                        *WLCp,padding = 100, 
                        nendpoints = 800,
                        var_component = 0):
    WLCp = copy.deepcopy(WLCp)
    WLCp[var_component][1] = state
    print(WLCp)
    trappos1x_trim = trappos1x[:trans_idx+1]
    force_trim = force[:trans_idx+1]
    
    #plt.plot(trappos1x_trim,force_trim)
    #plt.show()
    
    # IMPORTANT: Make sure units match! If using trappos1x in um, then set eWLC
    # molecular extension in um (by replacing the correct value in contour lengt)!
    # or convert trappos1x to nm.
    #keq true will be pN/nm if trappos1x was in nm, or pN/um if in um
    if len(trappos1x_trim[pretrans_idx + padding:]) > nendpoints:
        xtether = ini_eWLC(1, force_trim[-nendpoints:], *WLCp)
        xtrue_approx = trappos1x_trim[-nendpoints:] - xtether
        fitpars = fitting.robust_polyfit(xtrue_approx, force_trim[-nendpoints:],1)
        keq_true = fitpars[0]
        plt.plot(xtrue_approx, force_trim[-nendpoints:])
        plt.plot(xtrue_approx, np.polyval(fitpars, xtrue_approx))
        plt.show()

    else:
        print("not enough points, using all available")
        xtether = ini_eWLC(1, force_trim[pretrans_idx+ padding:], *WLCp)
        xtrue_approx = trappos1x_trim[pretrans_idx+ padding:] - xtether
        fitpars = fitting.robust_polyfit(xtrue_approx, force_trim[pretrans_idx+ padding:],1)
        keq_true = fitpars[0]
        plt.plot(xtrue_approx, force_trim[pretrans_idx+ padding:])
        plt.plot(xtrue_approx, np.polyval(fitpars, xtrue_approx))

        plt.show()
    
    xtrue = force / keq_true # here xtrue in um  
    #alpha = np.dot(xtrue, force_wrong/keq_wrong/1000)/np.linalg.norm(force_wrong/keq_wrong/1000)**2 # over 1000 to convert as k is expressed in pN/nm
    #force_true = force_wrong/alpha
    #keq_true = m/alpha
    corr_molext = trappos1x - xtrue
    
    return(keq_true,corr_molext)

#%% MAIN CLASSES

class CorrectSpatialDrift:
    def __init__(self,
                 fec_h5_file,
                 fec_time,
                 molext,
                 drift_model_order=1,
                 pause_edge_trim=0.1,
                 camera_trappos_slope_factor=1,
                 channels_for_drift_estimation="1x",
                 only_positive_drift=True,
                 trappos_mode_threshold=0.001,
                 trappos_mode_min_run_len=5,
                 trappos_mode_min_velocity_threshold=9,
                 min_drift_velocity = -1e-6):

        self.fec_h5_file = fec_h5_file
        self.fec_time = np.asarray(fec_time)
        self.molext = np.asarray(molext)
        self.drift_model_order = drift_model_order
        self.pause_edge_trim = pause_edge_trim
        self.camera_trappos_slope_factor = camera_trappos_slope_factor
        self.channels_for_drift_estimation = channels_for_drift_estimation
        self.only_positive_drift = only_positive_drift
        self.trappos_mode_threshold = trappos_mode_threshold
        self.trappos_mode_min_run_len = trappos_mode_min_run_len
        self.trappos_mode_min_velocity_threshold = trappos_mode_min_velocity_threshold
        self.min_drift_velocity = min_drift_velocity

        self.k1 = None
        self.k2 = None
        self.ds_time = None
        self.bead_position1 = None
        self.bead_position2 = None
        self.ds_trappos1x = None
        self.ds_force1x = None
        self.ds_force2x = None
        self.vm_trap_center1 = None
        self.vm_trap_center2 = None
        self.vm_trap_centers_sep = None
        self.spatial_drift = None
        self.pause_edges = []
        self.passive = None
        self.passive_segment_ids = None
        self.drift_fit_coeffs = None
        self.segment_offsets = None
        self.spatial_drift_fit = None
        self.drift_velocity = None
        self.drift = None
        self.corrected_molext = None

    def extract_data(self):
        fec_h5_file = self.fec_h5_file

        self.k1 = fec_h5_file.force1x.calibration[0].stiffness
        self.k2 = fec_h5_file.force2x.calibration[0].stiffness

        ds_trappos1x, bead_position1 = fec_h5_file["Trap position"]["1X"].downsampled_like(
            fec_h5_file["Bead position"]["Bead 2 X"]
        )
        ds_trappos1x, bead_position2 = fec_h5_file["Trap position"]["1X"].downsampled_like(
            fec_h5_file["Bead position"]["Bead 1 X"]
        )
        ds_trappos1x, ds_force1x = fec_h5_file["Trap position"]["1X"].downsampled_like(
            fec_h5_file.downsampled_force1x
        )
        ds_trappos1x, ds_force2x = fec_h5_file["Trap position"]["1X"].downsampled_like(
            fec_h5_file.downsampled_force2x
        )

        self.ds_time = bead_position2.seconds
        self.bead_position1 = bead_position1.data
        self.bead_position2 = bead_position2.data
        self.ds_trappos1x = ds_trappos1x.data
        self.ds_force1x = ds_force1x.data
        self.ds_force2x = ds_force2x.data

    def compute_spatial_drift(self):
        trappos1x_camera = self.ds_trappos1x*self.camera_trappos_slope_factor

        if self.channels_for_drift_estimation == "both":
            self.vm_trap_center1 = self.bead_position1 - self.ds_force1x/self.k1/1000
            self.vm_trap_center2 = self.bead_position2 - self.ds_force2x/self.k2/1000
            self.vm_trap_centers_sep = self.vm_trap_center1 - self.vm_trap_center2
            self.spatial_drift = self.vm_trap_centers_sep - trappos1x_camera

        elif self.channels_for_drift_estimation == "1x":
            self.vm_trap_center1 = self.bead_position1 - self.ds_force1x/self.k1/1000
            self.spatial_drift = self.vm_trap_center1 - trappos1x_camera

        elif self.channels_for_drift_estimation == "2x":
            self.vm_trap_center2 = self.bead_position2 - self.ds_force2x/self.k2/1000
            self.spatial_drift = -self.vm_trap_center2

        else:
            raise ValueError("channels_for_drift_estimation must be 'both', '1x', or '2x'")

    def find_pause_edges(self):
        _, _, self.pause_edges = ana.split_trappos_modes(
            time=self.ds_time,
            trappos=self.ds_trappos1x,
            threshold=self.trappos_mode_threshold,
            min_run_len=self.trappos_mode_min_run_len,
            min_velocity_threshold=self.trappos_mode_min_velocity_threshold
        )

    def fit_spatial_drift(self):
        passive = np.zeros(len(self.ds_time),dtype=bool)
        segment_ids = np.full(len(self.ds_time),-1,dtype=int)
        n_segments = 0

        for pause_start,pause_end in self.pause_edges:
            mask = (
                (self.ds_time >= pause_start + self.pause_edge_trim)
                & (self.ds_time <= pause_end - self.pause_edge_trim)
            )

            if np.any(mask):
                passive = passive | mask
                segment_ids[mask] = n_segments
                n_segments += 1

        self.passive = passive
        self.passive_segment_ids = segment_ids

        fit_mask = passive & np.isfinite(self.ds_time) & np.isfinite(self.spatial_drift)

        if np.sum(fit_mask) <= self.drift_model_order + n_segments:
            raise ValueError("Not enough passive points to fit spatial drift")

        t0 = self.ds_time[0]
        tfit = self.ds_time[fit_mask] - t0
        yfit = self.spatial_drift[fit_mask]
        segfit = segment_ids[fit_mask]

        polynomial_terms = np.column_stack([
            tfit**p for p in range(1,self.drift_model_order+1)
        ])

        segment_terms = np.zeros((len(tfit),n_segments))

        for j in range(n_segments):
            segment_terms[:,j] = segfit == j

        X = np.column_stack((polynomial_terms,segment_terms))

        fitpars,_,_,_ = np.linalg.lstsq(X,yfit,rcond=None)

        self.drift_fit_coeffs = fitpars[:self.drift_model_order]
        self.segment_offsets = fitpars[self.drift_model_order:]

        t_ds = self.ds_time - t0
        self.spatial_drift_fit = np.zeros(len(self.ds_time))

        for p,a in enumerate(self.drift_fit_coeffs,start=1):
            self.spatial_drift_fit += a*t_ds**p

        t_fec = self.fec_time - t0
        self.drift = np.zeros(len(self.fec_time))
        self.drift_velocity = np.zeros(len(self.fec_time))

        for p,a in enumerate(self.drift_fit_coeffs,start=1):
            self.drift += a*t_fec**p
            self.drift_velocity += p*a*t_fec**(p-1)

        self.drift = self.drift/self.camera_trappos_slope_factor
        self.drift_velocity = self.drift_velocity/self.camera_trappos_slope_factor

        if self.only_positive_drift and np.any(self.drift_velocity < self.min_drift_velocity):
            self.drift[:] = 0

    def correct_molext(self):
        self.corrected_molext = self.molext + self.drift

    def plot_drift_fit(self):
        plt.plot(
            self.ds_time[self.passive],
            self.spatial_drift[self.passive],
            ".",
            markersize=4
        )

        for j,offset in enumerate(self.segment_offsets):
            mask = self.passive_segment_ids == j

            plt.plot(
                self.ds_time[mask],
                self.spatial_drift_fit[mask] + offset,
                color="black",
                linewidth=2
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Spatial drift (um)")
        plt.show()

    def run(self):
        self.extract_data()
        self.compute_spatial_drift()
        self.find_pause_edges()
        self.fit_spatial_drift()
        self.correct_molext()