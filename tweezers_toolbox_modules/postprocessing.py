#%% IMPORT EXTERNAL MODULES

import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import signal
from matplotlib.widgets import RectangleSelector

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.models import ini_eWLC
from tweezers_toolbox_modules import input_handlers as hand
from tweezers_toolbox_modules import fec_plotting as fplt

#%% BASIC FUNCTIONS


def find_tether_rupture (forces, basepoints, force_threshold,smooth_factor, curvetype = "stretch",
                         maxforce = False):

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

    if maxforce:
        flipped_f_smooth_trimmed = flipped_f_smooth[rupture_index:len(flipped_f_smooth)]

        for c,i in enumerate(flipped_f_smooth_trimmed):
            if i <= maxforce:

                break

        rupture_index = rupture_index + c

    # Once a first-refined point is selected, then transform it to the unflipped
    # index.
    if curvetype == "stretch":
        rupture_index = len(forces) - 1 - rupture_index

    if not maxforce:
        # If there is a point after the temptative rupture index
        # that is higher in force, then take that one as the new rupture.
        rupture_index = np.argmax(forces[rupture_index:len(forces)]) + rupture_index


    return(rupture_index)

#%% MAIN CLASSES

class AlignFEC:
    def __init__(self, x, F, distance_aln_bounds, force_aln_bounds, *WLCpars,
                 custom_function = False, distance_aln_indexes = None,
                 respect_trans_ind = None):
        self.x = x
        self.F = F
        self.distance_aln_bounds = distance_aln_bounds
        self.force_aln_bounds = force_aln_bounds
        self.WLCpars = WLCpars
        self.custom_function = custom_function
        self.x_aln = x
        self.F_aln = F
        self.distance_aln_indexes = distance_aln_indexes
        self.respect_trans_ind = respect_trans_ind

    def dist_aln (self):
        if self.distance_aln_indexes:
            #print(self.distance_aln_indexes)
            x_slice = self.x_aln[self.distance_aln_indexes[0]:self.distance_aln_indexes[1]+1]
            F_slice = self.F_aln[self.distance_aln_indexes[0]:self.distance_aln_indexes[1]+1]
        else:
            distance_range = self.distance_aln_bounds[0]
            force_range = self.distance_aln_bounds[1]
            if self.respect_trans_ind:
                x_slice = self.x_aln[:self.respect_trans_ind + 1]
                F_slice = self.F_aln[:self.respect_trans_ind + 1]
                
                mask = (x_slice > distance_range[0]) & (x_slice < distance_range[1]) & \
                       (F_slice > force_range[0]) & (F_slice < force_range[1])
                
                x_slice = x_slice[mask]
                F_slice = F_slice[mask]
            else:
                mask = (self.x_aln > distance_range[0]) & (self.x_aln < distance_range[1]) & \
                       (self.F_aln > force_range[0]) & (self.F_aln < force_range[1])
                x_slice = self.x_aln[mask]
                F_slice = self.F_aln[mask]

        if self.custom_function:
            target_distances = self.custom_function(F_slice)
        else:
            target_distances = ini_eWLC(1, F_slice, *self.WLCpars)

        doff = np.mean(target_distances - x_slice)
        
        if not np.isnan(doff):
            self.x_aln = self.x_aln + doff

    def force_aln (self):
        distance_range = self.force_aln_bounds[0]
        force_range = self.force_aln_bounds[1]
        
        if self.respect_trans_ind:
            x_slice = self.x_aln[:self.respect_trans_ind + 1]
            F_slice = self.F_aln[:self.respect_trans_ind + 1]
            
            mask = (x_slice > distance_range[0]) & (x_slice < distance_range[1]) & \
                   (F_slice > force_range[0]) & (F_slice < force_range[1])
            
            x_slice = x_slice[mask]
            F_slice = F_slice[mask]
        
        else:
            mask = (self.x_aln > distance_range[0]) & (self.x_aln < distance_range[1]) & \
                   (self.F_aln > force_range[0]) & (self.F_aln < force_range[1])
            x_slice = self.x_aln[mask]
            F_slice = self.F_aln[mask]

        if self.custom_function:
            target_forces = self.custom_function(F_slice)
        else:
            target_forces = ini_eWLC(0, x_slice, *self.WLCpars)

        foff = np.mean(target_forces - F_slice)
        
        if not np.isnan(foff):
            self.F_aln = self.F_aln + foff

    def distance_force_aln (self):
        # Distance alignment 1
        self.dist_aln ()

        # Force alignment
        self.force_aln ()

    def iterative_dist_aln (self, distance_convergence = 1e-5, max_iter = 20):

        doff = 10000

        i = 0

        while abs(doff) > distance_convergence:

            if i < max_iter:
                prev_distances = self.x_aln
                self.dist_aln()
                doff = (self.x_aln - prev_distances)[0]

            else:
                break

            i += 1
            print(f'distance offset = {doff}')

    def iterative_fd_aln (self, distance_convergence = 1e-5,
                            force_convergence = 1e-3, max_iter = 20):

        doff = 10000
        foff = 10000

        i = 0
        print(doff)
        while (abs(doff) > distance_convergence) | ((abs(foff) > force_convergence)):

            if i < max_iter:
                prev_distances = self.x_aln
                prev_forces = self.F_aln
                self.distance_force_aln()
                doff = (self.x_aln - prev_distances)[0]
                foff = (self.F_aln - prev_forces)[0]

            else:
                break

            i += 1
            print(f'distance offset = {doff} \t force offset = {foff}')


class Curate_FECPostprocessing:
    def __init__(self, fec,d_aln, f_aln, dlf_aln, flf_aln,
                 rupture_index, discard,plot_bounds,max_force, *WLCpars, curve_type = "S",
                 var_component = 0, plot_xticks_space = 0.2, plot_yticks_space = 5):
        
        # Initialize the necessary data for the single fec
        self.fec = fec
        self.WLCpars = WLCpars
        self.d_aln = d_aln
        self.f_aln = f_aln
        self.dlf_aln = dlf_aln
        self.flf_aln = flf_aln
        self.plot_bounds = plot_bounds
        self.rupture_index = rupture_index
        self.discard = discard
        self.curve_type = curve_type
        self.max_force = max_force
        self.var_component = var_component
        self.plot_xticks_space = plot_xticks_space
        self.plot_yticks_space = plot_yticks_space

    def modify_rupture_index(self):
        # Original values of atributes
        ori = self.rupture_index
        
        while True:
            print(f"Current rupture index is {ori}")
            self.rupture_index = hand.is_valid_value (f"Please enter a new rupture index (0-{len(self.d_aln)-1}): ",
                                                                  int,
                                                                  min_value=0,
                                                                  max_value=len(self.d_aln)-1)
            if self.curve_type == "S":
                while (self.f_aln[self.rupture_index] > self.max_force) and self.rupture_index > 0:
                    self.rupture_index = self.rupture_index - 1
            elif self.curve_type == "R":
                while (self.f_aln[self.rupture_index] > self.max_force) and (self.rupture_index < len(self.d_aln)-1):
                    self.rupture_index = self.rupture_index + 1
            else:
                print("Invalid curve type.")
                break
                
            self._plot_fec()

            user_inp = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_inp == "n":
                self.rupture_index = ori
            elif user_inp == "q":
                self.rupture_index = ori
                break
            else:
                break
            
    def modify_alignment(self):
        # Testing values
        while True:
            while True:
                print("Please define bounds for distance alignment.")
                success, distaln_bounds = self._define_fd_bounds_interactive()

                # Esc, Enter without a selection, or closing the selector means
                # "cancel realignment". Keep the current alignment and return to
                # run(), which proceeds to the kw/r/c prompt.
                if not success:
                    print("Realignment cancelled. Keeping current alignment.")
                    return

                if success:
                    # Realign and redo rmsd filtering
                    # Force alignment option removed.
                    iter_aln_fec = AlignFEC(self.dlf_aln, self.flf_aln,
                                            distaln_bounds, [],
                                            *self.WLCpars,
                                            custom_function = False)
                    iter_aln_fec.iterative_dist_aln()

                    if not (np.isnan(iter_aln_fec.x_aln[0]) or np.isnan(iter_aln_fec.F_aln[0])):
                        break
                    else:
                        user_inp =  hand.option_handler("Alignment failed, please give new ranges (nr) or skip (s)? ",
                                                        valid_values=["nr", "s"])
                        if user_inp == "s":
                            return

            # Store original values of attributes
            od_aln = self.d_aln
            of_aln = self.f_aln
            odlf_aln = self.dlf_aln
            oflf_aln = self.flf_aln

            ## replace new values
            self.dlf_aln = iter_aln_fec.x_aln
            self.flf_aln = iter_aln_fec.F_aln

            ## Get offset
            doff = (self.dlf_aln - odlf_aln)[0]
            foff = (self.flf_aln - oflf_aln)[0]

            ## Apply offset to HF attribute
            if self.d_aln is not None:    
                self.d_aln = self.d_aln + doff
            
            if self.f_aln is not None:
                self.f_aln = self.f_aln + foff

            # Plot to check chandges
            self._plot_fec()  # Re-display the figure with adjusted data
            #print(f"New RMSD = {round(self.post_fec_obj, 3)}")

            # Confirmation
            user_inp = hand.option_handler("Satisfied (y/n) or quit without applying changes (q)? ",
                                             valid_values=["y", "n", "q"])

            if user_inp == "n":
                self.d_aln = od_aln
                self.f_aln = of_aln
                self.dlf_aln = odlf_aln
                self.flf_aln = oflf_aln
                
            elif user_inp == "q":
                self.d_aln = od_aln
                self.f_aln = of_aln
                self.dlf_aln = odlf_aln
                self.flf_aln = oflf_aln
                break
            else:
                break

    def run (self):
        self._plot_fec()
        state = "kept" if self.discard == 0 else "discarded"
        print(f"Now working on {self.fec}. This fec is {state}.")

        user_inp =  hand.option_handler("Modify trimming (mt), realign (a), keep (k), discard (d), return to previous (r)? ",
                                        valid_values = ["mt","a","k","d","r"])

        if user_inp == "mt":
            self.modify_rupture_index()
        elif user_inp == "a":
            self.modify_alignment()
        elif user_inp == "k":
            self.discard = 0
            return("c")
        elif user_inp == "d":
            self.discard = 5
            return("c")
        elif user_inp == "r":
            return("r")
            
        user_input = hand.option_handler("Keep working on the same fec (kw), return to previous (r) or continue to the next (c)? ",
                                          valid_values=["kw", "r", "c"])
        return(user_input)

    # Helper methods
    def _plot_fec(self):
        # Set the color based on whether it's discarded or not
        fec_color = "black" if self.discard == 0 else "red"

        fplt.plot_individual_fec(self.dlf_aln,
                                 self.flf_aln,
                                 self.fec,
                                 fec_color,
                                 self.plot_bounds[0],
                                 self.plot_bounds[1],
                                 "",
                                 *self.WLCpars,
                                 var_component = self.var_component,
                                 distances_hf=self.d_aln,
                                 forces_hf=self.f_aln,
                                 scatter=False,
                                 annot=False,
                                 savefig=False,
                                 rupture_index=self.rupture_index,
                                 curve_type = self.curve_type,
                                 lcstates=[self.WLCpars[self.var_component][1]],
                                 states_end_indexes=False,
                                 lc_trajectory=False,
                                 xtickspace=self.plot_xticks_space,
                                 ytickspace=self.plot_yticks_space)
        
    def _define_fd_bounds (self):
        min_distance = hand.is_valid_value ("Minimum distance: ",
                                            float,
                                            min_value=min(self.dlf_aln),
                                            max_value=max(self.dlf_aln))
        max_distance = hand.is_valid_value ("Maximum distance: ",
                                            float,
                                            min_value=min(self.dlf_aln),
                                            max_value=max(self.dlf_aln))
        min_force = hand.is_valid_value ("Minimum force: ",
                                            float,
                                            min_value=min(self.flf_aln),
                                            max_value=max(self.flf_aln))
        max_force = hand.is_valid_value ("Maximum force: ",
                                            float,
                                            min_value=min(self.flf_aln),
                                            max_value=max(self.flf_aln))

        # Testing ranges
        valid_dist_range = hand.is_valid_range ([min_distance, max_distance])
        valid_force_range = hand.is_valid_range ([min_force, max_force])
        success = False
        fd_bounds = None

        if all([valid_dist_range, valid_force_range]):
            success = True
            fd_bounds = [[min_distance, max_distance], [min_force, max_force]]

        return (success, fd_bounds)
    
    def _define_fd_bounds_interactive(self):
        """
        Draw a rectangle around the region to use for distance alignment.
    
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
            # Switch to Qt only when alignment is requested.
            try:
                plt.switch_backend("QtAgg")
            except Exception:
                try:
                    plt.switch_backend("Qt5Agg")
                except Exception as e:
                    print("Could not switch to Qt backend for interactive alignment.")
                    print(f"Current backend: {previous_backend}")
                    print(f"Error: {e}")
                    return False, None
    
            fig, ax = plt.subplots()
    
            # Full trace
            ax.plot(self.dlf_aln, self.flf_aln, lw=1.5, color="black")
    
            # Selected part of trace, updated after box selection
            selected_trace, = ax.plot([], [], lw=1.5, color="red")
    
            ax.set_xlabel("Distance")
            ax.set_ylabel("Force")
            ax.set_title(
                "Draw a box around the distance-alignment region.\n"
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
    
                d = np.asarray(self.dlf_aln)
                f = np.asarray(self.flf_aln)
    
                mask = (
                    (d >= min_distance) &
                    (d <= max_distance) &
                    (f >= min_force) &
                    (f <= max_force)
                )
    
                n_selected = np.sum(mask)
    
                if n_selected == 0:
                    selected["bounds"] = None
                    selected_trace.set_data([], [])
                    print(f"Raw selected bounds: {raw_bounds}")
                    print("No data points inside selected box.")
                    fig.canvas.draw_idle()
                    return
    
                selected_d = d[mask]
                selected_f = f[mask]
    
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
                # Fallback for older Matplotlib versions that do not accept handle_props.
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
            # This runs even if the function errors or is interrupted.
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
    
            # Switch back to whatever backend was active before alignment.
            try:
                plt.switch_backend(previous_backend)
            except Exception:
                # Spyder/IPython fallback: return to inline plotting if direct restore fails.
                try:
                    from IPython import get_ipython
                    ip = get_ipython()
                    if ip is not None:
                        ip.run_line_magic("matplotlib", "inline")
                except Exception:
                    pass
    
        if fd_bounds is None:
            print("Interactive selection cancelled.")
            return False, None
    
        # Testing ranges
        valid_dist_range = hand.is_valid_range(fd_bounds[0])
        valid_force_range = hand.is_valid_range(fd_bounds[1])
        success = False
    
        if all([valid_dist_range, valid_force_range]):
            success = True
            print(f"Selected bounds: {fd_bounds}")
    
        return (success, fd_bounds)
