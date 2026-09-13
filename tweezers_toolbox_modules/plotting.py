#%% IMPORT GENERAL MODULES

import matplotlib.colors as mc
import colorsys, sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

#%% IMPORT CUSTOM MODULES

sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.models import ini_eWLC
from matplotlib.gridspec import GridSpec

#%%

def plot_histogram(
    data,
    xlabel,
    ylabel,
    bins="sturges",
    density=True,
    xlim=None,
    figsize=(5, 4),
    label="Experimental data",
):
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        np.asarray(data).ravel(),
        bins=bins,
        density=density,
        color="gray",
        edgecolor="none",
        alpha=0.8,
        label=label,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=1,
        labelsize=10,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    if xlim is not None:
        ax.set_xlim(xlim)

    return fig, ax

def plot_Lc_colored_by_force(Lc, F, x=None, cmap='jet'):
    if x is None:
        x = np.arange(len(Lc))

    # Create segments
    points = np.array([x, Lc]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize force for colormap
    norm = Normalize(vmin=np.min(F), vmax=np.max(F))

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(F[:-1])   # color by force
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(Lc.min(), Lc.max())

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Force (pN)")

    ax.set_xlabel("Time (index)")
    ax.set_ylabel("Contour Length (nm)")

    plt.show()
    
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

def adjust_lightness(color, amount=0.5):

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def quick_plotter (indep, depend, fit, xtitle, ytitle, filename,alph, annotations = "",
                   guideline = False,
                   save = False):

    if isinstance(fit, list) or isinstance(fit, np.ndarray):
        if isinstance(indep, list):
            series_num = len(indep)
            colors = plt.cm.jet(np.linspace(0,1,series_num))
            plt.rcParams['figure.dpi'] = 300

            for x, y, f, col_index in zip (indep, depend, fit, range(series_num)):
                plt.scatter(x,y, s = 5, color = colors[col_index], alpha = alph)
                plt.plot(x, f, color = adjust_lightness(colors[col_index], amount=1.5))
        else:
            plt.scatter(indep,depend, s = 10, color = "black", alpha = alph)
            plt.plot(indep, fit, color = "black")

    else:
        if isinstance(indep, list):
            series_num = len(indep)
            colors = plt.cm.jet(np.linspace(0,1,series_num))
            plt.rcParams['figure.dpi'] = 300

            for x, y, col_index in zip (indep, depend, range(series_num)):
                plt.scatter(x,y, s = 5, color = colors[col_index], alpha = alph)
        else:
            plt.scatter(indep,depend, s = 10, color = "black", alpha = alph)

    if annotations:
        label_hpos = 0.05
        if isinstance(annotations, list):
            for i in annotations:
                plt.annotate(i,
                             xy = (0.25,label_hpos),
                             xycoords = "axes fraction",
                             fontsize=9)
                label_hpos = label_hpos + 0.05
        else:
            plt.annotate(annotations,
                         xy = (0.25,label_hpos),
                         xycoords = "axes fraction",
                         fontsize=9)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if guideline:
        plt.axhline(0, color = "Black", linestyle = "--", linewidth = 2)

    if save:
        plt.savefig(filename)

    plt.show()
    plt.close()

    return

def plot_individual_fec(distances,
                         forces,
                         fec_id,
                         data_color,
                         xlim,
                         ylim,
                         title,
                         *WLCpars,
                         var_component=0,
                         distances_hf=False,
                         forces_hf=False,
                         scatter=False,
                         annot=False,
                         savefig=False,
                         rupture_index=False,
                         curve_type = "S",
                         lcstates=False,
                         states_end_indexes=False,
                         lc_trajectory=False, 
                         xtickspace = 0.2):

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Set axis limits
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(np.arange(xlim[0], xlim[1], xtickspace))   # every 2 units on x-axis
    plt.yticks(np.arange(ylim[0], ylim[1], 5))   # every 2 units on x-axis

    
    plt.xlabel("Position (um)")
    plt.ylabel("Force (pN)")

    # Plot hf distances and forces if available
    if isinstance(distances_hf, (list, np.ndarray)):
        plt.scatter(distances_hf, forces_hf, s=5, alpha=0.15, color="gray")

        if rupture_index:
            if curve_type == "S":
                trimmed_distances = distances_hf[rupture_index + 1:]
                trimmed_forces = forces_hf[rupture_index + 1:]
            elif curve_type == "R":
                trimmed_distances = distances_hf[:rupture_index]
                trimmed_forces = forces_hf[:rupture_index]
                
            else:
                trimmed_distances = distances_hf[rupture_index + 1:]
                trimmed_forces = forces_hf[rupture_index + 1:]
                
                
            plt.scatter(trimmed_distances, trimmed_forces, s=5, alpha=0.15,
                        color="salmon", label="trimmed")

    # Plot lc states if available
    if isinstance(lcstates,(list,np.ndarray)):
        predicted_forces = np.arange(0.01, 60, 0.01)
        cmap = plt.get_cmap('tab20')  # Better than default, avoids gray early
        for j,i in enumerate(lcstates):
            color = cmap(j % 20)
            WLCpars_modified = [list(p) for p in WLCpars]
            WLCpars_modified[var_component][1] = i
            predicted_distances = ini_eWLC(1, predicted_forces, *WLCpars_modified)
            plt.plot(predicted_distances, predicted_forces, "--",
                     label="Lc = " + str(round(i, 3)) + " um", color = color)

    # Plot scatter or line based on input
    if scatter:
        plt.scatter(distances, forces, color=data_color, s=5, label=fec_id)
    else:
        plt.plot(distances, forces, color=data_color, label=fec_id)

    # Annotate if specified
    if annot:
        plt.annotate(annot,
                     xy=(0.98, 0.98),
                     xycoords='axes fraction',
                     ha='right',
                     va='top',
                     fontsize = 10)

    # Plot states end indexes if available
    if isinstance(states_end_indexes,(list,np.ndarray)):
        cmap = plt.get_cmap('tab20')  # Better than default, avoids gray early
        for j, i in enumerate(states_end_indexes):
            color = cmap(j % 20)
            plt.scatter(distances[i], forces[i], s=25,  # slightly larger
                        color=color, edgecolors='black', linewidths=0.5)

    # Plot lc trajectory if available
    if isinstance(lc_trajectory,(list,np.ndarray)):
        plt.plot(distances, lc_trajectory * 100 - 50, color="salmon", alpha=0.9)
        for i in lcstates:
            plt.axhline(i * 100 - 50, color="black", alpha=0.6)

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3)

    # Adjust layout to ensure legend and labels fit
    plt.tight_layout()
    plt.grid(True)

    # Save figure if specified
    if savefig:
        plt.savefig(title, dpi=300)

    # Display the plot
    plt.show(block=True)

def plot_individual_motor_trace(
    t,
    lctrace,
    ideal_trace,
    xlim,
    ylim,
    xtickspace,
    ytickspace,
    trace_id,
    lctrace_averaged=None,
    savefig=False,
    show=True,
    pause_intervals = None,
    path = None,
    
):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Set axis limits
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(np.arange(xlim[0], xlim[1], xtickspace))
    plt.yticks(np.arange(ylim[0], ylim[1], ytickspace))

    plt.xlabel("time (s)")
    plt.ylabel("Basepairs sled")

    plt.scatter(
        t, lctrace,
        s=5,
        alpha=0.15,
        color="gray"
    )
    
    if isinstance(pause_intervals, (list, np.ndarray)):
        for start, end in pause_intervals:
            plt.scatter(
                t[start:end+1], lctrace[start:end+1],
                s=5,
                alpha=0.15,
                color="salmon"
            )
        

    if isinstance(lctrace_averaged, (list, np.ndarray)):
        plt.plot(
            t,
            lctrace_averaged,
            label=trace_id,
            color="black",
            linewidth=0.5
        )

    plt.plot(
        t,
        ideal_trace,
        color="red",
        linewidth=0.8
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3
    )

    plt.tight_layout()
    plt.grid(True)

    if savefig:
        plt.savefig(f"{path}/{trace_id}.png", dpi=300)

    if show:
        plt.show(block=True)

    return fig, ax
    
    
    
    
    