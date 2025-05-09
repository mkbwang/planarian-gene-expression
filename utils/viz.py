#%%
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
color_gray="#808080"
color_red="#db382c"
color_green="#2f7028"
color_brown="#665223"
color_blue="#344885"
color_magenta="#b538b3"
all_colors = [color_blue,  color_red, color_green, color_brown, color_gray, color_magenta]



def single_line_plot(ymat, xmat=None, colors=None, linetypes=None,
                     xticks=None, xticknames=None, xname=None,
                     yticks=None, yticknames=None, yname=None,
                     vertical_lines = None,
                     horizontal_lines = None,
                     colors_map=None,
                     linetypes_map=None,
                     title=None, size=(6,4)):

    fig, ax = plt.subplots(figsize=size)
    nlines, _ = ymat.shape
    if colors is None:
        colors=all_colors

    # plot lines
    if linetypes is None:
        linetypes = ['-o' for _ in range(nlines)]
    if xmat is None:
        for j in range(nlines):
            ax.plot(xticks, ymat[j, :], linetypes[j], color=colors[j])
    else:
        for j in range(nlines):
            ax.plot(xmat[j, :], ymat[j, :], linetypes[j], color=colors[j])

    if vertical_lines is not None:
        for value in vertical_lines:
            ax.axvline(x=value, linestyle='--')
    if horizontal_lines is not None:
        for value in vertical_lines:
            ax.axhline(x=value, linestyle='--')

    # change x ticks
    if xticknames is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticknames)
    if yticknames is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticknames)

    if xname is not None:
        ax.set_xlabel(xname)
    if yname is not None:
        ax.set_ylabel(yname)

    if title is not None:
        ax.set_title(title)
    if colors_map is not None or linetypes_map is not None:

        colors_legend = []
        if colors_map is not None:
            for col_label in colors_map.keys():
                colors_legend.append(mlines.Line2D([], [], color=colors_map[col_label],
                                                   marker='', linestyle='-', label=col_label))
        linetypes_legend = []
        if linetypes_map is not None:
            for linetype_label in linetypes_map.keys():
                linetypes_legend.append(mlines.Line2D([], [],
                                                      color='black', linestyle=linetypes_map[linetype_label],
                                                      label=linetype_label))

        all_legends = colors_legend + linetypes_legend
        ax.legend(handles=all_legends, loc='upper left', bbox_to_anchor=(1, 1))


    return fig


def single_scatter_plot(ymat, xmat=None, colors=None,
                     xticks=None, xticknames=None, xname=None,
                     yticks=None, yticknames=None, yname=None,
                     legend_labels=None, diag_line=False,
                     title=None, size=(6,4)):
    fig, ax = plt.subplots(figsize=size)
    nlines, _ = ymat.shape
    if colors is None:
        colors = all_colors

    # plot lines
    if xmat is None:
        for j in range(nlines):
            if legend_labels is None:
                ax.scatter(xticks, ymat[j, :], color=colors[j], alpha=0.6)
            else:
                ax.scatter(xticks, ymat[j, :], color=colors[j], alpha=0.6, label=legend_labels[j])
    else:
        for j in range(nlines):
            if legend_labels is None:
                ax.scatter(xmat[j, :], ymat[j, :], color=colors[j], alpha=0.6)
            else:
                ax.scatter(xmat[j, :], ymat[j, :], color=colors[j], alpha=0.6, label=legend_labels[j])

    # change x ticks
    if xticknames is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticknames)
    if yticknames is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticknames)

    if xname is not None:
        ax.set_xlabel(xname)
    if yname is not None:
        ax.set_ylabel(yname)

    if diag_line:
        min_x = np.min(xmat) if xmat is not None else np.min(xticks)
        max_x = np.max(xmat) if xmat is not None else np.max(xticks)
        for j in range(nlines):
            ax.plot([min_x, max_x], [min_x, max_x], linestyle="--")

    if title is not None:
        ax.set_title(title)
    if legend_labels is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    return fig


#TODO: multipanel line plot, single panel scatter plot, single panel lineplot
def multi_scatter_plot(ymat, xmat=None, color=None,
                     xticks=None, xticknames=None, xrange=None, xname=None,
                     yticks=None, yticknames=None, yrange=None, yname=None,
                     titles=None, diag_line=False, size=(6,4)):

    nlines, _ = ymat.shape
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=nlines)
    if color is None:
        color=all_colors[0]

    # scatter plot
    if xmat is None:
        for j in range(nlines):
            ax[j].scatter(xticks, ymat[j, :], color=color, alpha=0.6)
    else:
        for j in range(nlines):
            ax[j].scatter(xmat[j,:], ymat[j, :], color=color, alpha=0.6)

    # add xlabel and ylabel and title
    if xname is not None:
        for j in range(nlines):
            ax[j].set_xlabel(xname)

    if yname is not None:
        for j in range(nlines):
            ax[j].set_ylabel(yname)

    if xticknames is not None:
        for j in range(nlines):
            ax[j].set_xticks(xticks)
            ax[j].set_xticklabels(xticknames)

    if yticknames is not None:
        for j in range(nlines):
            ax[j].set_yticks(yticks)
            ax[j].set_yticklabels(yticknames)

    if titles is not None:
        for j in range(nlines):
            ax[j].set_title(titles[j])

    if diag_line:
        min_x = np.min(xmat) if xmat is not None else np.min(xticks)
        max_x = np.max(xmat) if xmat is not None else np.max(xticks)
        for j in range(nlines):
            ax[j].plot([min_x, max_x], [min_x, max_x], linestyle="--")

    return fig


