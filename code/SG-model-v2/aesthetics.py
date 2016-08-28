import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    try:
        units = old_label[old_label.index("(") + 1:old_label.rindex(")")]
    except ValueError:
        units = ""
    label = old_label.replace("({})".format(units), "")
    exponent_text = exponent_text.replace("$\\times$", "")
    return "{} ({} {})".format(label, exponent_text, units)


def pretty_plot(fig, ax):
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_style("white")
    white_out(fig)
    sns.despine(offset=10, trim=True)
    pretty_label(ax)


def autolabel(rects):
    ''' Put labels on top of rectangles. '''
    for rect in rects:
        height = rect.get_height()
        i = rects.index(rect)
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '{:.4f} $\pm$ {:.4f}'.format(height, sems[i]),
                 ha='center', va='bottom')


def adjust_spines(ax, spines, plot_margin=0):
    ''' Inspired by Tufte-like axis limits. '''
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    x0, x1, y0, y1 = ax.axis()
    ax.axis((x0 - plot_margin,
             x1 + plot_margin,
             y0 - plot_margin,
             y1 + plot_margin))


def white_out(fig, facecolor='white'):
    ''' Make a white background on graphs, for better copy-paste functionality to Powerpoint.'''
    # See http://stackoverflow.com/questions/24542610/matplotlib-figure-facecolor-alpha-while-saving-background-color-transparency
    from matplotlib.colors import colorConverter
    if facecolor is False:
        # Not all graphs get color-coding
        facecolor = fig.get_facecolor()
        alpha = 1
    else:
        alpha = 0.5
    color_with_alpha = colorConverter.to_rgba(facecolor, alpha)
    fig.patch.set_facecolor(color_with_alpha)


def pretty_label(ax, axis='both'):
    ''' Format the label string with the exponent from the ScalarFormatter '''
    try:
        ax.xaxis
    except:
        ax = plt.gca()

    ax.ticklabel_format(axis=axis, style='sci')
    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw()  # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))


def plot_scan(xx, yy, zz, xlabel, ylabel, title, xlog=True, ylog=True):
    ''' Plot a 2D parameter scan with x, y, and z values.'''
    fig, ax = plt.subplots()
    if ylog == True:
        ax.set_yscale('log')
    if xlog == True:
        ax.set_xscale('log')
    ax.margins(x=0, y=0)
    cmap = mpl.cm.jet
    zz = np.array(zz)
    if ylog == True:
        im = ax.pcolor(xx, yy, zz,
                       norm=LogNorm(vmin=zz.min(), vmax=zz.max()),
                       cmap=cmap)
    else:
        im = ax.pcolor(xx, yy, zz,
                       cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    ax.set_title(title, y=1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pretty_label(ax)
    plt.show()


def fetching_plot(fig, adjustment=0):
    sns.set()
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_style("white")
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{helvet}',
        r'\usepackage{sansmath}',
        r'\sansmath',
        r'\renewcommand{\familydefault}{\sfdefault}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{graphicx}',
        r'\usepackage{upgreek}',
    ]
    for ax in fig.axes:
        ax.tick_params(which='major', direction='out', length=10)
        ax.tick_params(which='minor', direction='out', length=5)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        white_out(fig)
        if ax.xaxis.get_scale() == 'linear':
            pretty_label(ax)
            ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        elif ax.xaxis.get_scale() == 'log':
            pass
        # For scatter plots, where points get cut off
        if adjustment != 0:
            x0, x1, y0, y1 = ax.axis()
            ax.xaxis((x0 - adjustment,
                    x1 + adjustment,
                    ))
