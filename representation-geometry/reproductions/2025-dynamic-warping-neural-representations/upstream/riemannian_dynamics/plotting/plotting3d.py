from .plotting2d import *
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_manifold_functional(ax, xs, zs, center=False, cmap=manifold_cmap, zorder=2, cla=True):

    if cla: ax.cla()

    colors = cmap((zs - np.min(zs)) / (np.max(zs) - np.min(zs)) if center else zs)

    ax.plot_surface(xs[..., 0], xs[..., 1], xs[..., 2],
                    facecolors=colors,
                    linewidth=0.02,
                    ccount=xs.shape[0], rcount=xs.shape[1],
                    shade=False,
                    zorder=zorder)


def plot_with_gradient_3d(ax, xs, ys, zs, gradient, cmap,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          linewidth=1.0, alpha=1.0, zorder=3, set_lim=True):

    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    number_segments = len(segments)

    if dashed:
        temp = []
        temp_cols = []
        for i in range(number_dashes):
            temp.append(segments[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
            temp_cols.append(gradient[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])

        segments = np.concatenate(temp)
        gradient = np.concatenate(temp_cols)

    capstyle = 'round'# if dashed or alpha==1 else 'butt'
    lc = Line3DCollection(segments, cmap=cmap, alpha=alpha, zorder=zorder, capstyle=capstyle)
    lc.set_array(gradient)
    lc.set_clim(0,1)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    if set_lim:
        ax.set_xlim(min((np.min(xs), ax.get_xlim()[0])), max((np.max(xs), ax.get_xlim()[1])))
        ax.set_ylim(min((np.min(ys), ax.get_ylim()[0])), max((np.max(ys), ax.get_ylim()[1])))
        ax.set_zlim(min((np.min(zs), ax.get_zlim()[0])), max((np.max(zs), ax.get_zlim()[1])))


def plot_trajectories(ax, xs, condition, trials_plotted=None, cmap=trajectory_cmap, alpha=1.0, linestyle='-', linewidth=1.0, zorder=3, cla=True):

    if cla: ax.cla()

    trials_plotted = xs.shape[0] if trials_plotted is None else trials_plotted

    for trial in range(0, xs.shape[0], xs.shape[0]//trials_plotted):
        ax.plot(xs[trial, :, 0], xs[trial, :, 1], xs[trial, :, 2], zorder=zorder,
                color=cmap(condition[trial]), alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        '''ax.scatter(xs[trial, :, 0], xs[trial, :, 1], xs[trial, :, 2], zorder=zorder+1,
                color=cmap(trial/xs.shape[0]), alpha=0.3)'''

    #utils.set_equal_lim(ax)
    ax.set_box_aspect((ax.get_xlim()[1]-ax.get_xlim()[0],
                       ax.get_ylim()[1]-ax.get_ylim()[0],
                       ax.get_zlim()[1]-ax.get_zlim()[0]))

    ax.set_xlabel(r'$\mathregular{x_1}$'), ax.set_ylabel(r'$\mathregular{x_2}$'), ax.set_zlabel(r'$\mathregular{x_3}$')


def match_axes_lim(ax, ax_template):

    ax.set_xlim(*ax_template.get_xlim())
    ax.set_ylim(*ax_template.get_ylim())
    ax.set_zlim(*ax_template.get_zlim())


def set_box_aspect(ax):
    ax.set_box_aspect((ax.get_xlim()[1] - ax.get_xlim()[0],
                       ax.get_ylim()[1] - ax.get_ylim()[0],
                       ax.get_zlim()[1] - ax.get_zlim()[0]))



def plot_manifold(ax, xs, cmap=manifold_cmap,
                       min_alpha=0.0, max_alpha=0.5, 
                        min_alpha_grid=0.0, max_alpha_grid=1.0,
                  linewidth=0.02, edgecolor=(0.2, 0.2, 0.2),
                        gridlines=True,
                       ccount=6, rcount=6,
                       zorder=2, scatter=False, cla=True):

    if cla: ax.cla()

    #min_alpha, max_alpha = 0.0, 0.5
    alpha_decay = 1 - np.exp(np.linspace(-5, 0, xs.shape[0]))
    alpha_decay = normalize(alpha_decay, min_alpha, max_alpha)

    gradient_ts = np.linspace(0, 1, xs.shape[1])
    colors = cmap(np.tile(gradient_ts, (xs.shape[0], 1)))
    colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1)).T
    #colors[..., 3] = 0.2

    # ===== Surface =====
    '''ax.plot_surface(xs[..., 0], xs[..., 1], xs[..., 2],
                    facecolors=colors,
                    linewidth=0.02,
                    ccount=xs.shape[0], rcount=xs.shape[1],
                    shade=False,
                    zorder=zorder)'''

    if scatter:
        colors = cmap(np.tile(gradient_ts, (xs.shape[0], 1))).reshape(-1, 4)
        #colors[..., 3] = 0.1
        ax.scatter(xs[..., 0].reshape(-1), xs[..., 1].reshape(-1), xs[..., 2].reshape(-1),
                        c=colors,
                        edgecolor=None,
                        s=100,
                        zorder=zorder)

    else:
        ax.plot_surface(xs[..., 0], xs[..., 1], xs[..., 2],
                        facecolors=colors,
                        linewidth=linewidth,
                        ccount=xs.shape[0], rcount=xs.shape[1],
                        shade=False,
                        #alpha=0.1,
                        zorder=zorder)

    if gridlines:
        # ===== Grid lines =====
        #min_alpha_grid, max_alpha_grid = 0.0, 1.0
        alpha_decay = 1 - np.exp(np.linspace(-5, 0, xs.shape[0]))
        alpha_decay = normalize(alpha_decay, min_alpha, max_alpha)

        #colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1))

        #cmap = utils.set_saturation(cmap, 0.9)
        gradient_ts = np.linspace(0, 1, xs.shape[1])
        colors = cmap(np.tile(gradient_ts, (xs.shape[0], 1)))
        colors[..., 3] = np.tile(alpha_decay, (xs.shape[1], 1)).T

        cmap_alpha = utils.get_cmap_interpolated(*colors[0, :])
        # ====== Lines =====
        for i in range(0, xs.shape[0], xs.shape[0]//rcount):
            plot_with_gradient_3d(ax, xs[i, :, 0], xs[i, :, 1], xs[i, :, 2],
                                        gradient=gradient_ts, cmap=cmap_alpha, set_lim=False, zorder=zorder+0.5, alpha=None)

        #cmap_alpha = utils.get_cmap_interpolated(*colors[0, :])
        # ====== Circle =====
        for i in range(0, xs.shape[1], xs.shape[1]//ccount):
            #ax.plot(xs[:, i, 0], xs[:, i, 1], xs[:, i, 2], zorder=zorder+0.5, color=colors[0, i])
            cmap_ = riemannian_dynamics.plotting.utils.get_cmap_interpolated(colors[0, i], colors[0, i])
            plot_with_gradient_3d(ax, xs[:, i, 0], xs[:, i, 1], xs[:, i, 2],
                                        gradient=np.ones(xs.shape[0]), cmap=cmap_, set_lim=False, zorder=zorder + 0.5,
                                        alpha=None)
        #ax.plot(xs[:, -1, 0], xs[:, -1, 1], xs[:, -1, 2], zorder=zorder+0.5, color=colors[0, -1, :3])
