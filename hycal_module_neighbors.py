"""
    This script has 2 modes: LIST and PLOT
    LIST mode: it iterates all HyCal modules and prints out all the neighboring modules for each of them
    PLOT mode: it generates a plot showing the giving center module and its neighboring modules
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


ALLOWED_MODE = ['plot', 'list']
ANNO_FS = (8, 12)


# modules is a list of tuples (name, x, y, size_x, size_y)
# return a boolean array as the indices of the input modules
def find_neighbors(modules, im, norm_dist_x=1.01, norm_dist_y=1.01, including_center=False):
    name, x, y, sx, sy = modules[im]
    name2, x2, y2, sx2, sy2 = modules.T
    mask1 = (np.abs(x2 - x)/sx2 < norm_dist_x) & (np.abs(y2 - y)/sy2 < norm_dist_y)
    mask2 = (np.abs(x2 - x)/sx < norm_dist_x) & (np.abs(y2 - y)/sy < norm_dist_y)
    # neighbors excluding the center itself
    candidates = mask1 | mask2
    candidates[im] = including_center
    return name, candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='running mode selection, please choose one from {}.'.format(ALLOWED_MODE))
    parser.add_argument('-c', '--center', type=str, default='W1',
                        help='name of the center module, used for the plot mode.')
    parser.add_argument('-d', '--dist', type=float, default=1.01,
                        help='distance threshold to be a neighbor (normalized to the module size).')
    parser.add_argument('-o', '--output', type=str, default='./plots',
                        help='output directory.')
    parser.add_argument('--database', type=str, default='./database',
                        help='database directory.')
    parser.add_argument('--annotate-module', action='store_true',
                        help='show the module names in generated figures, used for the plot mode.')
    args = parser.parse_args()

    if str.lower(args.mode) not in ALLOWED_MODE:
        print('Unknowned mode {}, please choose one from {}.'.format(args.mode, ALLOWED_MODE))
        exit(1)

    database = args.database
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # read modules
    dfm = pd.read_csv(os.path.join(database, 'hycal_modules.dat'), sep=r'\s+', comment='#')
    module_list = dfm[['name', 'x', 'y', 'size_x', 'size_y']].values

    if args.mode == 'list':
        for i in np.arange(len(module_list)):
            cname, neighbors_boolean = find_neighbors(module_list, i, args.dist, args.dist)
            neighbors = module_list.T[0][neighbors_boolean]
            print(cname, neighbors)

    elif args.mode == 'plot':

        # center and neighbors
        if str.upper(args.center) not in dfm['name'].values:
            print('cannot find module {}.'.format(args.center))
            exit(-1)

        ic = np.where(module_list.T[0] == args.center)[0][0]
        cname, neighbors_boolean = find_neighbors(module_list, ic, args.dist, args.dist)
        neighbors = module_list.T[0][neighbors_boolean]
        print(cname, neighbors)

        # transfer centeral point to matplotlib anchor point (lower left corner)
        dfm.loc[:, 'anchor_x'] = dfm['x'] - dfm['size_x']/2.
        dfm.loc[:, 'anchor_y'] = dfm['y'] - dfm['size_y']/2.

        dfm.loc[:, 'anno'] = ''
        if args.annotate_module:
            mask = dfm['name'].isin(np.hstack([[cname], neighbors]))
            dfm.loc[mask, 'anno'] = dfm.loc[mask, 'name']

        # find the plot range
        min_x = dfm['anchor_x'].min()
        min_y = dfm['anchor_y'].min()
        max_x = dfm['anchor_x'].max() + dfm['size_x'].max()
        max_y = dfm['anchor_y'].max() + dfm['size_y'].max()

        fig, ax = plt.subplots(figsize=(18, 16), dpi=160, gridspec_kw=dict(left=0.1, right=0.9, bottom=0.08, top=0.98))
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()
        # colors for different boards
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for _, m in dfm.iterrows():
            fc = 'white'
            alpha = 0.8
            if m['name'] == cname:
                fc = 'lightcoral'
            if m['name'] in neighbors:
                fc = 'teal'
            rec = Rectangle(xy=m[['anchor_x', 'anchor_y']].values, width=m['size_x'], height=m['size_y'],
                            fc=mcolors.to_rgba(fc, alpha),
                            ec='black', lw=1.0)
            ax.add_patch(rec)
            # annotation
            if not pd.isnull(m['anno']):
                fs = ANNO_FS[0] if m['name'].startswith('W') else ANNO_FS[1]
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                ax.annotate(m['anno'], (cx, cy), color='k', fontsize=fs, ha='center', va='center')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(labelsize=26)
        ax.set_xlabel('X (mm)', fontsize=28)
        ax.set_ylabel('Y (mm)', fontsize=28)

        # color bar position
        fig.savefig(os.path.join(output_dir, 'hycal_neighbors_{}.png'.format(cname)))

