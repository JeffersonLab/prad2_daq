"""
    This script assigns the crates to the hycal modules according to the provided rules.
    It generates a report to summarize the FADC boards and VTP optical links needed for the giving assignment
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


ANNO_FS = (7, 12)


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


def calc_col_row(dfin, nmax=100):
    xmin, ymin = dfin[['x', 'y']].min()
    xsize, ysize = dfin[['size_x', 'size_y']].max()
    xrel, yrel = dfin[['x', 'y']].values.T
    xrel -= xmin - xsize/2.
    yrel -= ymin - ysize/2.
    return np.digitize(yrel, np.arange(0, ysize * nmax, ysize)), np.digitize(xrel, np.arange(0, xsize*nmax, xsize))


# column number of crystal boundaries (left edge)
PBWO4_BINS = [3, 9, 15, 21, 27, 33]
LG1_BINS = [6, 10, 14, 17, 21]
LG3_BINS = [0, 5, 9, 12, 16, 20]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dist', type=float, default=1.01,
                        help='distance threshold to be a neighbor (normalized to the module size).')
    parser.add_argument('-o', '--output', type=str, default='./plots',
                        help='output directory.')
    parser.add_argument('--database', type=str, default='./database',
                        help='database directory.')
    parser.add_argument('--annotate-module', action='store_true',
                        help='show the module names in generated figures, used for the plot mode.')
    args = parser.parse_args()

    database = args.database
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # read modules
    dfm = pd.read_csv(os.path.join(database, 'hycal_modules.dat'), sep=r'\s+', comment='#')
    module_list = dfm[['name', 'x', 'y', 'size_x', 'size_y']].values
    dfm.loc[:, 'crate'] = -1
    pwo_mask = dfm['name'].str.startswith('W')
    lg_mask = dfm['name'].str.startswith('G')

    # column and row numbers of module in their arrays (crystal array and 4 lead-glass arrays)
    dfm.loc[:, 'col'] = 0
    dfm.loc[:, 'row'] = 0

    # boundaries - crystal modules
    rows, cols = calc_col_row(dfm[pwo_mask])
    dfm.loc[pwo_mask, 'col'] = cols
    dfm.loc[pwo_mask, 'row'] = rows
    dfm.loc[pwo_mask, 'crate'] = np.digitize(dfm.loc[pwo_mask, 'col'].values, PBWO4_BINS)

    # boundaries - lead glass modules
    # separate lead glass into 4 groups
    dfm.loc[:, 'subgroup'] = 0
    xs, ys = dfm[['x', 'y']].values.T
    left, right, top, bot = -17*20.5, 17*20.5, 17*20.5, -17.*20.5
    masks = [
        (xs <= right) & (ys >= top),
        (xs >= right) & (ys >= bot),
        (xs >= left) & (ys <= bot),
        (xs <= left) & (ys <= top),
    ]
    for i, mask in enumerate(masks):
        dfm.loc[lg_mask & mask, 'subgroup'] = i + 1
        rows, cols = calc_col_row(dfm[lg_mask & mask])
        dfm.loc[lg_mask & mask, 'col'] = cols
        dfm.loc[lg_mask & mask, 'row'] = rows
    # define crates for each group
    # left group goes to crate 0, and right group goes to the crate Nmax
    dfm.loc[dfm['subgroup'] == 4, 'crate'] = 0
    dfm.loc[dfm['subgroup'] == 2, 'crate'] = 6
    mask1 = dfm['subgroup'] == 1
    mask3 = dfm['subgroup'] == 3
    dfm.loc[mask1, 'crate'] = np.digitize(dfm.loc[mask1, 'col'].values, LG1_BINS)
    dfm.loc[mask3, 'crate'] = np.digitize(dfm.loc[mask3, 'col'].values, LG3_BINS)

    # transfer centeral point to matplotlib anchor point (lower left corner)
    dfm.loc[:, 'anchor_x'] = dfm['x'] - dfm['size_x']/2.
    dfm.loc[:, 'anchor_y'] = dfm['y'] - dfm['size_y']/2.

    dfm.loc[:, 'anno'] = ''
    if args.annotate_module:
        dfm.loc[:, 'anno'] = dfm.loc[:, 'name'].str.strip('W')

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
        fc = colors[m['crate']]
        alpha = 0.8
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
    fig.savefig(os.path.join(output_dir, 'hycal_daq_boundaries.png'))

    # run a neighbor check
    module_list = dfm[['name', 'x', 'y', 'size_x', 'size_y']].values
    crates = dfm[['crate']].values
    neighbor_crates = []
    for i in np.arange(len(module_list)):
        cname, neighbors = find_neighbors(module_list, i, args.dist, args.dist)
        ncrate_list = [crates[i]]
        for crate in dfm.loc[neighbors, 'crate'].values:
            if crate not in ncrate_list:
                ncrate_list.append(crate)
        if len(ncrate_list) > 2:
            print(('Warning! Module {} is neighboring to more than one crate: {},'
                   + 'only keeping the information for crate {}').format(
                cname, ncrate_list[1:], ncrate_list[-1]))
        if len(ncrate_list) == 1:
            neighbor_crates.append(-1)
        else:
            neighbor_crates.append(ncrate_list[-1])
    dfm.loc[:, 'neighbor_crate'] = neighbor_crates

    # print out a report for the daq crates
    for crate in np.arange(0, dfm['crate'].max() + 1):
        dfc = dfm[dfm['crate'] == crate]
        dfc_pwo = dfm[pwo_mask & (dfm['crate'] == crate)]
        dfc_lg = dfm[lg_mask & (dfm['crate'] == crate)]
        fadc_pwo = int(np.ceil(len(dfc_pwo) / 16))
        fadc_lg = int(np.ceil(len(dfc_lg) / 16))
        print('Crate {}: Nb of modules = {} ({} FADCs), PbWO4 = {} ({} FADCs), LG = {} ({} FADCs)'.format(
            crate + 1, len(dfc), fadc_pwo + fadc_lg, len(dfc_pwo), fadc_pwo, len(dfc_lg), fadc_lg))
        dfn = dfc[dfc['neighbor_crate'] != -1]
        neighbor_str = '\t Nb of neighboring modules = {}: '.format(len(dfn))
        neighbor_substrs = []
        for nc in dfn['neighbor_crate'].unique():
            nn = np.sum(dfn['neighbor_crate'] == nc)
            nlinks = int(np.ceil(nn / 32))
            neighbor_substrs.append('{} to Crate {} ({} Optical Links)'.format(nn, nc + 1, nlinks))
        print(neighbor_str + ', '.join(neighbor_substrs))

    data_cols = ['name', 'subgroup', 'row', 'col', 'crate', 'neighbor_crate']
    with open(os.path.join(args.database, 'hycal_daq_map.txt'), 'w') as fo:
        fo.write(dfm[data_cols].to_string(index=False))
