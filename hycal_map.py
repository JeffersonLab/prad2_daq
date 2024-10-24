"""
    This script generates plots for the hycal DAQ map
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


ANNOTATE_TYPES = {'module', 'vpcb'}
ANNOTATE_FS = (6.5, 11.5)
# colors for different boards
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dist', type=float, default=1.01,
                        help='distance threshold to be a neighbor (normalized to the module size).')
    parser.add_argument('-o', '--output', type=str, default='./plots',
                        help='output directory.')
    parser.add_argument('--database', type=str, default='./database',
                        help='database directory.')
    parser.add_argument('--annotate', type=str, default='vpcb',
                        help='annotate type, choose one from {}.'.format(ANNOTATE_TYPES))
    parser.add_argument('--crystal-only', action='store_true',
                        help='generate the map without lead glass')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    annotate_type = args.annotate.lower()
    if annotate_type not in ANNOTATE_TYPES:
        print('annotate type {} not supported (choose one from {}), no annotation will be added to plots.'.format(args.annotate, ANNOTATE_TYPES))

    # color map style
    plt.close()
    cmap = plt.get_cmap('Spectral')

    lg_tag = 'central' if args.crystal_only else 'full'

    # geometrical information
    dfm = pd.read_csv(os.path.join(args.database, 'hycal_modules.dat'), sep=r'\s+', comment='#', index_col=0)
    # transfer centeral point to matplotlib anchor point (lower left corner)
    dfm['x'] -= dfm['size_x']/2.
    dfm['y'] -= dfm['size_y']/2.
    dfm.loc[:, 'anno'] = ''

    # primex daq information
    dfd = pd.read_csv(os.path.join(args.database, 'from_primex', 'trig_cable_report.txt'), sep=r'\s+', index_col=0)
    dfd.set_index('name', drop=True, inplace=True)

    dfm.loc[:, 'board'] = dfd['board']
    dfm.loc[:, 'connector'] = dfd['connector']

    if args.crystal_only:
        dfm = dfm[~dfm.index.str.startswith('G')]

    if annotate_type == 'module':
        dfm.loc[:, 'anno'] = dfm.index.str.strip('W')
    elif annotate_type == 'vpcb':
        dfm.loc[:, 'anno'] = dfm['connector']

    # find the plot range
    min_x = dfm['x'].min()
    min_y = dfm['y'].min()
    max_x = dfm['x'].max() + dfm['size_x'].max()
    max_y = dfm['y'].max() + dfm['size_y'].max()

    fig, ax = plt.subplots(figsize=(18, 16), dpi=160, gridspec_kw=dict(left=0.1, right=0.9, bottom=0.08, top=0.98))

    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()


    for r, m in dfm.iterrows():
        fc = colors[m['board'] % len(colors)] if annotate_type == 'vpcb' else 'wheat'
        # alpha = 0.2 + 0.8*(m['connector'] // 32) / 6.
        alpha = 0.8
        rec = Rectangle(xy=m[['x', 'y']].values, width=m['size_x'], height=m['size_y'],
                        fc=mcolors.to_rgba(fc, alpha),
                        ec='black', lw=1.0)
        ax.add_patch(rec)
        # annotation
        if not pd.isnull(m['anno']):
            if args.crystal_only:
                fs = ANNOTATE_FS[1]
            else:
                fs = ANNOTATE_FS[0] if r.startswith('W') else ANNOTATE_FS[1]        
            rx, ry = rec.get_xy()
            cx = rx + rec.get_width() / 2.0
            cy = ry + rec.get_height() / 2.0
            ax.annotate(m['anno'], (cx, cy), color='k', fontsize=fs, ha='center', va='center')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.tick_params(labelsize=26)
    ax.set_xlabel('X (mm)', fontsize=28)
    ax.set_ylabel('Y (mm)', fontsize=28)

    # annotate board numbers
    if annotate_type == 'vpcb':
        for bn in dfm['board'].unique():
            group = dfm[dfm['board'] == bn]
            x = group['x'].mean()
            y = group['y'].mean()
            ax.annotate(bn, (x, y), color='k', fontsize=40, ha='left', va='bottom')

    # color bar position
    fig.savefig(os.path.join(args.output, 'hycal_{}_map_{}.png'.format(annotate_type, lg_tag)))
