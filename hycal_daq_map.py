"""
    This script generates plots for the hycal DAQ map
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':
    database = 'database'
    output_dir = 'plots'

    os.makedirs(output_dir, exist_ok=True)

    annotate_module = False
    annotate_board = True
    annotate_connector = True
    anno_fs = (8, 12)

    # color map style
    plt.close()
    cmap = plt.get_cmap('Spectral')

    for include_lead_glass, lg_tag in [(True, 'full'), (False, 'central')]:
        # geometrical information
        dfm = pd.read_csv(os.path.join(database, 'hycal_modules.dat'), sep=r'\s+', comment='#', index_col=0)
        # transfer centeral point to matplotlib anchor point (lower left corner)
        dfm['x'] -= dfm['size_x']/2.
        dfm['y'] -= dfm['size_y']/2.
        dfm.loc[:, 'anno'] = ''

        # primex daq information
        dfd = pd.read_csv(os.path.join(database, 'from_primex', 'trig_cable_report.txt'), sep=r'\s+', index_col=0)
        dfd.set_index('name', drop=True, inplace=True)

        dfm.loc[:, 'board'] = dfd['board']
        dfm.loc[:, 'connector'] = dfd['connector']

        if not include_lead_glass:
            dfm = dfm[~dfm.index.str.startswith('G')]

        if annotate_module:
            dfm.loc[:, 'anno'] = dfm.index.str.strip('W')

        if annotate_connector:
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
        # colors for different boards
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for r, m in dfm.iterrows():
            fc = colors[m['board'] % len(colors)]
            # alpha = 0.2 + 0.8*(m['connector'] // 32) / 6.
            alpha = 0.8
            rec = Rectangle(xy=m[['x', 'y']].values, width=m['size_x'], height=m['size_y'],
                            fc=mcolors.to_rgba(fc, alpha),
                            ec='black', lw=1.0)
            ax.add_patch(rec)
            # annotation
            if not pd.isnull(m['anno']):
                fs = anno_fs[0] if r.startswith('W') else anno_fs[1]
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                ax.annotate(m['anno'], (cx, cy), color='k', fontsize=fs, ha='center', va='center')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(labelsize=26)
        ax.set_xlabel('X (mm)', fontsize=28)
        ax.set_ylabel('Y (mm)', fontsize=28)

        if annotate_board:
            for bn in dfm['board'].unique():
                group = dfm[dfm['board'] == bn]
                x = group['x'].mean()
                y = group['y'].mean()
                ax.annotate(bn, (x, y), color='k', fontsize=40, ha='left', va='bottom')

        # color bar position
        fig.savefig(os.path.join(output_dir, 'hycal_connectors_map_{}.png'.format(lg_tag)))
