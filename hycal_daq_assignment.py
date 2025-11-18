"""
    This script assigns the FADC slots and channels (as well as VTP optical links) to the modules
    It needs the output file of crate assignment from "hycal_daq_boundaries.py"
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


pd.set_option('display.max_rows', 2000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='./database',
                        help='database directory.')
    args = parser.parse_args()

    # read modules
    dfm = pd.read_csv(os.path.join(args.database, 'hycal_daq_map.txt'), sep=r'\s+', comment='#', index_col=0)

    # read info about hycal back boards and connectors
    dfd = pd.read_csv(os.path.join(args.database, 'from_primex', 'trig_cable_report.txt'), sep=r'\s+', index_col=0)
    dfd.set_index('name', drop=True, inplace=True)
    dfm.loc[:, 'VPCB.board'] = dfd['board']
    dfm.loc[:, 'VPCB.connector'] = dfd['connector']

    dfm.loc[:, 'lead glass'] = dfm['subgroup'] > 0
    # placeholders for the assignment
    dfm.loc[:, 'slot'] = -1
    dfm.loc[:, 'channel'] = -1
    dfm.loc[:, 'link_slot'] = -1
    dfm.loc[:, 'link_channel'] = -1

    # assign FADC slots based on the following rules
    # 1. each FADC has 16 connectors
    # 2. within each FADC group, try to avoid mixed types of modules
    # 3. within each FADC group, the connectors on VPCB should not be scattered around

    # sort the channels within the crates
    # assuming we separate the crates by vertically slicing he HyCal
    # VPCB boards then do not matter much (horizontal distance is small), so only sort VPCB connectors
    dfd = dfm.sort_values(['crate', 'lead glass', 'VPCB.connector'])
    pwo_mask = ~dfd['lead glass'].values
    lg_mask = dfd['lead glass'].values
    for cr in dfd['crate'].unique():
        cr_mask = (dfd['crate'] == cr)
        # assign FADC slot and channel
        slot_start = 0
        slot_cap = 16
        for mask in [pwo_mask & cr_mask, lg_mask & cr_mask]:
            i_mods = np.arange(np.sum(mask))
            #Shift all the slot number by 3 because the first fADC slot is 03 in all the VME crates.
            slot_raw = slot_start + i_mods // slot_cap + 3
            #Shift the slot number by another 2 if the current number is above 10 because slots 11 and 12
            #are occupied by VTPs in the real crates.
            slot_real = [x + 2 if x > 10 else x for x in slot_raw]
            dfd.loc[mask, 'slot'] = slot_real
            dfd.loc[mask, 'channel'] = i_mods % slot_cap
            # print(dfd.loc[mask, ['crate', 'slot', 'channel']])
            slot_start += int(np.ceil(max(i_mods) / slot_cap))
        # assign optical links
        link_start = 0
        link_cap = 16
        for nc in dfd.loc[cr_mask, 'neighbor_crate'].unique():
            if nc < 0:
                continue
            mask = cr_mask & (dfd['neighbor_crate'] == nc)
            i_mods = np.arange(np.sum(mask))
            link_slot_raw = link_start + i_mods // link_cap + 3
            link_slot_real = [l + 2 if l > 10 else l for l in link_slot_raw]
            dfd.loc[mask, 'link_slot'] = link_slot_real
            dfd.loc[mask, 'link_channel'] = i_mods % link_cap
            link_start += int(np.ceil(max(i_mods) / link_cap))
    
    # sanity checks
    print(dfd[['crate', 'slot', 'channel']].max())
    print(dfd[['crate', 'slot', 'channel']].min())

    dfd.rename(columns={'neighbor_crate': 'link_to_crate'}, inplace=True)
    data_cols = ['crate', 'slot', 'channel', 'VPCB.board', 'VPCB.connector', 'link_to_crate', 'link_slot', 'link_channel']
    with open(os.path.join(args.database, 'hycal_daq_connections.txt'), 'w') as fo:
        fo.write(dfd[data_cols].reset_index().to_string(index=False))
