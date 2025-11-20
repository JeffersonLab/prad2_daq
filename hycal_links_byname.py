"""
    This script makes a dataframe of the channels that are linked.
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

    data = {'base.name':[], 'link1':[], 'link2':[], 'link3':[]}
    linkbase = pd.DataFrame(data)

    k = 0
    for y in np.arange(6):
        if y == 2:
            k = 1
        for p in np.arange(5):
            linkbase.loc[y*46+p] = ['G'+str(5+30*p+4*y-k),'G'+str(6+30*p+4*y-k),'n','n']
            linkbase.loc[y*46+p+41] = ['G'+str(756+30*p+4*y-k),'G'+str(757+30*p+4*y-k),'n','n']
    
        for i in np.arange(32):
            linkbase.loc[y*46+i+7] = ['W'+str(36+34*i+6*y),'W'+str(37+34*i+6*y),'n','n']

    linkbase.loc[5] = ['G155','G156','G157','n']
    linkbase.loc[6] = ['W3','W2','W1','G186']
    linkbase.loc[39] = ['W1125','W1124','W1123','n']
    linkbase.loc[40] = ['G726','G727','G728','n']

    a = 46    
    linkbase.loc[5+a] = ['G159','G160','G161','n']
    linkbase.loc[6+a] = ['W9','W8','W7','n']
    linkbase.loc[39+a] = ['W1131','W1130','W1129','n']
    linkbase.loc[40+a] = ['G730','G731','n','n']

    a = a+46
    linkbase.loc[5+a] = ['G163','G164','n','n']
    linkbase.loc[6+a] = ['W14','W15','n','n']
    linkbase.loc[39+a] = ['W1136','W1137','W1138','n']
    linkbase.loc[40+a] = ['G734','G735','n','n']

    a = a+46
    linkbase.loc[5+a] = ['G166','G167','n','n']
    linkbase.loc[6+a] = ['W21','W19','W20','n']
    linkbase.loc[39+a] = ['W1142','W1143','n','n']
    linkbase.loc[40+a] = ['G737','G738','n','n']

    a = a+46
    linkbase.loc[5+a] = ['G170','G171','n','n']
    linkbase.loc[6+a] = ['W26','W27','W28','n']
    linkbase.loc[39+a] = ['W1148','W1149','W1150','n']
    linkbase.loc[40+a] = ['G742','G740','G741','n']

    a = a+46
    linkbase.loc[5+a] = ['G175','G174','G173','n']
    linkbase.loc[6+a] = ['W32','W33','W34','n']
    linkbase.loc[39+a] = ['W1154','W1155','W1156','n']
    linkbase.loc[40+a] = ['G746','G744','G745','n']

    linkbase.sort_index(inplace=True)
    #print(linkbase)

    with open(os.path.join(args.database, 'hycal_sidetoside_links_byname.txt'), 'w') as fo:
        fo.write(linkbase.to_string(index=False))
    #REMINDER this is how you concattenate strings and ints in python: print('G' + str(6))