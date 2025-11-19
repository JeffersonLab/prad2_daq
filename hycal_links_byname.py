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

    data = {'name':[], 'link1':[], 'link2':[], 'link3':[]}
    linkbase = pd.DataFrame(data)

    for y in np.arange(6)

        for p in np.arange(5):
            linkbase.loc[p] = ['G'+str(5+p*30),'G'+str(6+p*30),'n','n']
    
        for i in np.arange(32):
            linkbase.loc[i+7] = ['W'+str(36+34*i),'W'+str(37+34*i),'n','n']

        for j in np.arange(5):
            linkbase.loc[j+41] = ['G'+str(756+30*j),'G'+str(757+30*j),'n','n']

    linkbase.loc[5] = ['G155','G156','G157','n']
    linkbase.loc[6] = ['W3','W2','W1','G186']
    linkbase.loc[39] = ['W1125','W1124','W1123','n']
    linkbase.loc[40] = ['G726','G727','G728','n']

    print(linkbase)
    #REMINDER this is how you concattenate strings and ints in python: print('G' + str(6))