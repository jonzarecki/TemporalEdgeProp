import os
from os.path import join

EDGEPROP_BASE_DIR = os.path.dirname(os.path.dirname(__file__)) + "/"

DECAYED_TIME_WEIGHT = 'time_weight'
AGG_TIMES_LIST_ATTR = "agg_times_list"

DATA_PATH = join(EDGEPROP_BASE_DIR, 'data')
DATASET2PATH = {'epinions': join(DATA_PATH, 'soc-sign-epinions.txt'),
                'slashdot': join(DATA_PATH, 'soc-sign-Slashdot090221.txt'),
                'wiki': join(DATA_PATH, 'wikiElec.ElecBs3.txt')}
