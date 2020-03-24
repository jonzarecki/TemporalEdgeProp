import os
from os.path import join

EDGEPROP_BASE_DIR = os.path.dirname(os.path.dirname(__file__)) + "/"

DECAYED_TIME_WEIGHT = 'time_weight'
AGG_TIMES_LIST_ATTR = "agg_times_list"

LABEL_GT = 'label'
LABEL_TRAIN = 'label_train'
LABEL_PRED = 'label_pred'  #TODO: not used
NO_LABEL = -1

NODE2VEC_CACHE = os.path.join(EDGEPROP_BASE_DIR, 'edge_prop', 'models', 'saved_models')
TENSORBOARD_DIR = os.path.join(EDGEPROP_BASE_DIR, 'edge_prop', 'visualization', 'tb_runs')

DATA_PATH = join(EDGEPROP_BASE_DIR, 'data')
DATASET2PATH = {'epinions': join(DATA_PATH, 'soc-sign-epinions.txt'),
                'slashdot': join(DATA_PATH, 'soc-sign-Slashdot090221.txt'),
                'tribes': join(DATA_PATH, 'ucidata-gama.txt'),
                'wiki': join(DATA_PATH, 'elec.txt'),
                'aminer_s': join(DATA_PATH, 'aminer_s'),
                'aminer_m': join(DATA_PATH, 'aminer_m'),
                'aminer_l': join(DATA_PATH, 'aminer_l')}
