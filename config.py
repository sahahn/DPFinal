#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:06:10 2018

@author: sage
"""
import numpy as np

config = {}

config['ran_state'] = 42

config['csv_file'] = "fire-data.csv"
config['specs'] = 'fire-data-specs.json'
config['drop_na'] = True
config['bins'] = 20
config['seed_split'] = .3

config['struct_threshold'] = .1
config['max_struct_cost'] = 500


config['epsilon_nt'] = .005
config['epsilon'] = 1

config['num_features'] = 27
config['delta'] = 1e-5

#Indv epsilon value used in param_learn - learning the noisy cond prob
config['epsilon_p'] = config['epsilon'] / (2 * np.sqrt(2*config['num_features']) * np.sqrt(np.log(1/config['delta'])))

config['omega'] = 15
config['k'] = 7
config['num_to_generate'] = 1







