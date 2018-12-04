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

#Size of the full dataset to set delta
config['n'] = 305133 
config['delta'] = (1 / (config['n'] ** 2))

config['epsilon_nt'] = .005
config['epsilon'] = 120

config['num_features'] = 27

config['omega'] = 15

config['num_to_generate'] = 1000
config['lam'] = 1.0000001 #Has to be over 1~

config['k'] = 200

#Indv epsilon value used in param_learn - learning the noisy cond prob
config['epsilon_p'] = config['epsilon'] / (2 * np.sqrt(2*config['num_features']) * np.sqrt(np.log(1/config['delta'])))

if config['epsilon_p'] < (config['epsilon'] / config['num_features']):
    config['epsilon_p'] = (config['epsilon'] / config['num_features'])

config['t'] = config['k'] - 1

#Figure out the target epsilon per generated sample, where delta prime is delta/2
config['epsilon_per'] = config['epsilon'] / (2 * np.sqrt(2*config['num_to_generate']) * np.sqrt(np.log(2/config['delta'])))
config['delta_per'] = config['delta'] / config['num_to_generate'] #Rest of delta- the delta cost for each record released

epsilon_0 = -(np.log(config['delta_per']) / (config['k'] - config['t']))
epsilon_f = epsilon_0 + np.log(1 + (config['lam']/ config['t']))

while epsilon_f > config['epsilon_per']:
    
    config['t'] -= 1
    if config['t'] == 0:
        print('Error, invalid params, no solution for t')
    
    epsilon_0 = -(np.log(config['delta_per']) / (config['k'] - config['t']))
    epsilon_f = epsilon_0 + np.log(1 + (config['lam']/ config['t']))

config['epsilon_0'] = -(np.log(config['delta']) / (config['k'] - config['t']))








