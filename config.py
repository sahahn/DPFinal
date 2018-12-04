#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:06:10 2018

@author: sage
"""
import numpy as np
from scipy.optimize import minimize

def convert_zCDP_eps_delta(rho, delta):
    return rho + 2*np.sqrt(rho*np.log(1/delta))

config = {}

#Guiding contraints
config['ran_state'] = 42
config['test_size'] = .2

#Data loading information
config['csv_file'] = "fire-data.csv"
config['specs'] = 'fire-data-specs.json'
config['drop_na'] = True
config['bins'] = 20

#Set delta relative to size of training dataset
config['full_dataset_size'] = 305133
config['n'] = int(config['full_dataset_size'] * (1 - config['test_size']))
config['delta'] = (1 / (config['n'] ** 2))

config['num_features'] = 27

#Configs for learning structure
config['struct_threshold'] = .1
config['max_struct_cost'] = 500
config['struct_epsilon'] = 1

#Configs for generating conditional marginal and fake data
config['gen_epsilon'] = 1
config['omega'] = 22
config['zCDP'] = True

#Number of sampels to generate
config['num_to_generate'] = 10000

#Optional parameter in post processing for seed - aka ignore that marginal count if less then k
config['k'] = 5

#For adv comp version use 2/3 of delta budget
config['gen_delta'] = 2*config['delta']/3


def opt(x, delta=config['gen_delta'], epsilon=config['gen_epsilon']):
    ep = convert_zCDP_eps_delta(x, delta)
    
    if epsilon - ep > 0:
        return 1 - ep
    else:
        return 999999

if config['zCDP']:
    x=0
    res = minimize(opt, x, method='nelder-mead',options={'xtol': 1e-8})
    config['rho'] = res.x[0]
    
    config['rho_i'] = config['rho'] / config['num_features']

else:
    #Indv. epsilon value used in param_learn - and in generating data from from seed conditionals
    config['epsilon_p'] = config['gen_epsilon'] / (2 * np.sqrt(2*config['num_features']) * np.sqrt(np.log(1/config['gen_delta'])))
    
    #Check better bound on adv vs sequential comp
    if config['epsilon_p'] < (config['gen_epsilon'] / config['num_features']):
        config['epsilon_p'] = (config['gen_epsilon'] / config['num_features'])
        
        config['gen_delta'] = 0












