#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:15:22 2018

@author: sage
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import json

from config import config


def flip(data):
    return np.swapaxes(data,0,1)

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def laplace_mech_vec(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon, size=len(v))

def gaussian_mech_zcdp(v, sensitivity, rho):
    return v + np.random.normal(loc=0, scale=sensitivity / np.sqrt(2*rho))

def load_data():
    raw_data = pd.read_csv(config['csv_file'])
    
    if config['drop_na']:
        raw_data = raw_data.dropna(axis='columns')
    
    with open(config['specs']) as data_file:    
        specs = json.load(data_file)
        
    data = []
    names = []
    encoders = []
    
    for col in raw_data:
        names.append(col)
        
        if specs[col]['type'] == 'enum':
            
            if len(np.unique(raw_data[col])) > config['bins']:
                enc = KBinsDiscretizer(n_bins=config['bins'], encode='ordinal', strategy='uniform')
                binned = enc.fit_transform(np.array(raw_data[col]).reshape(-1, 1))
                binned = binned.squeeze()
                
                data.append(binned)
                encoders.append(enc)
                
            else:
                data.append(raw_data[col])
                encoders.append(None)
               
    
        elif specs[col]['type'] == 'float' or specs[col]['type'] == 'integer':
            

            
            if len(np.unique(raw_data[col])) > config['bins']:
 
                enc = KBinsDiscretizer(n_bins=config['bins'], encode='ordinal', strategy='uniform')
                binned = enc.fit_transform(np.array(raw_data[col]).reshape(-1, 1))
                binned = binned.squeeze()
            
                data.append(binned)
                encoders.append(enc)
                
            else:
                data.append(raw_data[col])
                encoders.append(None)
            
    return data, names, encoders

def convert(synth_data, names, encoders):
    
    with open(config['specs']) as data_file:    
        specs = json.load(data_file)
        
    full_output = []
    
    for d in synth_data:
        output = []
        
        for i in range(len(names)):
            if encoders[i] == None:
                output.append(d[i])
                
            else:
                bin_edges = encoders[i].bin_edges_[0]
                col_type = specs[names[i]]['type']
                
                e1 = bin_edges[int(d[i])]
                e2 = bin_edges[int(d[i]+1)]
                
                if col_type == 'float':
                    output.append(np.random.uniform(e1, e2))
                
                elif col_type == 'integer' or col_type == 'enum':
                    output.append(np.random.randint(round(e1), round(e2)))

        full_output.append(output)
    
    return full_output

    
    
        
    
    
    
    
    