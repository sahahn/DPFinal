#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:11:51 2018

@author: sage
"""
import numpy as np
from config import config
from utils import flip
from utils import laplace_mech


def k_check(fake, c_dict):
    
    entry = []
    for i in fake:
        entry.append(i)
    
    real_count = c_dict[tuple(entry)]
    noisy_k = laplace_mech(config['k'], 1, config['epsilon'])
    
    if real_count >= noisy_k:
        return True
    return False
        
        
def generate_data(data, order, parents, count_dicts, unique_vals):
    '''
    Generate fake samples with k-plausible deniability
    '''

    inds = order[:-config['omega']]
    unique, counts = np.unique(data[inds], return_counts=True, axis=1)
    unique = flip(unique)
    
    c_dict = {}

    for u in range(len(unique)):
        entry = []
        
        for i in unique[u]:
            entry.append(i)
       
        c_dict[tuple(entry)] = counts[u]
    
    resample_order = order[-config['omega']:]   
    to_release = []
    
    while len(to_release) < config['num_to_generate']:
        
        seed_ind = np.random.choice(np.arange(len(data[0])))
        seed = data[:,seed_ind]
    
    
        fake = []
        
        for j in order[:-config['omega']]:
            fake.append(seed[j])
    
        for i in resample_order:
    
            p = parents[i]
            p_vals = []
    
            for p_ind in p:
                
                spot_in_order = order.index(p_ind)
                p_vals.append(fake[spot_in_order])
    
            probs = []
    
            for option in unique_vals[i]:
                probs.append(count_dicts[i][tuple([option] + p_vals)])
            
            if np.sum(probs) == 0:
                print('warning')
    
            probs /= np.sum(probs)
    
            new_val = np.random.choice(unique_vals[i], p=list(probs))
    
            fake.append(new_val)
        
        #Put canidate back into normal order
        fake_re = [fake[order.index(i)] for i in range(len(data))]
        
        
        if k_check(np.array(fake_re)[inds], c_dict):
            to_release.append(fake_re)
            
    return to_release
        

   
        
        
    
        