#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:11:51 2018

@author: sage
"""
import numpy as np
import itertools
from config import config
from utils import flip
from utils import laplace_mech, gaussian_mech_zcdp


def generate_fake_data(data, order, parents, count_dicts, unique_vals):
    '''
    Determine conditional marginals across all of the seed index, and used these
    along with the remaining conditional marginals to generate DP synthetic data
    '''
    
    seed_inds = order[:-config['omega']]

    unique, counts = np.unique(data[seed_inds], return_counts=True, axis=1)
    unique = flip(unique)
    
    c_dict = {}
    
    for u in range(len(unique)):
        entry = []
    
        for i in unique[u]:
            entry.append(i)
            
        if config['zCDP']:
            c_dict[tuple(entry)] = max(gaussian_mech_zcdp(counts[u], 1, config['rho_i']), 0) 
        else:
            c_dict[tuple(entry)] =  max(laplace_mech(counts[u], 1, config['epsilon_p']), 0)
        
    u_vals = [unique_vals[p] for p in seed_inds]
    all_pairs = list(itertools.product(*u_vals))
    
    all_pairs = np.array([a for a in all_pairs if a not in c_dict])
    
    if config['zCDP']:
        all_noise = np.random.normal(loc=0, scale=1/np.sqrt(2*config['rho_i']), size=len(all_pairs))
    else:
        all_noise = np.random.laplace(loc=0, scale=1/config['epsilon_p'], size=len(all_pairs))
    
    valid_noise = all_noise[all_noise>config['k']]
    
    r_choices = np.random.choice(np.arange(len(all_pairs)), len(valid_noise), replace=False)
    
    for pair, noise in zip(all_pairs[r_choices], valid_noise):
        c_dict[tuple(pair)] = noise
    
    #Clear some memory
    del unique, counts, all_pairs, all_noise, valid_noise, r_choices
        
    total = 0
    for c in c_dict:
        total += c_dict[c]
    
    options = [c for c in c_dict]
    seed_probs = [c_dict[c] / total for c in options]
    
    to_release = []
    resample_order = order[-config['omega']:]
    
    print('generating samples')

    seed_inds = np.random.choice(np.arange(len(options)), config['num_to_generate'], p=list(seed_probs))
    
    for s_i in seed_inds:
        
        seed = options[s_i]
        fake = []
        
        for j in seed:
            fake.append(j)

        for i in resample_order:
    
            p = parents[i]
            p_vals = []
    
            for p_ind in p:
                
                spot_in_order = order.index(p_ind)
                p_vals.append(fake[spot_in_order])
    
            probs = []
    
            for option in unique_vals[i]:
                probs.append(count_dicts[i][tuple([option] + p_vals)])
                
            probs = probs / np.sum(probs)
            new_val = np.random.choice(unique_vals[i], p=list(probs))
    
            fake.append(new_val)

        #Put canidate back into normal order
        fake_re = [fake[order.index(i)] for i in range(len(data))]
        to_release.append(fake_re)
        
        if len(to_release) % 1000 == 0:
           print(len(to_release))
        
    return to_release

#Legacy stuff
def k_check(fake, c_dict):
    
    entry = []
    for i in fake:
        entry.append(i)
    
    real_count = c_dict[tuple(entry)]
    noisy_k = laplace_mech(config['k'], 1, config['epsilon_0'])
    
    if real_count >= noisy_k:
        return True
    return False
    
        
def old_generate_data(data, order, parents, count_dicts, unique_vals):
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
            
    
            probs /= np.sum(probs)
    
            new_val = np.random.choice(unique_vals[i], p=list(probs))
    
            fake.append(new_val)
        
        #Put canidate back into normal order
        fake_re = [fake[order.index(i)] for i in range(len(data))]
        
        
        if k_check(np.array(fake_re)[inds], c_dict):
            to_release.append(fake_re)
            
    return to_release
        

   
        
        
    
        
