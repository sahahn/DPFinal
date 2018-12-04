#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:12:13 2018

@author: sage
"""

import numpy as np
from utils import laplace_mech, gaussian_mech_zcdp
import itertools

from config import config


def get_configs(ind, parents, unique_vals):
    
    u_vals = [unique_vals[p] for p in parents[ind]]
    pars = [ind] + parents[ind]

    return pars, list(itertools.product(*u_vals))

def learn_cond_marginals(stats, parents, unique_vals, order):
    
    count_dicts = []
    seed_inds = order[:-config['omega']]

    for ind in range(len(stats)):
        
        count_dict = {}
        if ind not in seed_inds:
        
            ID, configs = get_configs(ind, parents, unique_vals)
            sub_data = stats[ID,]
        
            unique, counts = np.unique(sub_data, return_counts=True, axis=1)
            unique = np.column_stack(unique)
        
            u_list = []
            c_list = []
        
            for u,c in zip(unique, counts):
                u_list.append(list(u))
                c_list.append(c)
        
            for val in unique_vals[ind]:
                for c in configs:
                    entry = [val] + list(c)
                    
                    if entry in u_list:
                        cnt = c_list[u_list.index(entry)]
                    else:
                        cnt = 0
                    
                    if config['zCDP']:
                        count_dict[tuple(entry)] = max(gaussian_mech_zcdp(cnt, 1, config['rho_i']), 0)
                    else:
                        count_dict[tuple(entry)] = max(laplace_mech(cnt, 1, config['epsilon_p']), 0)
                
        count_dicts.append(count_dict)
        
    return count_dicts
    