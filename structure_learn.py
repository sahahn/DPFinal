#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:57:36 2018

@author: sage
"""

from FCBF_DP import FCBF
import networkx as nx
from utils import laplace_mech
from config import config
import numpy as np

def max_cost_check(G, unique_vals):
    
    parents = [list(G.predecessors(node)) for node in G]
    
    for p in parents:
        cost = np.prod((np.array([len(unique_vals[a]) for a in p])))
        if cost > config['max_struct_cost']:
            return False
        
    return True

def split_data(sample, ind):
    '''Helper function to split data by feature - seperate out the feature indicated by ind'''
    
    x = np.delete(sample, ind, axis=0)
    x = np.rollaxis(x, axis=-1)
    
    y = sample[ind,:]
    
    return x,y

def learn_structure(sample, unique_vals):
    
    n_noise = laplace_mech(len(sample[0]), 1, config['epsilon_nt'])
    m = len(sample)
    
    epsilon_i = (config['epsilon'] - config['epsilon_nt']) / (2 * np.sqrt(m**2 + m) * np.sqrt(np.log(1/config['delta'])))
    
    G = nx.DiGraph()
    
    fcbf = FCBF(n_noise, epsilon_i, config['struct_threshold'])

    arr = np.arange(len(sample))
    np.random.shuffle(arr)
    
    for i in arr:
        
        G.add_node(i)
        x,y = split_data(sample,i)
        
        fcbf.fit(x,y)
        
        for j in fcbf.idx_sel:
            
            if j >= i:
                j+=1
            
            G.add_edge(i,j)
    
            if not nx.is_directed_acyclic_graph(G) or not max_cost_check(G, unique_vals):
                G.remove_edge(i,j)
                
        fcbf.idx_sel = []
        
    parents = [sorted(list(G.predecessors(i))) for i in range(len(sample))]
    
    order = []
    for node in nx.topological_sort(G):
        order.append(node)
        
    return parents, order
    
