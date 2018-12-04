import numpy as np
from utils import laplace_mech


"""
Modified from -
FCBF - Fast Correlation Based Filter

L. Yu and H. Liu. Feature Selection for High‐Dimensional Data: A Fast Correlation‐Based Filter Solution. 
In Proceedings of The Twentieth International Conference on Machine Leaning (ICML‐03), 856‐863.
Washington, D.C., August 21‐24, 2003.

W/ addition of differential privacy
"""

class FCBF:
    
    idx_sel = []
    
    def __init__(self, n, epsilon_i, th=.01):
        '''
        Parameters
        ---------------
            n = num data entries
            epsilon_i = The indv epsilon value to be used for each calculate entropy cost
            th = The initial threshold
        '''
        
        self.th = th
        self.n = n
        self.epsilon_i = epsilon_i
        self.seen_SUs = {}
        self.seen_singleEntropies = {}
        self.cnt = 0
        
    def symmetricalUncertain(self,x,y):
        '''Calculate symmetricalUncertainty while memoizing previously calculated noisy entropy scores'''
        
        h = np.sum(x+y)
        
        if h not in self.seen_SUs:
    
            n = float(y.shape[0])
            vals = np.unique(y)
            
            seen_x = np.sum(x)
            seen_y = np.sum(y)
            
            if seen_x not in self.seen_singleEntropies:
                self.seen_singleEntropies[seen_x] = laplace_mech(self.entropy(x), np.log2(self.n)/
                                                     self.n, self.epsilon_i)
                
                self.cnt += 1
                
            if seen_y not in self.seen_singleEntropies:
                self.seen_singleEntropies[seen_y] = laplace_mech(self.entropy(y), np.log2(self.n)/
                                                     self.n, self.epsilon_i)
                
                self.cnt += 1
                
            Hx_noise = self.seen_singleEntropies[seen_x]
            Hy_noise = self.seen_singleEntropies[seen_y]

            # Computing Joint entropy between x and y.
            partial = np.zeros(shape = (vals.shape[0]))

            for i in range(vals.shape[0]):    
                partial[i] = self.entropy(x[y == vals[i]]) 

            partial[np.isnan(partial)==1] = 0      
            py = self.count_vals(y).astype(dtype = 'float64') / n

            Hxy = np.sum(py[py > 0]*partial)
            
            Hxy_noise = laplace_mech(Hxy, np.log2(self.n)/self.n, self.epsilon_i)
            self.cnt += 1

            IG = Hx_noise-Hxy_noise
            SU = 2*IG/(Hx_noise+Hy_noise)
            
            self.seen_SUs[h] = np.clip(SU, 0, 1)

        return self.seen_SUs[h]

    def fit(self, x, y):
        '''
        This function executes FCBF algorithm and saves indexes 
        of selected features in self.idx_sel
        
        Parameters
        ---------------
            x = dataset  [NxM] 
            y = label    [Nx1]
        '''
        self.idx_sel = []
        """
        First Stage: Computing the SU for each feature with the response.
        """
        
        SU_vec = np.apply_along_axis(self.symmetricalUncertain, 0, x, y)
        SU_list = SU_vec[SU_vec > self.th]
        SU_list[::-1].sort()
        
        m = x[:,SU_vec > self.th].shape
        x_sorted = np.zeros(shape = m)
        
        for i in range(m[1]):
            ind = np.argmax(SU_vec)
            SU_vec[ind] = 0
            x_sorted[:,i] = x[:,ind].copy()
            self.idx_sel.append(ind)
        
        """
        Second Stage: Identify relationships between feature to remove redundancy.
        """
        j = 0
        while True:
            """
            Stopping Criteria:The search finishes
            """
            if j >= x_sorted.shape[1]: break
            y = x_sorted[:,j].copy()
            x_list = x_sorted[:,j+1:].copy()
            if x_list.shape[1] == 0: break
                
            SU_list_2 = SU_list[j+1:]
            SU_x = np.apply_along_axis(self.symmetricalUncertain, 0, 
                                       x_list, y)
            
            comp_SU = SU_x >= SU_list_2
            to_remove = np.where(comp_SU)[0] + j + 1 
            if to_remove.size > 0:
                x_sorted = np.delete(x_sorted, to_remove, axis = 1)
                SU_list = np.delete(SU_list, to_remove, axis = 0)
                to_remove.sort()
                for r in reversed(to_remove): 
                    self.idx_sel.remove(self.idx_sel[r])
            j = j + 1        

            
    def count_vals(self, x):
        vals = np.unique(x)
        occ = np.zeros(shape = vals.shape)    
        for i in range(vals.size):
            occ[i] = np.sum(x == vals[i])
        return occ

    def entropy(self, x):
    
        n = float(x.shape[0])
        ocurrence = self.count_vals(x)
        px = ocurrence / n
        return -1* np.sum(px*np.log2(px))