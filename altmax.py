import sys
import numpy as np
import scipy.sparse as sp
import time
from random import seed
from random import randint
from gurobipy import *
from general_ten import nonten,krcomp,predict
import pyten
from pyten.method import *
from pyten.tenclass import Tensor  # Use it to construct Tensor object
from pyten.tools import tenerror
def run_exp(R,N,Corners,Reps):
    for i in range(len(R)):
        
        r = R[i]
        n = N[i]
        print('r',r)
        print('n',n)
        corners = Corners[i]
        reps = Reps[i]
        #with open(f'experiments/printout/r_{r}_n_{n}_corners_{corners}_reps_{reps}.txt', 'w') as sys.stdout:

        # Compute derived parameters
        p = len(r) # order of the tenor
        cum_r = np.insert(np.cumsum(r), 0, 0)

        
        nonten_results = np.zeros((reps, 8)) 
        start = int(np.sum(nonten_results[:,1] != 0))

        for rep in range(start,reps):
            print("Starting Reptition No.", rep+1)
            rng = np.random.default_rng(rep)
            # Generate random tensor within rank-1 tensor ball
            phi = np.zeros(np.prod(r))
            pho = np.zeros(r)
            lam = rng.uniform(size=corners)
            lam = lam/np.sum(lam)
            for ind in range(corners):
                the = rng.uniform(size=np.sum(r))
                the = 2*1.0*(the < 0.5) -1 
                
                the_t = the[cum_r[0]:cum_r[1]]
                for k in range(1,p):
                    the_t = np.tensordot(the_t, the[cum_r[k]:cum_r[k+1]], axes = 0)
                
                phi += lam[ind]*the_t.flatten()
                pho += lam[ind]*the_t
            
            # Generate n samples randomly drawn from the tensor entries # "n samples" is "n known entries (might be repetitive)"
            X = np.zeros(n, dtype=int) # stores the flatten indices of known entries
            Xs = np.zeros(np.prod(r), dtype=int)
            Y = np.zeros(n) # values of known entries
            Yo = -10000*np.ones(np.prod(r)) # -1 means that the true entry value is unknown
            for ind in range(n):
                ind_s2i = np.ravel_multi_index([rng.integers(low=0,high=r[k]-1,endpoint=True) for k in range(p)], r) # index of known entry
                X[ind] = ind_s2i
                Y[ind] = phi[ind_s2i]
                Yo[ind_s2i] = phi[ind_s2i]
            Yo = Yo.reshape(r)

            print('First 20 entries in X', X[:20], '\n')
            print('First 20 entries in Y', Y[:20], '\n')
            print("")
            print("Running BCG...")
            last_time = time.time()
            psi_n, iter_count, sigd_count, ip_count, as_size, as_drops, altmax_time = nonten(X, Y, r, rng, tol=1e-4)
            curr_time = time.time()
            elapsed_time = curr_time - last_time
            nonten_results[rep, 0] = np.dot(phi-psi_n,phi-psi_n)/np.dot(phi,phi)
            nonten_results[rep, 1] = elapsed_time
            nonten_results[rep, 2] = iter_count
            nonten_results[rep, 3] = sigd_count
            nonten_results[rep, 4] = ip_count
            nonten_results[rep, 5] = as_size
            nonten_results[rep, 6] = as_drops
            nonten_results[rep,7] = altmax_time
            
            
        return np.mean(nonten_results[:,1]),np.std(nonten_results[:,1])/np.sqrt(reps), np.mean(nonten_results[:,7]),np.std(nonten_results[:,7])/np.sqrt(reps)
    
# # # p 
# print('-------------P------------------')
# Reps = [10] #same as neurips paper
# Corners = [10] 
# N = [10000]
# results = np.zeros((5, 5)) 
# for p in reversed(range(4,8)): #(4,9)
#     R = [tuple(10  for _ in range(p))]
#     res = run_exp(R,N,Corners,Reps)
#     results[p-4,0]=p
#     results[p-4,1] = res[0]
#     results[p-4,2] = res[1]
#     results[p-4,3] = res[2]
#     results[p-4,4] = res[3]

    
    print('results',results)

results = np.zeros((10, 5)) 
Corners = [10] 
Reps = [10]
for r_ in range(1,11):
    r = r_*10
    R = [(r,r,r)]
    N = [1000]
    
    res=run_exp(R,N,Corners,Reps)
    results[r_-1,0]=r
    results[r_-1,1] = res[0]
    results[r_-1,2] = res[1]
    results[r_-1,3] = res[2]
    results[r_-1,4] = res[3]
    print(results)


#percent 1
# print('-------------Percent 1------------------')
# results = np.zeros((4, 5)) 
# R = [(10,10,10,10,10,10)]
# percent = [0.01,0.1,1,10]
# for i in range(4): 
#     N = [int(1000000*percent[i])]
#     res = run_exp(R,N,Corners,Reps)

#     results[i,0]=percent[i]
#     results[i,1] = res[0]
#     results[i,2] = res[1]
#     results[i,3] = res[2]
#     results[i,4] = res[3]
#     print('results',results)
# np.save(f'percent_1.npy', results)

# #percent 2
# print('-------------Percent 2------------------')
# results = np.zeros((4, 5)) 
# R = [(10,10,10,10,10,10,10)]
# percent = [0.01,0.1,1,10]
# for i in range(4): 
#     N = [int(10000000*percent[i])]
#     res = run_exp(R,N,Corners,Reps)

#     results[i,0]=percent[i]
#     results[i,1] = res[0]
#     results[i,2] = res[1]
#     results[i,3] = res[2]
#     results[i,4] = res[3]
#     print('results',results)
# np.save(f'percent_2.npy', results)