import sys
import numpy as np
import scipy.sparse as sp
import time
from random import seed
from random import randint
from gurobipy import *
from general_ten_2 import nonten,krcomp,predict
import pyten
from pyten.method import *
from pyten.tenclass import Tensor  # Use it to construct Tensor object
from pyten.tools import tenerror

def run_exp(R,N,Corners,Reps):
    for i in range(len(R)):
        r = R[i]
        n = N[i]
        corners = Corners[i]
        reps = Reps[i]
        #with open(f'experiments/printout/r_{r}_n_{n}_corners_{corners}_reps_{reps}.txt', 'w') as sys.stdout:

        # Compute derived parameters
        p = len(r) # order of the tenor
        cum_r = np.insert(np.cumsum(r), 0, 0)

        
        nonten_results = np.zeros((reps, 7)) 
        np.save(f'experiments/record/r_{r}_n_{n}_corners_{corners}_reps_{reps}.npy', nonten_results)
        amcomp_results = np.zeros((reps, 2)) 
        silrtc_results = np.zeros((reps, 2)) 
        tncomp_results = np.zeros((reps, 2)) 
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
            psi_n, iter_count, sigd_count, ip_count, as_size, as_drops = nonten(X, Y, r, rng, tol=1e-4)
            curr_time = time.time()
            elapsed_time = curr_time - last_time
            nonten_results[rep, 0] = np.dot(phi-psi_n,phi-psi_n)/np.dot(phi,phi)
            nonten_results[rep, 1] = elapsed_time
            nonten_results[rep, 2] = iter_count
            nonten_results[rep, 3] = sigd_count
            nonten_results[rep, 4] = ip_count
            nonten_results[rep, 5] = as_size
            nonten_results[rep, 6] = as_drops
            
            print("")
            print("Running ALS...")
            last_time = time.time()
            psi_q = krcomp(X, Y, r, corners, 0.1, tol=1e-4)
            #[Tpsi, psi_q] = cp_als(Tensor(Yo), corners, Yo > -0.5, maxiter=1000, printitn=100)
            curr_time = time.time()
            elapsed_time = curr_time - last_time
            amcomp_results[rep, 0] = np.dot(phi-psi_q,phi-psi_q)/np.dot(phi,phi)
            #amcomp_results[rep, 0] = np.dot(phi-psi_q.data.flatten(),phi-psi_q.data.flatten())/np.dot(phi,phi)
            amcomp_results[rep, 1] = elapsed_time

            print("")
            print("Running SiLRTC...")
            last_time = time.time()
            psi_q = silrtc(Tensor(Yo), Yo > -9999)
            curr_time = time.time()
            elapsed_time = curr_time - last_time
            silrtc_results[rep, 0] = np.dot(phi-psi_q.data.flatten(),phi-psi_q.data.flatten())/np.dot(phi,phi)
            silrtc_results[rep, 1] = elapsed_time
            last_time = time.time()

            print("Running TNCP...")
            last_time = time.time()
            selft = TNCP(Tensor(Yo), Yo > -9999, corners)
            selft.run()
            curr_time = time.time()
            elapsed_time = curr_time - last_time
            tncomp_results[rep, 0] = np.dot(phi-selft.X.data.flatten(),phi-selft.X.data.flatten())/np.dot(phi,phi)
            tncomp_results[rep, 1] = elapsed_time
            print("")

            
        return [np.mean(nonten_results[:,0]),np.std(nonten_results[:,0])/np.sqrt(reps),np.mean(nonten_results[:,1]),np.std(nonten_results[:,1])/np.sqrt(reps)
                ,np.mean(amcomp_results,0),np.mean(silrtc_results,0),np.mean(tncomp_results,0)
                ]
    
Reps = [100] #same as neurips paper
Corners = [10] 

#percent 1
print('-------------Percent 1------------------')
results = np.zeros((4, 11))
R = [(10,10,10,10,10,10)]
percent = [0.0001,0.001,0.01,0.1]
for i in range(4):
    N = [int(1000000*percent[i])]
    res = run_exp(R,N,Corners,Reps)

    results[i,0]=percent[i]
    results[i,1] = res[0]
    results[i,2] = res[1]
    results[i,3] = res[2]
    results[i,4] = res[3]
    results[i,5] = res[4][0]
    results[i,6] = res[4][1]
    results[i,7] = res[5][0]
    results[i,8] = res[5][1]
    results[i,9] = res[6][0]
    results[i,10] = res[6][1]
    np.save('results/percent_1.npy', results)
    print('results',results)

