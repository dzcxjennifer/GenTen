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

##################################################################
# EDIT THESE TO CHANGE PROBLEM SETUP

# R = [(2,2)]
# N = [10]
# Corners = [10]
# Reps = [1]

R = [(10,10,10)]
N = [500]
Corners = [10]
Reps = [1]

# R = [(2,2)]
# N = [10]
# Corners = [10]
# Reps = [1]

##################################################################


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
    
    #for rep in range(start,reps):
    for rep in range(0,reps):
        print("Starting Reptition No.", rep+1)
        rng = np.random.default_rng(rep)
        # Generate random tensor within rank-1 tensor ball
        phi = np.zeros(np.prod(r))
        pho = np.zeros(r)
        lam = rng.uniform(size=corners)
        lam = lam/np.sum(lam)
        #lam = [0.70246795, 0.29753205]
        #thes = [[1,  1, -1, -1],[-1, -1, -1, -1,]]
        for ind in range(corners):
            the = rng.uniform(size=np.sum(r))
            the = 2*1.0*(the < 0.5) -1 
            #the = thes[ind]
            the_t = the[cum_r[0]:cum_r[1]]
            #print(the_t)
            for k in range(1,p):
                #print(the[cum_r[k]:cum_r[k+1]])
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

        #X = np.array([3, 2, 0 ,0, 1, 3, 3, 3, 3, 3])
        #Y = np.array([-0.4049359, -0.4049359, -0.4049359, -0.4049359, -0.4049359, -0.4049359,-0.4049359, -0.4049359, -0.4049359, -0.4049359])
        
        #print('Y',Y)
        print('First 20 entries in X', X[:20], '\n')
        print('First 20 entries in Y', Y[:20], '\n')
        print('R',R)
        print('N',N)
        print("")
        print("Running BCG...")
        last_time = time.time()
        psi_n, iter_count, sigd_count, ip_count, as_size, as_drops = nonten(X, Y, r, rng, tol=1e-4)
        curr_time = time.time()
        elapsed_time = curr_time - last_time
        # print('phi',phi)
        # print('psi_n',psi_n)
        
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


        np.save(f'experiments/record/r_{r}_n_{n}_corners_{corners}_reps_{reps}.npy', nonten_results)


    print("")
    print("Experiment Results: ")
    print("")

    # np.save(f'experiments/record/r_{r}_n_{n}_corners_{corners}_reps_{reps}.npy', nonten_results)

    print(f"BCG:")
    print("Mean (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.mean(nonten_results,0))
    print("")

    print("Standard Error (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.std(nonten_results,0)/np.sqrt(reps))
    print("")

    print("Minimum (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.min(nonten_results,0))
    print("")

    print("Maximum (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.max(nonten_results,0))
    print("")

    print("25% Percentile (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.percentile(nonten_results, 25, 0))
    print("")

    print("50% Percentile (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.percentile(nonten_results, 50, 0))
    print("")
    
    print("75% Percentile (NMSE, Time, iter_count, sigd_count, ip_count, as_size, as_drops)")
    print(np.percentile(nonten_results, 75, 0))
    #print(nonten_results[:,0])
    print("")
    print("ALS:")
    print("Mean (NMSE, Time)")
    print(np.mean(amcomp_results,0))

    print("Standard Error (NMSE, Time)")
    print(np.std(amcomp_results,0)/np.sqrt(reps))

    print("")
    print("SiLRTC:")
    print("Mean (NMSE, Time)")
    print(np.mean(silrtc_results,0))

    print("Standard Error (NMSE, Time)")
    print(np.std(silrtc_results,0)/np.sqrt(reps))

    print("")
    print("TNCP:")
    print("Mean (NMSE, Time)")
    print(np.mean(tncomp_results,0))

    print("Standard Error (NMSE, Time)")
    print(np.std(tncomp_results,0)/np.sqrt(reps))