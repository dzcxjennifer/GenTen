import numpy as np
import scipy.sparse as sp
import time
from gurobipy import *

# Prune active set 
def prune(Pts, Vts, psi_q):
    o = Model()
    lamb = o.addMVar(Pts.shape[1], lb = -1, ub = 1, vtype = GRB.CONTINUOUS)
    o.update()
    o.Params.OutputFlag = 0
    o.addConstr(lamb.sum() == 1)
    simcon = o.addConstr(psi_q == Pts @ lamb)
    o.optimize()
        
    if o.status == GRB.INFEASIBLE:
        simcon = simcon.tolist()
        o.feasRelax(0, False, None, None, None, simcon, np.ones(len(simcon)))
        o.optimize()
    
    return(Pts[:,lamb.X > 0], Vts[:,lamb.X > 0], lamb.X[lamb.X > 0][:,None])

# Golden section search
def golden(X, Y, psi_q, psi_n, lamb_q, lamb_n, lpar):
    n = X.shape[0]
    
    dpsi = psi_q[X] - psi_n[X]
    gam = np.dot(Y/lpar-psi_n[X],dpsi)/np.dot(dpsi,dpsi)
    
    if (gam < 0):
        gam = 0
    elif (gam > 1):
        gam = 1
        
    return(gam*psi_q + (1-gam)*psi_n, gam*lamb_q + (1-gam)*lamb_n, gam)

# Callback function to make Gurobi "lazy"
def callback(model, where):
    if where == GRB.Callback.MIP:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIP_OBJBND))/2)
        if (model._cmin - model.cbGet(GRB.Callback.MIP_OBJBST) > model._gap):
            model._oracle = "LazyIP"
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIPSOL_OBJBND))/2)
        if (model._cmin - model.cbGet(GRB.Callback.MIPSOL_OBJ) > model._gap):
            model._oracle = "LazyIP"
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        model._gap = np.minimum(model._gap, (model._cmin - model.cbGet(GRB.Callback.MIPNODE_OBJBND))/2)
            
# Alternating minimization oracle
def altmin(r, lpar, p, tol, cmin, gap, c, the_q, Un):
    the = the_q.copy()
    cum_r = np.insert(np.cumsum(r), 0, 0)
    un = Un.shape[0]

    last_cmin = cmin + 1e6
    cnt = 0
    while (True):
        cnt += 1
        for ind in range(p):
            # fpro is the c_k when converting <theta_1 x theta_2 x ... x theta_p, grad(f(psi_q))>
            # to <theta_ind, c_k>. Conversion is done between lines 69 to 75
            # fpro is the pro transformed to have the same shape as theta_ind
            # pro has the same shape as grad(f(psi_q)). 
            # pro has the same information as fpro
            pro = lpar*c.copy()
            for k in range(p):
                if (ind != k):
                    pro = np.multiply(pro, the[cum_r[k] + Un[:,k]]) 

            fpro = np.zeros(r[ind])
            for k in range(un):
                fpro[Un[k,ind]] += pro[k]
            the[cum_r[ind]:cum_r[ind+1]] = np.where(fpro < 0, 1, -1) # if negative coeff, then * 1, if postivie, then * -1
            curr_cmin = np.sum(fpro[fpro < 0]) - np.sum(fpro[fpro > 0]) #negative coeffs - positive coeffs

        if (curr_cmin > last_cmin - tol):
            break
        else:
            last_cmin = curr_cmin

    psi = np.ones(un)
    for k in range(p):
        psi = np.multiply(psi, the[cum_r[k] + Un[:,k]])

    return(psi, the, curr_cmin)

# (Khatri-Rao) Alternating minimization completion
def krcomp(X, Y, r, rank, lpar = 1, tol = 1e-6, verbose = True):

    # Compute derived parameters
    n = X.shape[0]
    p = len(r)
    cum_r = np.insert(np.cumsum(r), 0, 0)

    if (len(X.shape) == 1):
        # Convext X into wide indices
        wideX = False
        Xo = X.copy()
        X = np.array(np.unravel_index(Xo, r)).T+1
    else:
        # X already has wide indices
        wideX = True

    # Initialize bookkeeping variables
    prnt_count = 0
    last_objVal = 1e6
    cnt = 0
    
    # Best point to date
    the_q = 2 * np.random.uniform(0,np.max(Y)**(1/p),((np.sum(r), rank)))-1

    # Setup timer
    elapsed_time = 0
    last_time = time.process_time()
    last_time = time.time()
    prnt_time = last_time - 10
    
    # Run alternating minimization
    the = the_q.copy()
    is_true = True
    while is_true:
        cnt += 1
        for ind in range(p):
            A = np.zeros((n+rank*r[ind], rank*r[ind]))
            A[n:(n+rank*r[ind]),:] = np.sqrt(lpar)*np.eye(rank*r[ind])
            b = np.hstack((Y, np.zeros((rank*r[ind],))))

            fflag = True
            for ink in range(p):
                if (ink == ind):
                    continue
                
                if (fflag):
                    for inr in range(rank):
                        A[np.arange(n), inr*r[ind]+X[:,ind]-1] = the[cum_r[ink]+X[:,ink]-1,inr]
                    fflag = False
                else:
                    for inr in range(rank):
                        A[np.arange(n), inr*r[ind]+X[:,ind]-1] *= the[cum_r[ink]+X[:,ink]-1,inr]
                
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            for inr in range(rank):
                the[cum_r[ind]:cum_r[ind+1],inr] = x[inr*r[ind]:(inr+1)*r[ind]]
            #res = A[0:n,:] @ x - b[0:n]
            res = A @ x - b
            objVal = np.dot(res,res)/n
                
        if (objVal > last_objVal - tol):
            is_true = False
            
        if (verbose & (prnt_count % 20 == 0)):
            print("")
            print("   Objective   |   Iters   Time")
            print("")

        curr_time = time.process_time()
        curr_time = time.time()
        elapsed_time = curr_time - last_time
        if (verbose & (prnt_count % 20 == 0 or not is_true or prnt_time <= curr_time - 5)):
            prnt_time = curr_time
            prnt_count += 1
            print("%12.3e     %6i %6is" % 
                (objVal, cnt, elapsed_time))
        
        last_objVal = objVal
    
    if verbose:
        print("\n")
        print("Solution found (tolerance %7.2e)" % (tol))
        print("Best objective %10.6e" % (objVal))

    psi_q = np.zeros(np.prod(r))
    for ind in range(rank):
        the_n = the[cum_r[0]:cum_r[1],ind]
        for k in range(1,p):
            the_n = np.tensordot(the_n, the[cum_r[k]:cum_r[k+1],ind], axes = 0)

        psi_q += the_n.flatten()
        
    if not wideX:
        X = Xo
        
    return(psi_q)

def nonten(X, Y, r, rng, lpar = 1, tol = 1e-6, verbose = True):
    """
    X: (n, ) the indices of known entries in the flatten version of the true tensor
    Y: (n, ) values of known entries corresponding to the indices in X
    r: (r_1, ..., r_p) dimension of the truth tensor
    lpar: lambda parameter in eqn (8)
    """
    
    # Setup timer
    elapsed_time = 0
    last_time = time.process_time()
    last_time = time.time()
    prnt_time = last_time - 10

    # Compute derived parameters
    n = X.shape[0]
    p = len(r)
    rho = np.sum(r)
    cum_r = np.insert(np.cumsum(r), 0, 0)
    
    if (len(X.shape) == 1):
        # X already has flat indices
        wideX = False
    else:
        # Convert X into flat indices
        wideX = True
        Xo = X.copy()
        X = np.ravel_multi_index((Xo.T-1).tolist(), r)

    # Variables for projected polytope 
    # see last two sentences in the secend paragraph of the section 4.3
    [uinds, ucnt] = np.unique(X, return_counts=True) # uinds: (sorted) unique indices of known entries
    un = len(uinds) # unique number of known entries
    Un = np.zeros((un, p), dtype=int) # unique coordinate indices of known entries
    Xn = np.zeros(n, dtype=int) # i-th element is the index of the i-th sample
    imup = un/(2*lpar*np.max(ucnt)/n)
    
    # Variables for the linearized optimization problem
    m = Model()
    #E3m.Params.OutputFlag = 0
    
    m.Params.Method=2
    # m.Params.LogToConsole = 0
    #m.Params.OutputFlag=1
    # m.Params.LogFile = 'IP_Log3'
   

# Optimize

# Iterate over all found solutions
    ind_vec = np.zeros(p, dtype=int)
    d = {} 
    inds = [0]*un 
    psi_count=0
    for cnt in range(un):
        Xn[X == uinds[cnt]] = cnt # original X that the unique indices' ith value represents, which psi it is in dec var
        ind_vec = np.unravel_index(uinds[cnt], r) #obtaining x1,x2,...,xp for this unique sample
        Un[cnt,:] = ind_vec #building the list of index for the samples
        d[ind_vec]=cnt
        psi_count += 1
        inds[cnt]=ind_vec
    d_k=d

    cstr,cstr2,num_psi=0,0,0
    for k in range(p): #total number of psi^k's
        num_psi += len(np.unique(Un[:,k:],axis=0))
    
    #csr_array((data, (row_ind, col_ind)), [shape=(M, N)]) where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
    var = m.addMVar(int(num_psi + 2*rho), vtype = GRB.BINARY) 
    # psi^k + theta + v
    psi = var[0:un] # variable in FW, the direction to move along with
    the = var[num_psi:num_psi+rho]
    var[:num_psi+rho].setAttr('VType', 'c')
    var[:num_psi+rho].setAttr('UB', 1)
    var[:num_psi+rho].setAttr('LB', -1)
    data,row_ind,col_ind = np.zeros(3*4*num_psi),np.zeros(3*4*num_psi) ,np.zeros(3*4*num_psi)   
    
    for k in range(p-1): 
        if k!=0: 
            inds = d_k.keys()
        d = {}
        for ind_k in inds:
            psi_k = d_k[ind_k]
            ind_k_plus_1 = ind_k[1:]
            if ind_k_plus_1 not in d.keys():
                d[ind_k_plus_1] = psi_count
                psi_count += 1
            psi_k_plus_1= d[ind_k_plus_1]
            data[3*cstr:3*cstr+12]=[-1,-1,-1,-1,1,1,1,1,-1,1,-1,1]
            for i in range(4):
                row_ind[3*cstr:3*cstr+3]=[cstr,cstr,cstr]
                col_ind[3*cstr:3*cstr+3]=[psi_k, psi_k_plus_1,num_psi+ cum_r[k] + ind_k[0]]
                cstr += 1
        d_k=d
    A = sp.csr_matrix((data[:cstr*3], (row_ind[:cstr*3], col_ind[:cstr*3]))
                      , shape=(cstr, num_psi + 2*rho))
    b = np.ones(cstr) 
    m.addConstr(A @ var <= b) # the matrix form of the constraints in eqn (13)
    data,row_ind,col_ind = np.zeros(2*len(d_k.keys())) ,np.zeros(2*len(d_k.keys())),np.zeros(2*len(d_k.keys()))
    #build the equality constraint for theta_x_p and psi^p, cstr count restart to 1
    for ind_p in d_k.keys():
        data[2*cstr2:2*cstr2+2]=[1,-1] #psi^p,theta_x_p
        row_ind[2*cstr2:2*cstr2+2] = [cstr2,cstr2]
        psi_p = d_k[ind_p]
        col_ind[2*cstr2:2*cstr2+2]=[psi_p,num_psi+ cum_r[p-1] + ind_p[0]]
        cstr2 += 1
    A = sp.csr_matrix((data, (row_ind, col_ind))
                      , shape=(cstr2,num_psi + 2*rho))
    b = np.zeros(cstr2) 
    m.addConstr(A @ var == b)
    data = np.zeros(2*rho) 
    row_ind = np.zeros(2*rho)
    col_ind = np.zeros(2*rho)
    for i in range(rho):
        data[2*i:2*i+2]=[1,-2] #the_x_k, v_x_k
        row_ind[2*i:2*i+2] = [i,i]
        col_ind[2*i:2*i+2] = [num_psi + i,num_psi + rho + i ]
    A = sp.csr_matrix((data, (row_ind, col_ind)), shape=(rho, num_psi + 2*rho))
    b = np.zeros(rho) - 1
    m.addConstr(A @ var == b)
    m.update()

    # m.write('nonten.lp')

    # Initialize bookkeeping variables
    iter_count = 0
    prnt_count = 0 # print count
    sigd_count = 0 # simplex gradient descent (in BCG paper)
    orcl_count = 0 # oracle count
    ip_count = 0 # integer program count 
    bestbd = 0 # best lower bound for obj func in eqn 8
    as_drops = 0 # active set drop (in BCG paper)
    m._gap = float('inf') # Phi_0 in Line 1, but we haven't really initialized the gap estimate until the Integer LP at the first iteration
    last_gap = float('inf')
    alt_times = []

    # Best point to date
    # Initialization
    Pts = np.ones((un,1)) # (projected) active vertex set (Proj_U(S_t) in Line 2). Elements (psi) are the vertices of Proj_U(C_1)
    Vts = np.ones((np.sum(r),1)) # (non-projected) active vertex set (S_t in Line 2). Elements (theta which can be used to recover psi) are the vertices of C_1
    psi_q = np.ones(un) # the current iterate x_t (in Proj_U(C_1)) which is a cvx combination of elements in Pts using lamb as the coefficients
    the_q = np.ones(np.sum(r))
    lamb = np.array([[1]]) # convex comb coefficients to get the current iterate

    ### BCG ###
    is_true = True
    while is_true:
        iter_count += 1
        # calculate linearized cost
        c = np.zeros(un) # partial derivatives of the obj function w.r.t. each known entry. 
        
        for ind in range(n):    
            c[Xn[ind]] += -2/n*(Y[ind] - lpar*psi_q[Xn[ind]]) # assign the derivative w.r.t. the known entry with the i-th flatten index of the tensor to the i-th element
        pro = np.dot(lpar*c,Pts) # grad(f(x_t))v
        psi_a = Pts[:,np.argmax(pro)] # v_t_A (Line 4)
        psi_f = Pts[:,np.argmin(pro)] # v_t_FW-S (Line 5)
        
        if (np.dot(lpar*c, np.subtract(psi_a,psi_f)) >= m._gap): # Line 6
            ### Simplex Gradient Descent ###
            sigd_count += 1
            d = pro - np.sum(pro)/Pts.shape[1] # line 3, Projection onto the hyperplane of the probability simplex}
            
            if (np.equal(d,0).all()): # line 4
                as_size = Pts.shape[1]
                
                psi_q = Pts[:,0]
                Pts = Pts[:,0]
                Vts = Vts[:,0]
                lamb = np.array([[1]])
                #(Pts, Vts, lamb) = prune(Pts, Vts, psi_q)
                as_drops += as_size - Pts.shape[1]
            else:
                eta = np.divide(lamb,d[:,None]) # line 7
                eta = np.min(eta[d > 0]) # line 7

                # Equivalent to psi_n = Pts @ (lamb - eta*d)
                psi_n = psi_q - eta*(Pts @ d) # line 8, psi_n is y
                #psi_n = Pts @ (lamb.flatten() - eta*d)
                res = Y - lpar*psi_n[Xn]
                fn = np.dot(res,res)/n # fn is f(y)

                if (objVal >= fn): # line 9
                    psi_q = psi_n # line 10
                    objVal = fn
                    as_size = Pts.shape[1]
                    lamb = lamb - eta*d[:,None]
                    inds = lamb.flatten() > 0
                    Pts = Pts[:, inds]
                    Vts = Vts[:, inds]
                    lamb = lamb[inds]/np.sum(lamb[inds])
                    #(Pts, Vts, lamb) = prune(Pts, Vts, psi_q)
                    as_drops += as_size - Pts.shape[1]
                else:
                    grap = Pts @ d
                    gam = -np.dot(Y/lpar-psi_q[Xn], grap[Xn])/np.dot(grap[Xn], grap[Xn])
                    psi_q = psi_q - gam*(Pts @ d)
                    lamb = lamb - gam*d[:,None]
            
        else:
            ### Weak Separation ###
            orcl_count += 1
            
            m._cmin = np.dot(lpar*c,psi_q) # <grad(f(x_0)), x_0>
            if (iter_count == 1):
                # solve linearized (integer) optimization problem
                ip_count += 1
                m.setObjective(lpar*c @ psi) # minimization
                m._oracle = "FullIP"
                m.optimize() 
             
                # for c in m.getConstrs():
                #     print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
                # for v in m.getVars():
                #     print(f'\t{v.varname} ≥ {v.LB}')
                #     print(f'\t{v.varname} ≤ {v.UB}')

                if m.Status == GRB.INFEASIBLE:
                    print('INFEASIBLE')
                    m.computeIIS()
                    print('\nThe following constraints and variables are in the IIS:')
                    for c in m.getConstrs():
                        if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

                # for v in m.getVars():
                #     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                #     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
                m.display()

                psi_n = psi.X
                the_n = the.X
                # for c in m.getConstrs():
                #    print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
                
                
                m._gap = (m._cmin - m.objVal)/2
              
            else:
                oflg = True # True if we do not find an "improving point"
                altmin_count = 0
                out_count = 0
                best_cmin = float('inf')
                last_cmin = float('inf')
                while (oflg and out_count < 1000):
                    ### Heuristic: Alternating Minimization ###
                    alt_start = time.process_time()
                    alt_start = time.time()

                    altmin_count = 0
                    #while (oflg and altmin_count < 100):  # abort altmin
                    altmin_count += 1
                    out_count += 1
                    if (altmin_count == 1) & (out_count == 0):
                        the_n = the.X
                    elif (altmin_count == 2) & (out_count == 0):
                        the_n = -1* the.X
                    elif last_cmin < best_cmin:
                        the_n = the_b
                    else:
                        the_n = np.round(rng.uniform(size=np.sum(r)))
                        the_n = 2*1.0*(the_n < 0.5) -1
                    (psi_n, the_n, last_cmin) = altmin(r, lpar, p, tol, m._cmin, m._gap, c, the_n, Un)
                    
                    #print(m._cmin - last_cmin,m._gap)
                    if (m._cmin - last_cmin > m._gap): # first case in the output of Weak Separation Oracle
                        m._oracle = "AltMin"
                        oflg = False
                    elif (last_cmin < best_cmin):
                        best_cmin = last_cmin
                        psi_b = psi_n
                        the_b = the_n                        
                    # improve the gap estimate when certain conditions hold
                    if oflg and m._cmin - best_cmin > (objVal - bestbd)/2:
                        m._gap = (objVal - bestbd)/2
                        psi_n = psi_b
                        the_n = the_b
                        m._oracle = "AltMin"
                        oflg = False
                    # else:
                    #     out_count += 1
                    alt_end = time.process_time()
                    alt_end = time.time()
                    alt_time = alt_end - alt_start
                    alt_times.append(alt_time)
                
                if oflg:
                    #PASS BEST SOLUTION SO FAR TO MIP SOLVER WHEN NEEDED
                    print('ip called')
                    ip_count += 1
                    psi.Start = psi_b
                    the.Start = the_b
                    m.setObjective(lpar*c @ psi)
                    m._oracle = "FullIP"
                    m.optimize(callback)
                    m.display()
             
                    psi_n = psi.X
                    the_n = the.X
                
                    # for c in m.getConstrs():
                    #     print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')
                    # print('entries that are not -1 or 1',len(psi_n[(psi_n!=1) & (psi_n!=-1)]))
                    # print('entries that are not -1 or 1',len(the_n[(the_n!=1) & (the_n!=-1)]))
                    
              
                    if (m._cmin - m.objVal < m._gap): # second case in the output of the weak separation oracle
                        m._gap = m._gap/2 # line 13
                        # MAYBE UPDATE GAP USING FULLIP SOLUTION?!?!?!

            Pts = np.hstack((Pts,psi_n[:,None]))
            Vts = np.hstack((Vts,the_n[:,None]))
            lamb_q = np.vstack((lamb,0))
            lamb_n = np.vstack((np.zeros(lamb.shape),1))
          
            (psi_q, lamb, gam) = golden(Xn, Y, psi_q, psi_n, lamb_q, lamb_n, lpar) # find the optimal cvx combination of the current iterate and the new vertex
            

        res = Y - lpar*psi_q[Xn]
        objVal = np.dot(res,res)/n
        bestbd = np.max([bestbd, objVal - 2*m._gap])
        #bestbd = np.max([bestbd, objVal - 2*m._gap, objVal - 8*imup*m._gap**2])
        as_size = Pts.shape[1]
        
        if (2*m._gap < tol or (objVal - bestbd) < tol):
        #if (2*m._gap < tol or (objVal - bestbd) < tol or 8*imup*m._gap**2 < tol):
            is_true = False

        if (verbose & (prnt_count % 20 == 0)):
            print("")
            print("   Active Sets   |           Objective Bounds            |         Work")
            print("  Size    Drops  |  Incumbent       BestBd       AddGap  |  SiGD  IntPrg   Time")
            print("")

        curr_time = time.process_time()
        curr_time = time.time()
        elapsed_time = curr_time - last_time
        if (verbose & (prnt_count % 20 == 0 or not is_true or prnt_time <= curr_time - 5 or 0.9*last_gap > m._gap)):
            prnt_time = curr_time
            prnt_count += 1
            last_gap = m._gap
            print(" %5i   %6i  %12.3e %12.3e %12.3e   %6s %6s %6is" % 
                (as_size, as_drops, objVal, bestbd, objVal - bestbd, sigd_count, ip_count, elapsed_time))
            
    if verbose:
        print("\n")
        print("Optimal solution found (tolerance %7.2e)" % (tol))
        print("Best objective %10.6e, best bound %10.6e, additive gap %10.6e" % 
              (objVal, bestbd, objVal - bestbd))
    
    # recover the solution which is a cvx comb of 
    psi_q = np.zeros(np.prod(r))
    for ind in range(as_size):
        the_n = Vts[cum_r[0]:cum_r[1],ind]
        for k in range(1,p):
            the_n = np.tensordot(the_n, Vts[cum_r[k]:cum_r[k+1],ind], axes = 0)

        psi_q += lamb[ind]*the_n.flatten()
    sol = lpar*psi_q
    if wideX:
        X = Xo
    return (sol, iter_count, sigd_count, ip_count, as_size, as_drops, sum(alt_times))

def predict(psi_q, X, r):

    # Compute derived parameters
    p = len(r)
    
    if (X.shape[1] == p):
        # Convert X into flat indices
        return(psi_q[np.ravel_multi_index((X.T-1).tolist(), r)])
    else:
        # X already has flat indices
        return(psi_q[X])
