from numba import njit
import numpy as np
import utils 

def solve_ODES(args, OP_init, gamma):
    num_steps = int( args.alpha / args.integration_step)

    #Parameters for first half of training
    Q = OP_init["Q0"].copy() #stud-stud
    R = OP_init["R0"].copy() #Stud - T1
    U = OP_init["U0"].copy() #Stud - T2
    V = OP_init["V0"].copy() # T1 - T2
    T = OP_init["T0"].copy() # T1 - T1
    S = OP_init["S0"].copy() # T2 - T2
    Ha = OP_init["Ha0"].copy() #Stud First head
    Hb = OP_init["Hb0"].copy() #Stud Second head
    v_a = OP_init["v_a"].copy() #T1 head
    v_b = OP_init["v_b"].copy() #T2 head
    D = OP_init["D0"].copy() # LoRA low rank matrix

    logs_ODES = utils.initialize_ODE_dictionnary(args)


    #first half of training  
    for step in range(num_steps):
      if step % 100 == 0:
        print(f"Integration step {step} over {num_steps}")

      R1 = np.concatenate([Q , R, U],axis=1)
      R2 = np.concatenate([R.T,T,V],axis=1)
      R3 = np.concatenate([U.T,V.T,S],axis=1)
      C = np.concatenate( [R1,R2,R3],axis=0 )
    
      lossA = compute_lossA_first(C, Ha, v_a, args.K, args.M)
      lossB = compute_lossB_first(C, Hb, v_b, args.K, args.M)

      logs_ODES["test_loss1"].append(lossA)
      logs_ODES["test_loss2"].append(lossB)
      
      R = R + args.integration_step * update_R(C,Ha,args.alpha_W,v_a,args.K,args.M)
      Q = Q + args.integration_step * update_Q(C,Ha,args.alpha_W,v_a,args.K,args.M)
      U = U + args.integration_step * update_U(C,Ha,args.alpha_W,v_a,args.K,args.M)
      Ha = Ha + args.integration_step * update_Ha(C,Ha,args.alpha_H,v_a,args.K,args.M)
      
      logs_ODES["steps"].append(step)
      logs_ODES = utils.save_ODES_in_dictionnary(logs_ODES, args,task=1, Ha=Ha, R=R, Q=Q, U=U)

    #Parameters for second half of training 
    Gamma = OP_init["Gam0"].copy()
    Phi = OP_init["Phi0"].copy()
    Lambda = OP_init["Lam0"].copy()

    ##-----DEBUG : Using initializations for second half of training as well, to check if the updates are correct -----##
    Q = OP_init["Q0_switch"].copy()
    U = OP_init["U0_switch"].copy()
    R = OP_init["R0_switch"].copy()
    G = OP_init["G0_switch"].copy()
    Ha = OP_init["Ha0_switch"].copy()
    D = OP_init["D0_switch"].copy()
    ###



    #Second half of training 
    for step in range(num_steps):

      if step % 100 == 0:
        print(f"Integration step {step} over {num_steps}")

      R1 = np.concatenate([ Q, Q + gamma*G @ D.T , U , G],axis=1)
      R2 = np.concatenate([ Q.T + gamma * D @ G.T, Q + gamma*(G @ D.T + D @ G.T) + (gamma**2) * D @ Phi @ D.T , U + gamma * D @ Gamma.T , G + gamma * D @ Phi.T ],axis=1)
      R3 = np.concatenate([ U.T , U.T + gamma *  Gamma @ D.T , S, Gamma ],axis=1)
      R4 = np.concatenate([ G.T , G.T + gamma * Phi @ D.T , Gamma.T , Phi ],axis=1)
      C = np.concatenate([R1,R2,R3,R4],axis=0)
    
      lossB = compute_lossB_second(C, Hb, v_b, args.K, args.M)
    
      R1_a = np.concatenate([ Q + gamma*(G @ D.T + D @ G.T) + (gamma**2) * D @ Phi @ D.T , R + gamma * D @ Lambda.T , U + gamma * D @ Gamma.T],axis=1)
      R2_a = np.concatenate([ R.T + gamma * Lambda @ D.T, T , V ],axis=1)
      R3_a = np.concatenate([ U.T + gamma * Gamma @ D.T , V.T, S ],axis=1)
      C_a = np.concatenate([R1_a,R2_a,R3_a],axis=0)
    
      lossA = compute_lossA_second(C_a, Ha, v_a, args.K, args.M)
    
      logs_ODES["test_loss1"].append(lossA)
      logs_ODES["test_loss2"].append(lossB)
    
      Phi = Phi + args.integration_step * update_Phi(C, Hb, args.alpha_a, v_b, args.K, args.M, args.L, D, gamma)
      Gamma = Gamma + args.integration_step * update_Gamma(C,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      Lambda = Lambda + args.integration_step * update_Lambda(C_a,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      G = G + args.integration_step * update_G(C,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      Hb = Hb + args.integration_step * update_Hb(C,Hb,args.alpha_H,v_b,args.K,args.M,args.L,D)
      if args.A_only:
          D = D
      else:
          D = D + args.integration_step * update_D(C,Hb,args.alpha_b,v_b,args.K,args.M,args.L,gamma)

      logs_ODES["steps"].append(num_steps + step)
      logs_ODES = utils.save_ODES_in_dictionnary(logs_ODES, args,task=2,
                             G= G, D=D,
                             Phi=Phi, Gamma=Gamma, Lambda=Lambda, Hb=Hb)
      
    logs_ODES['N'] = [args.N] * len(logs_ODES["steps"])
    logs_ODES['M'] = [args.M] * len(logs_ODES["steps"])
    logs_ODES['K'] = [args.K] * len(logs_ODES["steps"])
    logs_ODES['rho'] = [args.rho] * len(logs_ODES["steps"])
    logs_ODES['alpha'] = [args.alpha] * len(logs_ODES["steps"])
    logs_ODES['beta'] = [args.beta] * len(logs_ODES["steps"])
    logs_ODES['L'] = [args.L] * len(logs_ODES["steps"])
    logs_ODES['A_only'] = [args.A_only] * len(logs_ODES["steps"])
    if args.LoRA:
      logs_ODES['method'] = ['LoRA'] * len(logs_ODES["steps"])
    elif args.wt:
      logs_ODES['method'] = ['LoRA_full_rank'] * len(logs_ODES["steps"])
    else:
      logs_ODES['method'] = ['standard'] * len(logs_ODES["steps"])

    return logs_ODES
    
    
@njit
def I2_val_numba(C, a, b):
    cij = C[a, b]
    denom1 = np.sqrt(1.0 + C[a, a])
    denom2 = np.sqrt(1.0 + C[b, b])
    return 2.0 * np.arcsin(cij / denom1 / denom2) / np.pi


@njit
def I3_val_numba(C, a, b, c):
    C00 = C[a,a]
    C11 = C[b,b]
    C22 = C[c,c]
    C02 = C[a,c]
    C01 = C[a,b]
    C12 = C[b,c]

    sDel3 = np.sqrt((1+C00)*(1+C22) - C02*C02)
    num = C12*(1+C00) - C01*C02
    den = 1 + C00
    return (2.0*num)/den/sDel3/np.pi

@njit
def I4_val_numba(C, a, b, c, d):
    Caa = C[a, a]
    Cbb = C[b, b]
    Ccc = C[c, c]
    Cdd = C[d, d]

    Cab = C[a, b]
    Cac = C[a, c]
    Cad = C[a, d]
    Cbc = C[b, c]
    Cbd = C[b, d]
    Ccd = C[c, d]

    Del4 = (1.0 + Caa)*(1.0 + Cbb) - Cab*Cab

    Del0 = (
        Del4*Ccd
        - Cbc*Cbd*(1.0 + Caa)
        - Cac*Cad*(1.0 + Cbb)
        + Cab*Cac*Cbd
        + Cab*Cad*Cbc
    )

    Del1 = (
        Del4*(1.0 + Ccc)
        - (Cbc*Cbc)*(1.0 + Caa)
        - (Cac*Cac)*(1.0 + Cbb)
        + 2.0*Cab*Cac*Cbc
    )

    Del2 = (
        Del4*(1.0 + Cdd)
        - (Cbd*Cbd)*(1.0 + Caa)
        - (Cad*Cad)*(1.0 + Cbb)
        + 2.0*Cab*Cad*Cbd
    )

    return 4.0 * np.asin(Del0 / np.sqrt(Del1*Del2)) \
           / np.pi / np.pi / np.sqrt(Del4)
           
#ODE LGS
@njit
def update_R(C, h, eta, v, K, M):
    ng = np.zeros((K, M))
    delay = K
    for i in range(K):
        for n in range(M):
            sum1 = 0.0
            for m in range(M):
                sum1 += h[i] * I3_val_numba(C, i, K + n, delay + m) * v[m]
            sum2 = 0.0
            for k in range(K):
                sum2 += h[i] * I3_val_numba(C, i, K + n, k) * h[k]
            ng[i, n] = sum1 - sum2
    return eta * ng

@njit
def update_Q(C, h, eta, v, K, M):
    ng1 = np.zeros((K, K))
    ng2 = np.zeros((K, K))
    ng3 = np.zeros((K, K))
    delay = K
    for i in range(K):
        for k in range(K):
            sum1 = 0.0
            for m in range(M):
                sum1 += h[i] * v[m] * I3_val_numba(C, i, k, delay + m)
            sum2 = 0.0
            for j in range(K):
                sum2 += h[i] * h[j] * I3_val_numba(C, i, k, j)
            ng1[i, k] = sum1 - sum2
            sum3 = 0.0
            for m in range(M):
                sum3 += h[k] * v[m] * I3_val_numba(C, k, i, delay + m)
            sum4 = 0.0
            for j in range(K):
                sum4 += h[k] * h[j] * I3_val_numba(C, k, i, j)
            ng2[i, k] = sum3 - sum4
            sum5 = 0.0
            for j in range(K):
                for l in range(K):
                    sum5 += h[j] * h[l] * I4_val_numba(C, i, k, j, l)
            sum6 = 0.0
            for m in range(M):
                for n in range(M):
                    sum6 += v[m] * v[n] * I4_val_numba(C, i, k, delay + m, delay + n)
            sum7 = 0.0
            for j in range(K):
                for m in range(M):
                    sum7 += h[j] * v[m] * I4_val_numba(C, i, k, j, delay + m)
            ng3[i, k] = h[k] * h[i] * (sum5 + sum6 - 2.0 * sum7)
    return eta * (ng1 + ng2) + (eta * eta) * ng3

@njit
def update_U(C, h, eta, v, K, M):
    ng = np.zeros((K, M))
    delay = K
    for i in range(K):
        for p in range(M):
            sum1 = 0.0
            for m in range(M):
                sum1 += h[i] * I3_val_numba(C, i, K + M + p, delay + m) * v[m]
            sum2 = 0.0
            for k in range(K):
                sum2 += h[i] * I3_val_numba(C, i, K + M + p, k) * h[k]
            ng[i, p] = sum1 - sum2
    return eta * ng

@njit
def update_Ha(C, h, eta, v, K, M):
    ng = np.zeros((K,))
    delay = K
    for i in range(K):
        sum1 = 0.0
        for m in range(M):
            sum1 += v[m] * I2_val_numba(C, delay + m, i)
        sum2 = 0.0
        for k in range(K):
            sum2 += h[k] * I2_val_numba(C, k, i)
        ng[i] = sum1 - sum2
    return eta * ng

@njit
def update_D(C, h, eta, v, K, M, L, gamma):
    ng = np.zeros((K, L))
    for j in range(K):
        for s in range(L):
            sum1 = 0.0
            for i in range(K):
                sum1 += h[j] * h[i] * I3_val_numba(C, K + j, 2 * K + M + s, K + i)
            sum2 = 0.0
            for m in range(M):
                sum2 += h[j] * v[m] * I3_val_numba(C, K + j, 2 * K + M + s, 2 * K + m)
            ng[j, s] = gamma * eta * (sum2 - sum1)
    return ng

@njit
def update_Gamma(C, h, eta, v, K, M, L, D, gamma):
    ng = np.zeros((M, L))
    for m in range(M):
        for s in range(L):
            sum1 = 0.0
            for i in range(K):
                for l in range(K):
                    sum1 += h[i] * h[l] * D[i, s] * I3_val_numba(C, K + i, 2 * K + m, K + l)
            sum2 = 0.0
            for i in range(K):
                for n in range(M):
                    sum2 += h[i] * v[n] * D[i, s] * I3_val_numba(C, K + i, 2 * K + m, 2 * K + n)
            ng[m, s] = sum2 - sum1
    return eta * gamma * ng

@njit
def update_Lambda(C, h, eta, v, K, M, L, D, gamma):
    ng = np.zeros((M, L))
    for m in range(M):
        for s in range(L):
            sum1 = 0.0
            for i in range(K):
                for l in range(K):
                    sum1 += h[i] * h[l] * D[i, s] * I3_val_numba(C, i, K + m, l)
            sum2 = 0.0
            for i in range(K):
                for n in range(M):
                    sum2 += h[i] * v[n] * D[i, s] * I3_val_numba(C, i, K + m, K + M + n)
            ng[m, s] = sum2 - sum1
    return eta * gamma * ng

@njit
def update_G(C, h, eta, v, K, M, L, D, gamma):
    ng = np.zeros((K, L))
    for j in range(K):
        for s in range(L):
            sum1 = 0.0
            for i in range(K):
                for l in range(K):
                    sum1 += h[i] * h[l] * D[i, s] * I3_val_numba(C, K + i, j, K + l)
            sum2 = 0.0
            for i in range(K):
                for m in range(M):
                    sum2 += h[i] * v[m] * D[i, s] * I3_val_numba(C, K + i, j, 2 * K + m)
            ng[j, s] = sum2 - sum1
    return eta * gamma * ng

@njit
def update_Hb(C, h, eta, v, K, M, L, D):
    ng = np.zeros((K,))
    for j in range(K):
        sum1 = 0.0
        for m in range(M):
            sum1 += v[m] * I2_val_numba(C, 2 * K + m, K + j)
        sum2 = 0.0
        for i in range(K):
            sum2 += h[i] * I2_val_numba(C, K + i, K + j)
        ng[j] = sum1 - sum2
    return eta * ng

@njit
def update_Phi(C, h, eta, v, K, M, L, D, gamma):
    ng = np.zeros((L, L))
    for s in range(L):
        for t in range(s, L):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            sum4 = 0.0
            for i in range(K):
                idx_i = K + i
                for m in range(M):
                    idx_t = 2 * K + M + t
                    idx_m = 2 * K + m
                    sum1 += h[i] * v[m] * D[i, s] * I3_val_numba(C, idx_i, idx_t, idx_m)
            for i in range(K):
                idx_i = K + i
                for k in range(K):
                    idx_t = 2 * K + M + t
                    idx_k = K + k
                    sum2 += h[i] * h[k] * D[i, s] * I3_val_numba(C, idx_i, idx_t, idx_k)
            for i in range(K):
                idx_i = K + i
                for m in range(M):
                    idx_s = 2 * K + M + s
                    idx_m = 2 * K + m
                    sum3 += h[i] * v[m] * D[i, t] * I3_val_numba(C, idx_i, idx_s, idx_m)
            for i in range(K):
                idx_i = K + i
                for k in range(K):
                    idx_s = 2 * K + M + s
                    idx_k = K + k
                    sum4 += h[i] * h[k] * D[i, t] * I3_val_numba(C, idx_i, idx_s, idx_k)
            sum5 = 0.0
            sum6 = 0.0
            sum7 = 0.0
            for i in range(K):
                idx_i = K + i
                for j in range(K):
                    idx_j = K + j
                    for k in range(K):
                        idx_k = K + k
                        for l in range(K):
                            idx_l = K + l
                            sum5 += h[i] * h[j] * h[k] * h[l] * D[i, t] * D[j, s] * I4_val_numba(C, idx_i, idx_j, idx_k, idx_l)
            for i in range(K):
                idx_i = K + i
                for j in range(K):
                    idx_j = K + j
                    for l in range(K):
                        idx_l = K + l
                        for m in range(M):
                            idx_m = 2 * K + m
                            sum6 += h[i] * h[j] * h[l] * v[m] * D[i, t] * D[j, s] * I4_val_numba(C, idx_i, idx_j, idx_l, idx_m)
            for i in range(K):
                idx_i = K + i
                for j in range(K):
                    idx_j = K + j
                    for m in range(M):
                        idx_m = 2 * K + m
                        for n in range(M):
                            idx_n = 2 * K + n
                            sum7 += h[i] * h[j] * v[n] * v[m] * D[i, t] * D[j, s] * I4_val_numba(C, idx_i, idx_j, idx_m, idx_n)
            ng[t, s] = eta * gamma * (sum1 - sum2 + sum3 - sum4) + (eta * eta) * (gamma * gamma) * (sum5 - 2.0 * sum6 + sum7)
            ng[s, t] = ng[t, s]
    return ng

@njit
def compute_lossA_first(C, Ha, v_a, K, M):
    loss = 0.0
    for i in range(K):
        for k in range(K):
            loss += Ha[i] * Ha[k] * I2_val_numba(C, i, k)
    for m in range(M):
        for n in range(M):
            loss += v_a[m] * v_a[n] * I2_val_numba(C, K + n, K + m)
    for k in range(K):
        for n in range(M):
            loss -= 2.0 * Ha[k] * v_a[n] * I2_val_numba(C, k, K + n)
    return loss / 2.0

@njit
def compute_lossB_first(C, Hb, v_b, K, M):
    loss = 0.0
    for i in range(K):
        for k in range(K):
            loss += Hb[i] * Hb[k] * I2_val_numba(C, i, k)
    for m in range(M):
        for n in range(M):
            loss += v_b[m] * v_b[n] * I2_val_numba(C, K + M + n, K + M + m)
    for k in range(K):
        for n in range(M):
            loss -= 2.0 * Hb[k] * v_b[n] * I2_val_numba(C, k, K + M + n)
    return loss / 2.0

@njit
def compute_lossB_second(C, Hb, v_b, K, M):
    loss = 0.0
    for i in range(K):
        for j in range(K):
            loss += Hb[i] * Hb[j] * I2_val_numba(C, K + i, K + j)
    for i in range(K):
        for m in range(M):
            loss -= 2.0 * Hb[i] * v_b[m] * I2_val_numba(C, K + i, 2 * K + m)
    for m in range(M):
        for n in range(M):
            loss += v_b[m] * v_b[n] * I2_val_numba(C, 2 * K + m, 2 * K + n)
    return loss / 2.0

@njit
def compute_lossA_second(C_a, Ha, v_a, K, M):
    loss = 0.0
    for i in range(K):
        for j in range(K):
            loss += Ha[i] * Ha[j] * I2_val_numba(C_a, i, j)
    for i in range(K):
        for m in range(M):
            loss -= 2.0 * Ha[i] * v_a[m] * I2_val_numba(C_a, i, K + m)
    for m in range(M):
        for n in range(M):
            loss += v_a[m] * v_a[n] * I2_val_numba(C_a, K + m, K + n)
    return loss / 2.0
