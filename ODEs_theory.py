from numba import njit
import numpy as np
from logger import ExperimentLogger

def solve_ODES(args, OP_init, OP_init_switch, logs_ODES):
    
    gamma = args.beta/np.sqrt(args.L) #Effective learning rate of LoRA updates
    num_steps = int( args.alpha / args.integration_step)

    #Parameters for first half of training
    Q = OP_init["Q0"] #stud-stud
    R = OP_init["R0"] #Stud - T1
    U = OP_init["U0"] #Stud - T2
    V = OP_init["V0"] # T1 - T2
    T = OP_init["T0"] # T1 - T1
    S = OP_init["S0"] # T2 - T2
    Ha = OP_init["Ha0"] #Stud First head
    Hb = OP_init["Hb0"] #Stud Second head
    v_a = OP_init["v_a"] #T1 head
    v_b = OP_init["v_b"] #T2 head
    D = OP_init["D0"] # LoRA low rank matrix
    #Parameters for second half of training 
    Gamma = OP_init["Gam0"].copy()
    Phi = OP_init["Phi0"].copy()
    Lambda = OP_init["Lam0"].copy()

    #first half of training  
    for step in range(num_steps):

      Q, R, U, Ha, Hb, test_loss_1, test_loss_2 = solve_ODES_standard(Q, R, U, T, V, S, Ha, Hb, v_a, v_b, args,task=1)
      
      logs_ODES.log_many(step=step, test_loss_1=test_loss_1, test_loss_2=test_loss_2,
      Q=Q.copy(), R=R.copy(), U=U.copy(), Ha=Ha.copy())
  
    #Used to compute forgetting and transfer 
    test_1_switch = logs_ODES.last("test_loss_1")
    test_2_switch = logs_ODES.last("test_loss_2")

    ##Using initializations of experiment to have perfect match, as the ODEs are capricious
    Ha = OP_init_switch["Ha0_switch"]
    Q = OP_init_switch["Q0_switch"]
    R = OP_init_switch["R0_switch"]
    U = OP_init_switch["U0_switch"]
    G = OP_init_switch["G0_switch"]
    if args.A_only : 
        D = OP_init_switch["D0_switch"]

    #Second half of training 
    for step in range(num_steps):

      if args.LoRA:
        G, D, Hb, Phi,Gamma,Lambda, test_loss_1, test_loss_2 = solve_ODES_LORA( Q, R, U, T, V, S, G, D, Ha, Hb, v_a, v_b, Phi,Gamma,Lambda, gamma, args, task=2)
      else :
        Q, R, U, Ha, Hb, test_loss_1, test_loss_2 = solve_ODES_standard(Q, R, U, T, V, S, Ha, Hb, v_a, v_b, args,task=2)
          
      forgetting = test_loss_1 - test_1_switch
      transfer = test_2_switch - test_loss_2

      if args.LoRA:
        logs_ODES.log_many(
            step=num_steps + step,
            test_loss_1=test_loss_1,
            test_loss_2=test_loss_2,
            forgetting=forgetting,
            transfer=transfer,

            G =G.copy(), Phi=Phi.copy(), Gamma=Gamma.copy(), Lambda=Lambda.copy(), Hb=Hb.copy(), D=D.copy(),
            v_a=v_a.copy(), v_b=v_b.copy(),)
      else:
        logs_ODES.log_many(
            step=num_steps+step,
            test_loss_1=test_loss_1,
            test_loss_2=test_loss_2,
            forgetting=forgetting,
            transfer=transfer,

            Q=Q.copy(), R=R.copy(), U=U.copy(),
            T=T.copy(), V=V.copy(), S=S.copy(),
            Ha=Ha.copy(), Hb=Hb.copy(),
            v_a=v_a.copy(), v_b=v_b.copy(),
        )

    ExperimentLogger.append_to_file(
    f"results/ODES_K={args.K}_M={args.M}.npy",
    logs_ODES
    )
    return logs_ODES
    
def solve_ODES_standard(Q, R, U, T, V, S, Ha, Hb, v_a, v_b, args,task=1):
    if task ==1:
        H=Ha
        v=v_a
    else : 
        H=Hb
        v=v_b
    R1 = np.concatenate([Q , R, U],axis=1)
    R2 = np.concatenate([R.T,T,V],axis=1)
    R3 = np.concatenate([U.T,V.T,S],axis=1)
    C = np.concatenate( [R1,R2,R3],axis=0 )
    
    loss1 = compute_loss_standard(C, Ha, v_a, args.K, args.M, task=1)
    loss2 = compute_loss_standard(C, Hb, v_b, args.K, args.M, task=2)

      
    R = R + args.integration_step * update_R(C, args.alpha_W,v,args.K,args.M, H, task)
    Q = Q + args.integration_step * update_Q(C, args.alpha_W,v,args.K,args.M, H, task)
    U = U + args.integration_step * update_U(C, args.alpha_W,v,args.K,args.M, H, task)
    if task==1:
        Ha = Ha + args.integration_step * update_Ha(C,Ha,args.alpha_H,v_a,args.K,args.M)
    elif task==2 :
        Hb = Hb + args.integration_step * update_Hb_standard(C,Hb,args.alpha_H,v_b,args.K,args.M)
    return Q, R, U, Ha, Hb, loss1,loss2
    
def solve_ODES_LORA(Q, R, U, T, V, S, G, D, Ha, Hb, v_a, v_b, Phi,Gamma,Lambda, gamma, args, task=2):
    
      R1 = np.concatenate([ Q, Q + gamma*G @ D.T , U , G],axis=1)
      R2 = np.concatenate([ Q.T + gamma * D @ G.T, Q + gamma*(G @ D.T + D @ G.T) + (gamma**2) * D @ Phi @ D.T , U + gamma * D @ Gamma.T , G + gamma * D @ Phi.T ],axis=1)
      R3 = np.concatenate([ U.T , U.T + gamma *  Gamma @ D.T , S, Gamma ],axis=1)
      R4 = np.concatenate([ G.T , G.T + gamma * Phi @ D.T , Gamma.T , Phi ],axis=1)
      C = np.concatenate([R1,R2,R3,R4],axis=0)
      
      loss2 = compute_loss_LoRA(C, Hb, v_b, args.K, args.M)
    
      R1_a = np.concatenate([ Q + gamma*(G @ D.T + D @ G.T) + (gamma**2) * D @ Phi @ D.T , R + gamma * D @ Lambda.T , U + gamma * D @ Gamma.T],axis=1)
      R2_a = np.concatenate([ R.T + gamma * Lambda @ D.T, T , V ],axis=1)
      R3_a = np.concatenate([ U.T + gamma * Gamma @ D.T , V.T, S ],axis=1)
      C_a = np.concatenate([R1_a,R2_a,R3_a],axis=0)

      loss1 = compute_loss_standard(C_a, Ha, v_a, args.K, args.M, task=1)
    
      Phi = Phi + args.integration_step * update_Phi(C, Hb, args.alpha_a, v_b, args.K, args.M, args.L, D, gamma)
      Gamma = Gamma + args.integration_step * update_Gamma(C,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      Lambda = Lambda + args.integration_step * update_Lambda(C_a,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      G = G + args.integration_step * update_G(C,Hb,args.alpha_a,v_b,args.K,args.M,args.L,D,gamma)
      Hb = Hb + args.integration_step * update_Hb(C,Hb,args.alpha_H,v_b,args.K,args.M)
      if args.A_only:
          D = D
      else:
          D = D + args.integration_step * update_D(C,Hb,args.alpha_b,v_b,args.K,args.M,args.L,gamma)
          
        
      return G, D, Hb, Phi,Gamma,Lambda, loss1, loss2

@njit
def I2_val_numba(C, a, b):
    cij = C[a, b]
    denom1 = np.sqrt(1.0 + C[a, a])
    denom2 = np.sqrt(1.0 + C[b, b])
    return 2.0 * np.arcsin(cij / denom1 / denom2) / np.pi


@njit
def I3_val_numba(C, a, b, c):
    C00 = C[a,a]
    C01 = C[a,b]
    C02 = C[a,c]
    C12 = C[b,c]
    C22 = C[c,c]

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
def update_R(C, eta, v, K, M, H, task):
    ng = np.zeros((K, M))
    if task == 1:
        delay = K
    else:
        delay = K+M
    for i in range(K):
        for n in range(M):
            sum1 = 0.0
            for m in range(M):
                sum1 += H[i] * I3_val_numba(C, i, K + n, delay + m) * v[m]
            sum2 = 0.0
            for k in range(K):
                sum2 += H[i] * I3_val_numba(C, i, K + n, k) * H[k]
            ng[i, n] = sum1 - sum2
    return eta * ng

@njit
def update_Q(C, eta, v, K, M, H, task):
    ng1 = np.zeros((K, K))
    ng2 = np.zeros((K, K))
    ng3 = np.zeros((K, K))
    if task == 1:
        delay = K
    else:
        delay = K+M
    for i in range(K):
        for k in range(K):
            sum1 = 0.0
            for m in range(M):
                sum1 += H[i] * v[m] * I3_val_numba(C, i, k, delay + m)
            sum2 = 0.0
            for j in range(K):
                sum2 += H[i] * H[j] * I3_val_numba(C, i, k, j)
            ng1[i, k] = sum1 - sum2
            sum3 = 0.0
            for m in range(M):
                sum3 += H[k] * v[m] * I3_val_numba(C, k, i, delay + m)
            sum4 = 0.0
            for j in range(K):
                sum4 += H[k] * H[j] * I3_val_numba(C, k, i, j)
            ng2[i, k] = sum3 - sum4
            sum5 = 0.0
            for j in range(K):
                for l in range(K):
                    sum5 += H[j] * H[l] * I4_val_numba(C, i, k, j, l)
            sum6 = 0.0
            for m in range(M):
                for n in range(M):
                    sum6 += v[m] * v[n] * I4_val_numba(C, i, k, delay + m, delay + n)
            sum7 = 0.0
            for j in range(K):
                for m in range(M):
                    sum7 += H[j] * v[m] * I4_val_numba(C, i, k, j, delay + m)
            ng3[i, k] = H[k] * H[i] * (sum5 + sum6 - 2.0 * sum7)
    return eta * (ng1 + ng2) + (eta * eta) * ng3

@njit
def update_U(C, eta, v, K, M, H, task):
    ng = np.zeros((K, M))
    if task == 1:
        delay = K
    else:
        delay = K+M
    for i in range(K):
        for p in range(M):
            sum1 = 0.0
            for m in range(M):
                sum1 += H[i] * I3_val_numba(C, i, K + M + p, delay + m) * v[m]
            sum2 = 0.0
            for k in range(K):
                sum2 += H[i] * I3_val_numba(C, i, K + M + p, k) * H[k]
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
def update_Hb_standard(C, h, eta, v, K, M, ):
    ng = np.zeros((K,))
    for i in range(K):
        sum1 = 0.0
        for m in range(M):
            sum1 += v[m] * I2_val_numba(C, K + M + m, i)
        sum2 = 0.0
        for k in range(K):
            sum2 += h[k] * I2_val_numba(C,k, i)
        ng[i] = sum1 - sum2
    return eta * ng

@njit
def update_Hb(C, h, eta, v, K, M, ):
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
def compute_loss_LoRA(C, Hb, v_b, K, M):
    loss = 0.0
    for i in range(K):
        for j in range(K):
            loss += Hb[i] * Hb[j] * I2_val_numba(C, K + i, K + j)
    for m in range(M):
        for n in range(M):
            loss += v_b[m] * v_b[n] * I2_val_numba(C, 2 * K + m, 2 * K + n)
    for i in range(K):
        for m in range(M):
            loss -= 2.0 * Hb[i] * v_b[m] * I2_val_numba(C, K + i, 2 * K + m)
    return loss / 2.0

@njit
def compute_loss_standard(C, H, v, K, M,task=1):
    loss = 0.0
    if task == 1:
        delay = K
    else:
        delay = K+M
    for i in range(K):
        for j in range(K):
            loss += H[i] * H[j] * I2_val_numba(C, i, j)
    for m in range(M):
        for n in range(M):
            loss += v[m] * v[n] * I2_val_numba(C, delay + n, delay + m)
    for i in range(K):
        for m in range(M):
            loss -= 2.0 * H[i] * v[m] * I2_val_numba(C, i, delay + m)
    return loss / 2.0
