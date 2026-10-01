from numba import njit
import numpy as np
from logger import ExperimentLogger


def initialize_B_freeze_K_minus_L(Ha, C, v_T1, args):

    num_keep = args.K - args.rows_to_freeze
    if num_keep <= 0:
        idx = np.array([], dtype=np.int64)
    elif num_keep >= args.K:
        idx = np.arange(args.K, dtype=np.int64)
    else:
        HA_abs = np.abs(Ha)
        if args.scoring_method == "SDGM":
            idx = np.argpartition(HA_abs, -num_keep)[-num_keep:]
        elif args.scoring_method == "inv_SDGM":
            idx = np.argpartition(HA_abs, num_keep)[:num_keep]
        elif args.scoring_method == "re-use":
            base_loss = compute_loss_standard(
                C, Ha, v_T1, args.K, args.M, task=1
            )
            importance_scores = np.zeros(args.K)
            for k in range(args.K):
                Ha_ablated = Ha.copy()
                Ha_ablated[k] = 0.0
                ablated_loss = compute_loss_standard(
                    C, Ha_ablated, v_T1, args.K, args.M, task=1
                )
                importance_scores[k] = ablated_loss - base_loss
            idx = np.argpartition(
                importance_scores, -num_keep
            )[-num_keep:]
        else:
            raise ValueError(
                f"Unknown scoring_method: {args.scoring_method}"
            )
    B = np.zeros((args.K, args.L), dtype=np.float64)
    remaining = np.setdiff1d(np.arange(args.K), idx)
    if len(remaining) > 0:
        B[remaining, np.arange(len(remaining)) % args.L] = 1.0
    print("B after procedure for standard+SDGM:", B)
    return B, idx

def initialize_B_freeze_L(Ha, C, v_T1, args):
    if not 0 <= args.rows_to_freeze <= args.K:
        raise ValueError(
            f"rows_to_freeze must satisfy 0 <= rows_to_freeze <= K, "
            f"got rows_to_freeze={args.rows_to_freeze}, K={args.K}"
        )

    num_active = args.K - args.rows_to_freeze

    B = np.zeros((args.K, args.L), dtype=np.float64)

    if num_active == 0:
        return B, np.array([], dtype=np.int64)

    if num_active == args.K:
        idx = np.arange(args.K, dtype=np.int64)

    else:
        HA_abs = np.abs(Ha)

        if args.scoring_method == "SDGM":
            # Freeze largest |Ha| -> keep smallest |Ha| active
            idx = np.argpartition(
                HA_abs, num_active
            )[:num_active]

        elif args.scoring_method == "inv_SDGM":
            # Freeze smallest |Ha| -> keep largest |Ha| active
            idx = np.argpartition(
                HA_abs, -num_active
            )[-num_active:]

        elif args.scoring_method == "re-use":
            base_loss = compute_loss_standard(
                C, Ha, v_T1, args.K, args.M, task=1
            )

            importance_scores = np.zeros(args.K)

            for k in range(args.K):
                Ha_ablated = Ha.copy()
                Ha_ablated[k] = 0.0

                ablated_loss = compute_loss_standard(
                    C, Ha_ablated, v_T1,
                    args.K, args.M, task=1
                )

                importance_scores[k] = ablated_loss - base_loss

            idx = np.argpartition(
                importance_scores, num_active
            )[:num_active]

        else:
            raise ValueError(
                f"Unknown scoring_method: {args.scoring_method}"
            )

    # idx = active/plastic rows
    B[idx, np.arange(num_active) % args.L] = 1.0
    return B, idx

def solve_ODES(args, OP_init, logs_ODES):
    lr_LoRA = args.gamma/np.sqrt(args.L)
    #lr_LoRA = args.gamma
    num_steps = int( args.alpha / args.integration_step)

    #Parameters for first half of training
    Q = OP_init["Q0"] #stud-stud
    R = OP_init["R0"] #Stud - T1
    U = OP_init["U0"] #Stud - T2
    V = OP_init["V0"] # T1 - T2
    T = OP_init["T0"] # T1 - T1
    S = OP_init["S0"] # T2 - T2
    h1 = OP_init["h1_0"] #Stud First head
    h2 = OP_init["h2_0"] #Stud Second head
    v_T1 = OP_init["v_T1"] #T1 head
    v_T2 = OP_init["v_T2"] #T2 head
    #Parameters for second half of training 
    Xi = OP_init["Xi0"] #Stud - A
    Gamma = OP_init["Gam0"] #T2 - A
    Phi = OP_init["Phi0"] #A - A
    Lambda = OP_init["Lam0"] # T1- A
    B = OP_init["B0"] # B
    
    #Compute first covariance matrix and save the losses 
    R1 = np.concatenate([Q , R, U],axis=1)
    R2 = np.concatenate([R.T,T,V],axis=1)
    R3 = np.concatenate([U.T,V.T,S],axis=1)
    C = np.concatenate( [R1,R2,R3],axis=0 )
    
    test_loss_1 = compute_loss_standard(C, h1, v_T1, args.K, args.M, task=1)
    test_loss_2 = compute_loss_standard(C, h2, v_T2, args.K, args.M, task=2)
    logs_ODES.log_many(step=0, test_loss_1=test_loss_1, test_loss_2=test_loss_2)
    

    #first half of training  
    for step in range(num_steps):
        Q, R, U, h1, h2, test_loss_1, test_loss_2, C = solve_ODES_standard(Q, R, U, T, V, S, h1, h2, v_T1, v_T2, args,task=1)
        logs_ODES.log_many(step=step, test_loss_1=test_loss_1, test_loss_2=test_loss_2,
            Q=Q.copy(), R=R.copy(), U=U.copy(), h1=h1.copy())
        
    #Used to compute forgetting and transfer 
    test_1_switch = logs_ODES.last("test_loss_1")
    test_2_switch = logs_ODES.last("test_loss_2")
    
    #Used to select rows to freeze if we use a scoring procedure
    idx = None
    if args.method in {"Sco-LoRA", "Sco-standard"}:
        B, idx= initialize_B_freeze_K_minus_L(h1, C, v_T1, args)
        
    #If we choose to use only LoRA, we set all different teachers-student overlaps to be 0.
    if args.method =="only_LoRA" : 
        print("setting OPS to 0 due to only_LoRA")
        R = np.zeros((args.K, args.M))
        U = np.zeros((args.K, args.M))
        Q = np.zeros((args.K, args.K))
        Xi = np.zeros((args.K, args.L))
        
    #Second half of training 
    for step in range(num_steps):
        if args.method in {"LoRA", "Sco-LoRA", "only_LoRA", "Sco-standard"}:
            Xi, B, h2, Phi,Gamma,Lambda, test_loss_1, test_loss_2 = solve_ODES_LORA( Q, R, U, T, V, S, Xi, B, h1, h2, v_T1, v_T2, Phi,Gamma,Lambda, lr_LoRA, args, task=2)
        else :
            Q, R, U, h1, h2, test_loss_1, test_loss_2, C = solve_ODES_standard(Q, R, U, T, V, S, h1, h2, v_T1, v_T2,args, idx, task=2)

        forgetting = test_loss_1 - test_1_switch
        transfer = test_2_switch - test_loss_2
        logs_ODES.log_many(step=num_steps + step, test_loss_1=test_loss_1, test_loss_2=test_loss_2,
            forgetting=forgetting, transfer=transfer,)

        if args.method in {"LoRA", "Sco-LoRA", "Sco-standard"}:
            logs_ODES.log_many(step=num_steps + step,
                B=B.copy(), Xi =Xi.copy(), Phi=Phi.copy(), Gamma=Gamma.copy(), Lambda=Lambda.copy(), h2=h2.copy(),)
        
        if args.method == "only_LoRA":
            logs_ODES.log_many(step=num_steps+step,
                Q=Q.copy(), R=R.copy(),  U=U.copy(), h2=h2.copy(),)
        else:
            logs_ODES.log_many(step=num_steps+step,
                Q=Q.copy(), R=R.copy(),  U=U.copy(), h2=h2.copy(),)


    ExperimentLogger.append_to_file(f"results/ODES_K={args.K}_M={args.M}.npy",logs_ODES)
    return logs_ODES
    
def solve_ODES_standard(Q, R, U, T, V, S, h1, h2, v_T1, v_T2, args, idx=None, task=1):
    if task ==1:
        H=h1
        v=v_T1
    else : 
        H=h2
        v=v_T2
    R1 = np.concatenate([Q , R, U],axis=1)
    R2 = np.concatenate([R.T,T,V],axis=1)
    R3 = np.concatenate([U.T,V.T,S],axis=1)
    C = np.concatenate( [R1,R2,R3],axis=0)
    
    loss1 = compute_loss_standard(C, h1, v_T1, args.K, args.M, task=1)
    loss2 = compute_loss_standard(C, h2, v_T2, args.K, args.M, task=2)

    R = R + args.integration_step * update_R(C, args.alpha_W,v,args.K,args.M, H, task)
    Q = Q + args.integration_step * update_Q(C, args.alpha_W,v,args.K,args.M, H, task)
    U = U + args.integration_step * update_U(C, args.alpha_W,v,args.K,args.M, H, task)
        
    if task==1:
        h1 = h1 + args.integration_step * update_H(C,h1,args.alpha_H,v_T1,args.K,args.M, task=1)
    elif task==2:
        h2 = h2 + args.integration_step * update_H(C,h2,args.alpha_H,v_T2,args.K,args.M, task=2)
    return Q, R, U, h1, h2, loss1,loss2, C
    
def solve_ODES_LORA(Q, R, U, T, V, S, Xi, B, h1, h2, v_T1, v_T2, Phi,Gamma,Lambda, lr_LoRA, args, task=2):
    
    R1 = np.concatenate([ Q, Q + lr_LoRA*Xi @ B.T, U, Xi, R],axis=1)
    R2 = np.concatenate([ Q.T + lr_LoRA * B @ Xi.T, Q + lr_LoRA*(Xi @ B.T + B @ Xi.T) + (lr_LoRA**2) * B @ Phi @ B.T, U + lr_LoRA * B @ Gamma.T, Xi + lr_LoRA * B @ Phi.T, R + lr_LoRA*B@ Lambda.T ],axis=1)
    R3 = np.concatenate([ U.T, U.T + lr_LoRA *  Gamma @ B.T, S, Gamma, V.T ],axis=1)
    R4 = np.concatenate([ Xi.T, Xi.T + lr_LoRA * Phi @ B.T, Gamma.T, Phi, Lambda.T ],axis=1)
    R5 = np.concatenate([R.T, R.T + lr_LoRA * Lambda @ B.T, V, Lambda, T], axis=1)
    C = np.concatenate([R1,R2,R3,R4, R5],axis=0)

    loss1 = compute_loss_LoRA(C, h1, v_T1, args.K, args.M, args.L, task=1)    
    loss2 = compute_loss_LoRA(C, h2, v_T2, args.K, args.M, args.L, task=2)
    
    Phi = Phi + args.integration_step * update_Phi(C, h2, args.alpha_a, v_T2, args.K, args.M, args.L, B, lr_LoRA)
    Gamma = Gamma + args.integration_step * update_Gamma(C,h2,args.alpha_a,v_T2,args.K,args.M,args.L,B,lr_LoRA)
    Lambda = Lambda + args.integration_step * update_Lambda(C,h2,args.alpha_a,v_T2,args.K,args.M,args.L,B,lr_LoRA)
    Xi = Xi + args.integration_step * update_Xi(C,h2,args.alpha_a,v_T2,args.K,args.M,args.L,B,lr_LoRA)
    h2 = h2 + args.integration_step * update_h2(C,h2,args.alpha_H,v_T2,args.K,args.M)
    if args.method in["Sco-LoRA", "Sco-standard"]:
        B = B
    else:
        B = B + args.integration_step * update_B(C,h2,args.alpha_b,v_T2,args.K,args.M,args.L,lr_LoRA)

    return Xi, B, h2, Phi, Gamma, Lambda, loss1, loss2

@njit
def I2_val_numba(C, a, b):
    cij = C[a, b]
    denom1 = np.sqrt(1.0 + C[a, a])
    denom2 = np.sqrt(1.0 + C[b, b])
    return 2.0 * np.asin(cij / denom1 / denom2) / np.pi

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


#Implementation of the ODEs from "Continual Learning in the Teacher-Student Setup: Impact of Task Similarity", Lee et. Al, https://arxiv.org/abs/2107.04384
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
def update_U(C, eta, v, K, M, H, task, idx=None):
    ng = np.zeros((K, M))
    if task == 1:
        delay = K
    else:
        delay = K+M
    for i in range(K):
        for p in range(M):
            sum1 = 0.0
            for m in range(M):
                sum1 += H[i] * v[m] * I3_val_numba(C, i, K + M + p, delay + m)
            sum2 = 0.0
            for k in range(K):
                sum2 += H[i] * H[k] * I3_val_numba(C, i, K + M + p, k)
            ng[i, p] = sum1 - sum2
            
    if idx is not None :
        for i in range(len(idx)):
            row_idx = idx[i]
            for n in range(M):
                ng[row_idx, n] = 0.0
                
    return eta * ng

@njit
def update_H(C, h, eta, v, K, M,task):
    ng = np.zeros((K,))
    if task ==1 :
        delay = K
    else :
        delay = K+M
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
def update_h2(C, h, eta, v, K, M, ):
    ng = np.zeros((K,))
    for k in range(K):
        sum1 = 0.0
        for p in range(M):
            sum1 += v[p] * I2_val_numba(C, 2*K + p, K + k)
        sum2 = 0.0
        for l in range(K):
            sum2 += h[l] * I2_val_numba(C, K + l, K + k)
        ng[k] = sum1 - sum2
    return eta * ng

@njit
def update_B(C, h, eta, v, K, M, L, lr_LoRA):
    ng = np.zeros((K, L))
    for k in range(K):
        for i in range(L):
            sum1 = 0.0
            for l in range(K):
                sum1 += h[k] * h[l] * I3_val_numba(C, K + k, 2*K+M + i, K + l)
            sum2 = 0.0
            for m in range(M):
                sum2 += h[k] * v[m] * I3_val_numba(C, K + k, 2*K+M + i, 2*K + m)
            ng[k, i] = lr_LoRA * eta * (sum2 - sum1)
    return ng

@njit
def update_Gamma(C, h, eta, v, K, M, L, B, lr_LoRA):
    ng = np.zeros((M, L))
    for p in range(M):
        for i in range(L):
            sum1 = 0.0
            for k in range(K):
                for l in range(K):
                    sum1 += h[k] * h[l] * B[k, i] * I3_val_numba(C, K + k, 2*K + p, K + l)
            sum2 = 0.0
            for k in range(K):
                for q in range(M):
                    sum2 += h[k] * v[q] * B[k, i] * I3_val_numba(C, K + k, 2*K + p, 2*K + q)
            ng[p, i] = sum2 - sum1
    return eta * lr_LoRA * ng

@njit
def update_Lambda(C, h, eta, v, K, M, L, B, lr_LoRA):
    ng = np.zeros((M, L))
    for m in range(M):
        for i in range(L):
            sum1 = 0.0
            for k in range(K):
                for l in range(K):
                    sum1 += h[k] * h[l] * B[k, i] * I3_val_numba(C, K + k, 2*K+M+L + m, K + l)
            sum2 = 0.0
            for k in range(K):
                for p in range(M):
                    sum2 += h[k] * v[p] * B[k, i] * I3_val_numba(C, K + k, 2*K+M+L + m, 2*K + p)
            ng[m, i] = sum2 - sum1
    return eta * lr_LoRA * ng

@njit
def update_Xi(C, h, eta, v, K, M, L, B, lr_LoRA):
    ng = np.zeros((K, L))
    for k in range(K):
        for i in range(L):
            sum1 = 0.0
            for l in range(K):
                for kprime in range(K):
                    sum1 += h[l] * h[kprime] * B[l, i] * I3_val_numba(C, K + l, k, K + kprime)
            sum2 = 0.0
            for l in range(K):
                for p in range(M):
                    sum2 += h[l] * v[p] * B[l, i] * I3_val_numba(C, K + l, k, 2*K + p)
            ng[k, i] = sum2 - sum1
    return eta * lr_LoRA * ng

@njit
def update_Phi(C, h, eta, v, K, M, L, B, lr_LoRA):
    ng = np.zeros((L, L))
    for j in range(L):
        for i in range(j, L):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            sum4 = 0.0
            for k in range(K):
                for p in range(M):
                    sum1 += h[k] * v[p] * B[k, i] * I3_val_numba(C, K + k, 2*K+M + j, 2*K + p)
                for l in range(K):
                    sum2 += h[k] * h[l] * B[k, i] * I3_val_numba(C, K + k, 2*K+M + j, K + l)
            for k in range(K):
                for p in range(M):
                    sum3 += h[k] * v[p] * B[k, j] * I3_val_numba(C, K + k, 2*K+M + i, 2*K + p)
                for l in range(K):
                    sum4 += h[k] * h[l] * B[k, j] * I3_val_numba(C, K + k, 2*K+M + i, K + l)
            sum5 = 0.0
            sum6 = 0.0
            sum7 = 0.0
            for k in range(K):
                for l in range(K):
                    for kprime in range(K):
                        for lprime in range(K):
                            sum5 += h[k] * h[l] * h[kprime] * h[lprime] * B[k, i] * B[l, j] * I4_val_numba(C, K + k, K + l, K + kprime, K + lprime)
                        for p in range(M):
                            sum6 += h[k] * h[l] * h[kprime] * v[p] * B[k, i] * B[l, j] * I4_val_numba(C, K + k, K + l, K + kprime, 2*K + p)
                    for p in range(M):
                        for q in range(M):
                            sum7 += h[k] * h[l] * v[p] * v[q] * B[k, i] * B[l, j] * I4_val_numba(C, K + k, K + l, 2*K + p, 2*K + q)
            ng[i, j] = eta * lr_LoRA * (sum1 - sum2 + sum3 - sum4) + (eta**2) * (lr_LoRA**2) * (sum5 - 2.0 * sum6 + sum7)
            ng[j, i] = ng[i, j]
    return ng

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

@njit
def compute_loss_LoRA(C, h2, v_T2, K, M, L, task=1):
    loss = 0.0
    if task== 1:
        delay= 2*K + M + L
    else:
        delay = 2*K
    for k in range(K):
        for l in range(K):
            loss += h2[k] * h2[l] * I2_val_numba(C, K + k, K + l)
    for p in range(M):
        for q in range(M):
            loss += v_T2[p] * v_T2[q] * I2_val_numba(C, delay + p, delay + q)
    for k in range(K):
        for p in range(M):
            loss -= 2.0 * h2[k] * v_T2[p] * I2_val_numba(C, delay + p, K + k)
    return loss / 2.0