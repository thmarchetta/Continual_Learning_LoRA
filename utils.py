import os
import numpy as np
import torch
import pandas as pd
import csv

class args:
    def __init__(self, argsDict):
        self.N = argsDict['N'] #Number of examples
        self.K = argsDict['K'] #Number of student heads
        self.M = argsDict['M'] #Number of teacher heads
        self.L = argsDict['L'] #Low_dim of LoRA
        self.rho = argsDict['rho'] #task similarity
        self.LoRA = argsDict['LoRA'] #Whether to use LoRA or not
        self.wt = argsDict['wt'] #Whether to use the full rank matrix or not
        self.A_only = argsDict['A_only'] #Whether to learn only the A matrix of LoRA, using a selection rule for B or both A and B
        self.beta = argsDict['beta'] #additional learning rate of LoRA
        self.alpha = argsDict['alpha'] # ratio num_examples/input_size
        self.alpha_W = argsDict['alpha_W'] #learning rate of student first layer
        self.alpha_H = argsDict['alpha_H'] #learning rate of student heads
        self.alpha_a = argsDict['alpha_a'] #learning rate of LoRA A matrix
        self.alpha_b = argsDict['alpha_b'] #learning rate of LoRA B matrix
        self.P_test = argsDict['P_test'] #Number of examples for test set
        self.device = argsDict['device'] #device, must be set to CPU due to online learning
        self.seed = argsDict['seed'] #random seed for reproducibility
        self.path_to_res_folder = argsDict['path_to_res_folder'] #path to results folder
        self.integration_step = argsDict['integration_step'] #integration step for ODE solver

def loadJson(path_to_json):
    import json
    with open(path_to_json, 'r') as f:
        argsDict = json.load(f)
    return args(argsDict)

def save_to_file(df, args, ODES=False):

    if os.path.isdir(args.path_to_res_folder) == False: # create the results folder if not there
        try:
            os.makedirs(args.path_to_res_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % args.path_to_res_folder)
            raise
    
    df = pd.DataFrame.from_dict(df)
    if ODES :
        print("Saving ODEs results...")
        filename = f"/sim_K={args.K}_M={args.M}_ODES.csv"
    else :
        print("Saving training results...")
        filename = f"/sim_K={args.K}_M={args.M}.csv"
    if os.path.isfile(args.path_to_res_folder + filename) == False:
        with open(args.path_to_res_folder + filename, mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(df.keys().to_list())
    df.to_csv(args.path_to_res_folder + filename, mode = 'a', index = False, header = None)
    return

def set_seed(seed=42):
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def initialize_dictionary(args):
    log = {}

    # scalar histories
    log["steps"] = []
    log["test_loss1"] = []
    log["test_loss2"] = []
    log["forgetting"] = []
    log["transfer"] = []

    # fixed metadata (optional but cleaner here than later)
    log["N"] = []
    log["M"] = []
    log["K"] = []
    log["L"] = []
    log["rho"] = []
    log["alpha"] = []
    log["beta"] = []
    log["method"] = []

    # LoRA norm
    for i in range(1, args.K + 1):
        log[f"norm_LoRA_{i}{i}"] = []

    # overlaps
    for i in range(1, args.K + 1):
        for j in range(1, args.M + 1):

            log[f"overlap_t1_{i}{j}"] = []
            log[f"overlap_t2_{i}{j}"] = []

            log[f"overlap_t1_LoRA_{i}{j}"] = []
            log[f"overlap_t2_LoRA_{i}{j}"] = []

            log[f"overlap_t1_wt_{i}{j}"] = []
            log[f"overlap_t2_wt_{i}{j}"] = []

            log[f"overlap_t1_full_{i}{j}"] = []
            log[f"overlap_t2_full_{i}{j}"] = []
    
    #Order parameters
    for i in range(1, args.K + 1):
        log[f'ha_{i}'] = []
        log[f'hb_{i}'] = []
        for j in range(1, args.M + 1):
            log[f'R_{i}{j}'] = []
            log[f'U_{i}{j}'] = []
        for j in range(1, args.K + 1):
            log[f'Q_{i}{j}'] = []
        for j in range(1, args.L + 1):
            log[f'G_{i}{j}'] = []
    for i in range(1, args.L + 1):
        for j in range(1, args.L + 1):
            log[f'Phi_{i}{j}'] = []
    for i in range(1, args.M + 1):
        for j in range(1, args.L + 1):
            log[f'Gamma_{i}{j}'] = []
            log[f'Lambda_{i}{j}'] = []
    for i in range(1, args.K + 1):
        for j in range(1, args.L + 1):
            log[f'D_{i}{j}'] = []
        for j in range(1, args.K + 1):
            log[f'Q_{i}{j}'] = []

    return log

def initialize_ODE_dictionnary(args):
    logs_ODES = {}
    logs_ODES["test_loss1"] = []
    logs_ODES["test_loss2"] = []
    logs_ODES["steps"] = []
    logs_ODES["forgetting"] = []
    logs_ODES["transfer"] = []
    for i in range (1, args.K+1):
        logs_ODES[f"ha_{i}"] = []
        logs_ODES[f"hb_{i}"] = []
    for i in range (1, args.K+1):
        for j in range (1,args.M+1):
            logs_ODES[f"R_{i}{j}"] = []
            logs_ODES[f"U_{i}{j}"] = []
    
    for i in range (1, args.K+1):
        for j in range (1,args.K+1):
            logs_ODES[f"Q_{i}{j}"] = []

            
    for i in range (1, args.K+1):
        for j in range (1,args.L+1):
            logs_ODES[f"G_{i}{j}"] = []
            logs_ODES[f'D_{i}{j}'] = []
    for i in range (1, args.L+1):
        for j in range (1,args.L+1):
            logs_ODES[f"Phi_{i}{j}"] = []
    for i in range (1, args.M+1):
        for j in range (1, args.L+1):
            logs_ODES[f"Gamma_{i}{j}"] = []
            logs_ODES[f"Lambda_{i}{j}"] = []

    return logs_ODES

def save_ODES_in_dictionnary(logs_ODES, args,task=1,
                             Ha=None, R=None, Q=None, U=None,
                             G= None, D=None,Phi=None, Gamma=None, Lambda=None, Hb=None):
    if task == 1:
        #OPs for first half of training
        for i in range (1, args.K+1):
            logs_ODES[f"ha_{i}"].append(Ha[i-1].item())
                
        for i in range (1, args.K+1):
            for j in range (1,args.M+1):
                logs_ODES[f"R_{i}{j}"].append(R[i-1, j-1].item())
                logs_ODES[f"U_{i}{j}"].append(U[i-1, j-1].item())
    
        for i in range (1, args.K+1):
            for j in range (1,args.K+1):
                logs_ODES[f"Q_{i}{j}"].append(Q[i-1, j-1].item())
            
        #OPs for second half of training
        for i in range (1,args.K+1):
            for j in range (1,args.L+1):
                logs_ODES[f"G_{i}{j}"].append(0)
                logs_ODES[f'D_{i}{j}'].append(0)
        for i in range(1, args.K+1):
            logs_ODES[f"hb_{i}"].append(0)
        for i in range (1, args.L+1):
            for j in range (1,args.L+1):
                logs_ODES[f"Phi_{i}{j}"].append(0)
        for i in range (1, args.M+1):
            for j in range (1, args.L+1):
                logs_ODES[f"Gamma_{i}{j}"].append(0)
                logs_ODES[f"Lambda_{i}{j}"].append(0)
                
    elif task == 2:   
        for i in range (1, args.K+1):
            logs_ODES[f"ha_{i}"].append(0)
                
        for i in range (1, args.K+1):
            for j in range (1,args.M+1):
                logs_ODES[f"R_{i}{j}"].append(0)
                logs_ODES[f"U_{i}{j}"].append(0)
    
        for i in range (1, args.K+1):
            for j in range (1,args.K+1):
                logs_ODES[f"Q_{i}{j}"].append(0)
            
        #OPs for second half of training
        for i in range (1,args.K+1):
            for j in range (1,args.L+1):
                logs_ODES[f"G_{i}{j}"].append(G[i-1, j-1].item())
                logs_ODES[f'D_{i}{j}'].append(D[i-1, j-1].item())
        for i in range(1, args.K+1):
            logs_ODES[f"hb_{i}"].append(Hb[i-1].item())
        for i in range (1, args.L+1):
            for j in range (1,args.L+1):
                logs_ODES[f"Phi_{i}{j}"].append(Phi[i-1, j-1].item())
        for i in range (1, args.M+1):
            for j in range (1, args.L+1):
                logs_ODES[f"Gamma_{i}{j}"].append(Gamma[i-1, j-1].item())
                logs_ODES[f"Lambda_{i}{j}"].append(Lambda[i-1, j-1].item())
    else :
        raise ValueError("task must be either 1 or 2")
    return logs_ODES
    

def save_overlaps_in_dictionnary(logs, args, overlap_t1, overlap_t2, task=1, 
                                 overlap_t1_LoRA=None, overlap_t2_LoRA=None, overlap_t1_wt=None, overlap_t2_wt=None, overlap_t1_full=None, overlap_t2_full=None, Norm_LoRA=None):
    if task ==1 :
        for i in range(1,args.K+1):
            logs[f'norm_LoRA_{i}{i}'].append(0)
            for j in range(1,args.M+1):
                logs[f'overlap_t1_{i}{j}'].append(overlap_t1[i-1, j-1].item())
                logs[f'overlap_t2_{i}{j}'].append(overlap_t2[i-1, j-1].item())
                logs[f'overlap_t1_LoRA_{i}{j}'].append(0)
                logs[f'overlap_t2_LoRA_{i}{j}'].append(0)
                logs[f'overlap_t1_wt_{i}{j}'].append(0)
                logs[f'overlap_t2_wt_{i}{j}'].append(0)     
                logs[f'overlap_t1_full_{i}{j}'].append(overlap_t1[i-1, j-1].item())
                logs[f'overlap_t2_full_{i}{j}'].append(overlap_t2[i-1, j-1].item())

    elif task == 2:
        for i in range(1,args.K+1):
          logs[f'norm_LoRA_{i}{i}'].append(Norm_LoRA[i-1].item())
          for j in range(1,args.M+1):
            logs[f'overlap_t1_{i}{j}'].append(overlap_t1[i-1, j-1].item())
            logs[f'overlap_t2_{i}{j}'].append(overlap_t2[i-1, j-1].item())
            logs[f'overlap_t1_LoRA_{i}{j}'].append(overlap_t1_LoRA[i-1,j-1].item())
            logs[f'overlap_t2_LoRA_{i}{j}'].append(overlap_t2_LoRA[i-1,j-1].item())
            logs[f'overlap_t1_wt_{i}{j}'].append(overlap_t1_wt[i-1,j-1].item())
            logs[f'overlap_t2_wt_{i}{j}'].append(overlap_t2_wt[i-1,j-1].item()) 
            logs[f'overlap_t1_full_{i}{j}'].append(overlap_t1_full[i-1,j-1].item())
            logs[f'overlap_t2_full_{i}{j}'].append(overlap_t2_full[i-1,j-1].item())
        
    else :
        raise ValueError("task must be either 1 or 2")
    return logs

def save_order_parameters_in_dictionnary(logs, args, task=1, 
                                         Q=None, R=None, U=None, Ha=None,
                                         D=None, Hb=None, G=None, Phi=None, Gamma=None, Lambda=None):
    if task ==1 :
        #Ops for first half of training
        for i in range(1,args.K+1):
            logs[f'ha_{i}'].append(Ha[i-1].item())
            for j in range(1,args.M+1):
                logs[f'R_{i}{j}'].append(R[i-1, j-1].item())
                logs[f'U_{i}{j}'].append(U[i-1, j-1].item())
        for i in range(1,args.K+1):
            for j in range(1,args.K+1):
                logs[f'Q_{i}{j}'].append(Q[i-1, j-1].item())
                
        #OPs for second half of training
        for i in range(1,args.K+1):
            logs[f'hb_{i}'].append(0)
            for j in range(1,args.L+1):
                logs[f'G_{i}{j}'].append(0)
                logs[f'D_{i}{j}'].append(0)
        for i in range(1, args.L+1):
            for j in range(1,args.L+1):
                logs[f'Phi_{i}{j}'].append(0)
        for i in range(1, args.M+1):
            for j in range(1, args.L+1):
                logs[f'Gamma_{i}{j}'].append(0)
                logs[f'Lambda_{i}{j}'].append(0)
        
    elif task ==2 :
                #Ops for first half of training
        for i in range(1,args.K+1):
            logs[f'ha_{i}'].append(0)
            for j in range(1,args.M+1):
                logs[f'R_{i}{j}'].append(0)
                logs[f'U_{i}{j}'].append(0)
        for i in range(1,args.K+1):
            for j in range(1,args.K+1):
                logs[f'Q_{i}{j}'].append(0)
                
        #OPs for second half of training
        for i in range(1,args.K+1):
            logs[f'hb_{i}'].append(Hb[i-1].item())
            for j in range(1,args.L+1):
                logs[f'G_{i}{j}'].append(G[i-1, j-1].item())
                logs[f'D_{i}{j}'].append(D[i-1, j-1].item())
        for i in range(1, args.L+1):
            for j in range(1,args.L+1):
                logs[f'Phi_{i}{j}'].append(Phi[i-1, j-1].item())
        for i in range(1, args.M+1):
            for j in range(1, args.L+1):
                logs[f'Gamma_{i}{j}'].append(Gamma[i-1, j-1].item())
                logs[f'Lambda_{i}{j}'].append(Lambda[i-1, j-1].item())
        
    else : 
        raise ValueError("task must be either 1 or 2")
    return logs