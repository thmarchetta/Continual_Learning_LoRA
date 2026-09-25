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
        self.dataset = argsDict['dataset'] #dataset used for training. Choose between "synthetic", "MNIST"
        self.method=argsDict['method'] #training that one would like to launch. Choose between "standard", "Sco-Standard", "LoRA", "Sco-LoRA", "only_LoRA", "full_rank"
        self.scoring_method=argsDict['scoring_method'] #scoring method for selecting rows to freeze. Choose between "SDGM", "inv_SDGM", "re-use"
        self.rows_to_freeze=argsDict['rows_to_freeze'] #number of rows to freeze if we use a scoring procedure
        self.informed_initialization=argsDict['informed_initialization'] #Choose to initialize student with overlap given by theory
        self.gamma = argsDict['gamma'] #additional learning rate of LoRA
        self.alpha = argsDict['alpha'] # ratio num_examples/input_size
        self.alpha_W = argsDict['alpha_W'] #learning rate of student first layer
        self.alpha_H = argsDict['alpha_H'] #learning rate of student heads
        self.alpha_a = argsDict['alpha_a'] #learning rate of LoRA A matrix
        self.alpha_b = argsDict['alpha_b'] #learning rate of LoRA B matrix
        self.P_test = argsDict['P_test'] #Number of examples for test set
        self.device = argsDict['device'] #device, must be set to CPU due to online learning
        self.training = argsDict['training'] #Set to true if you want to launch training of a net
        self.theory = argsDict['theory'] #Set to true if you want to compute theory
        self.seed = argsDict['seed'] #random seed for reproducibility
        self.teacher_initialization = argsDict['teacher_initialization'] #choose between "graded" or "committee" initialization of teacher heads
        self.teacher_hl_initialization = argsDict['teacher_hl_initialization'] #choose between "graded" or "committee" initialization of hidden layer rows
        self.teacher_activation_function= argsDict['teacher_activation_function'] #choose between erf (with theory) or ReLU (only exps)
        self.student_activation_function= argsDict['student_activation_function'] #choose between erf (with theory) or ReLU, sigmoid (only exps)
        self.student_initialization= argsDict['student_initialization'] #choose between "specializing" or "unspecializing"
        self.path_to_res_folder = argsDict['path_to_res_folder'] #path to results folder
        self.integration_step = argsDict['integration_step'] #integration step for ODE solver

def loadJson(path_to_json):
    import json
    with open(path_to_json, 'r') as f:
        argsDict = json.load(f)
    return args(argsDict)


def set_seed(seed=42):
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_OP_init(stud,data,args):
    W0 = stud.fc1.weight.detach().numpy().copy()
    A0 = stud.A.weight.detach().numpy().copy()
    W_T1 = data.teacher_1.fc1.weight.detach().numpy().copy()
    W_T2 = data.teacher_2.fc1.weight.detach().numpy().copy()
    
    Q0   = ((W0 @ W0.T) / args.N).copy()
    R0   = ((W0 @ W_T1.T) / args.N).copy()
    U0   = ((W0 @ W_T2.T) / args.N).copy()
    T0   = ((W_T1 @ W_T1.T) / args.N).copy()
    S0   = ((W_T2 @ W_T2.T) / args.N).copy()
    V0   = ((W_T1 @ W_T2.T) / args.N).copy()
    Xi0  = ((W0 @ A0.T) / args.N).copy()
    Lam0 = ((W_T1 @ A0.T) / args.N).copy()
    Gam0 = ((W_T2 @ A0.T) / args.N).copy()
    Phi0 = ((A0 @ A0.T) / args.N).copy()
    h1_0  = stud.head_1.weight.detach().numpy().reshape((args.K,)).copy()
    h2_0  = stud.head_2.weight.detach().numpy().reshape((args.K,)).copy()
    v_T1  = data.teacher_1.fc2.weight.detach().numpy().copy()
    v_T2  = data.teacher_2.fc2.weight.detach().numpy().copy()
    B0   = stud.B.weight.detach().numpy().copy()
    
    #Q0 = 0.001* np.identity(args.K)
    #T0 = np.identity(args.M)
    #S0 = np.identity(args.M)
    #V0 = args.rho * np.identity(args.M)
    #R0 = np.ones((args.K, args.M)) /args.N
    #U0 = np.ones((args.K, args.M)) /args.N
    #Xi0 = np.ones((args.K, args.L)) /args.N
    #Lam0 = np.ones((args.M, args.L)) /args.N
    #Gam0  = np.ones((args.M,args.L)) /args.N
    #Phi0 = np.ones((args.L, args.L)) /args.N
    #Ha0 = np.ones((args.K)) /args.N
    #Hb0 = np.ones((args.K)) /args.N
    #v_T1 = np.ones((args.M)) /args.N
    #v_T2 = np.ones((args.M)) /args.N
    #B0 = np.ones((args.K, args.L)) /args.N
    
    OP_init = {
    "Q0": Q0,
    "R0": R0,
    "U0": U0,
    "T0": T0,
    "S0": S0,
    "V0": V0,
    "Xi0": Xi0,
    "Lam0": Lam0,
    "Gam0": Gam0,
    "Phi0": Phi0,
    "h1_0": h1_0,
    "h2_0": h2_0,
    "v_T1": v_T1,
    "v_T2": v_T2,
    "B0" : B0,
    }
    
    
    return OP_init

def get_metadata(args):
    
    VALID_METHODS = {"standard", "Sco-standard", "LoRA", "Sco-LoRA", "only_LoRA", "full_rank"}

    if args.method not in VALID_METHODS:
        raise ValueError(f"Method '{args.method}' is not implemented or has a bad spelling. "
                     f"Valid options are: {', '.join(VALID_METHODS)}")
    if args.rows_to_freeze > args.L :
        print(f"WARNING : Rows to freeze must be smaller than L. Currently : L={args.L}, TO FREEZE : {args.rows_to_freeze}. \n If needed, set TO FREEZE to {args.L}")
        args.rows_to_freeze = args.L
    
    if args.dataset == "MNIST" or args.dataset == "fMNIST":
        print("Working with MNIST. Setting N=784")
        args.N = int(784)
    if args.dataset == "CIFAR":
        print("Working with CIFAR. Setting N=1024")
        args.N = int(1024)
    
    metadata = {
        "N": args.N,
        "M": args.M,
        "K": args.K,
        "rho": args.rho,
        "dataset": args.dataset,
        "method": args.method,
        "scoring_method": args.scoring_method,
        "rows_to_freeze": args.rows_to_freeze,
        "informed_initialization": args.informed_initialization,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "L": args.L,
        "alpha_W": args.alpha_W,
        "alpha_H": args.alpha_H,
        "alpha_a": args.alpha_a,
        "alpha_b": args.alpha_b,
        "teacher_initialization": args.teacher_initialization,
        "teacher_hl_initialization": args.teacher_hl_initialization,
        "teacher_activation_function": args.teacher_activation_function,
        "student_activation_function": args.student_activation_function,
        "student_initialization": args.student_initialization,
        "seed": args.seed
    }
    
    return metadata