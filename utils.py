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
        self.method=argsDict['method'] #training that one would like to launch. Choose between "standard", "Sco-Standard", "LoRA", "Sco-LoRA", "only_LoRA", "full_rank"
        self.beta = argsDict['beta'] #additional learning rate of LoRA
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
    # Order parameters after initialization of the network
    W0 = stud.fc1.weight.detach().numpy().copy()
    A0 = stud.A.weight.detach().numpy().copy()
    #D0 = stud.B.weight.detach().numpy().copy()
    B_a = data.teacher_1.fc1.weight.detach().numpy().copy()
    B_b = data.teacher_2.fc1.weight.detach().numpy().copy()
    
    OP_init = {
    "Q0": ((W0 @ W0.T) / args.N).copy(),
    "R0": ((W0 @ B_a.T) / args.N).copy(),
    "T0": ((B_a @ B_a.T) / args.N).copy(),
    "U0": ((W0 @ B_b.T) / args.N).copy(),
    "S0": ((B_b @ B_b.T) / args.N).copy(),
    "V0": ((B_a @ B_b.T) / args.N).copy(),
    "Xi0": ((W0 @ A0.T) / args.N).copy(),
    "Lam0": ((B_a @ A0.T) / args.N).copy(),
    "Gam0": ((B_b @ A0.T) / args.N).copy(),
    "Phi0": ((A0 @ A0.T) / args.N).copy(),
    "Ha0": stud.head_1.weight.detach().numpy().reshape((args.K,)).copy(),
    "Hb0": stud.head_2.weight.detach().numpy().reshape((args.K,)).copy(),
    "v_a": data.teacher_1.fc2.weight.detach().numpy().copy(),
    "v_b": data.teacher_2.fc2.weight.detach().numpy().copy(),
    "B0" : stud.B.weight.detach().numpy().copy()
    }
    return OP_init

def get_metadata(args):
    metadata = {
        "N": args.N,
        "M": args.M,
        "K": args.K,
        "rho": args.rho,
        "method": args.method,
        "alpha": args.alpha,
        "beta": args.beta,
        "L": args.L,
        "alpha_W": args.alpha_W,
        "alpha_H": args.alpha_H,
        "alpha_a": args.alpha_a,
        "alpha_b": args.alpha_b
    }
    
    if args.method  != ( "standard" or "Sco-standard" or "LoRA" or "Sco-LoRA" or "Only_LoRA" or "full_rank"):
        raise ValueError("METHOD IS NOT IMPLEMENTED OR HAS A BAD SPELLING ")
    #    
    return metadata