#TODO : WHy is gamma not used in training ?

import utils
import training
import ODEs_theory
import networks
import logger
import numpy as np

argfile = "/home/theom/continual_learning/parameters.json"
args = utils.loadJson(argfile)

#rhos = np.arange(0,1.01, 0.025) #For theory
#rhos = np.arange(0,1.01, 0.1)
#rhos = np.arange(0,1.01, 0.2) #For training
seeds = np.rint(np.linspace(1001, 10000, 10)).astype(int)
#Ls = np.arange(1,11,1)

Ls = [args.L]
rhos = [args.rho]
#seeds = [args.seed]
for seed in seeds:
    for L in Ls:
        for rho in rhos:
            print("seed=", seed)
            print("L=", L)
            print("rho=", rho)
            
            args = utils.loadJson(argfile)
            args.L = L
            args.seed = seed
            args.rho = round(float(rho), 3)
            
            if args.method == "Sco-standard":
                print("Sco-standard method found. Setting LoRA rank to the number of hidden neurons.")
                args.gamma = np.sqrt(args.K)
                args.L = args.K
            print("Method=", args.method)
            metadata = utils.get_metadata(args)
            logs_ODES = logger.ExperimentLogger(metadata)
            logs_training = logger.ExperimentLogger(metadata)
            utils.set_seed(args.seed)
            if args.dataset == "synthetic" :
                data = networks.Data_and_Teachers(args)
            elif args.dataset == "MNIST" :
                data = networks.ContinualMnist(args)
            elif args.dataset == "fMNIST" :
                data = networks.ContinualFashionMNIST(args)
            elif args.dataset == "CIFAR":
                data = networks.ContinualCifar(args)
            else:
                raise ValueError("Not good dataset !")
                
            t1, t2 = data.get_Teachers()
                
            stud = networks.Student(args)
            if args.theory and args.student_activation_function == "erf" and args.teacher_activation_function == "erf" and args.dataset=="synthetic" :
                OP_init = utils.get_OP_init(stud, data, args)
                print("Launching ODEs")
                ODEs_theory.solve_ODES(args, OP_init, logs_ODES)
                print("Done and saved ODEs")

            if args.training: 
                training.training(args, data, stud, logs_training, t1, t2)
                print('Done and saved training')
            