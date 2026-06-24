#TODO : WHy is gamma not used in training ?

import utils
import training
import ODEs_theory
import networks
import logger
from numpy import sqrt, arange
argfile = "/home/theom/continual_learning/parameters.json"
args = utils.loadJson(argfile)
#rhos = arange(0,1.01, 0.025) #For theory
#rhos = arange(0,1.01, 0.1) #For theory
#rhos = arange(0,1.01, 0.2) #For training
rhos=[0.5]
print(rhos)
for rho in rhos :
    print("rho=", rho)
    args.rho = round(float(rho),3)
    metadata=utils.get_metadata(args)
    logs_ODES = logger.ExperimentLogger(metadata)
    logs_training = logger.ExperimentLogger(metadata)
    utils.set_seed(args.seed)
    data = networks.Data_and_Teachers(args)
    t1,t2 = data.get_Teachers()
    stud = networks.Student(args)
    OP_init = utils.get_OP_init(stud,data,args)
     
    if args.training : 
        training.training(args, data, t1, t2, stud, logs_training)
        print('Done and saved training')
    if args.theory :
        ODEs_theory.solve_ODES(args, OP_init, logs_ODES)
        print("Done and saved ODEs")
