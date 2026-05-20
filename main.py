#TODO : WHy is gamma not used in training ?

import utils
import training
import ODEs_theory
import networks
import logger
from numpy import sqrt, arange
argfile = "/home/theom/continual_learning_trial/parameters.json"
args = utils.loadJson(argfile)
rhos = arange(0,1.01, 0.1)
for rho in rhos :
    args.rho = float(rho)
    metadata=utils.get_metadata(args)

    logs_ODES = logger.ExperimentLogger(metadata)
    logs_training = logger.ExperimentLogger(metadata)
    utils.set_seed(args.seed)
    data = networks.Data_and_Teachers(args)
    t1,t2 = data.get_Teachers()
    stud = networks.Student(args)
    OP_init = utils.get_OP_init(stud,data,args)

    OP_init_switch = training.training(args, data, t1, t2, stud, logs_training)
    print('Done training')

    ODEs_theory.solve_ODES(args, OP_init, OP_init_switch, logs_ODES)
    print("Done ODEs")
