#TODO : WHy is gamma not used in training ?

import utils
import training
import ODEs_theory
import networks
from numpy import sqrt
argfile = "/home/theom/continual_learning/parameters.json"
args = utils.loadJson(argfile)
P = args.alpha * args.N # Total number of examples seen during training at each phase
gamma = args.beta/sqrt(args.L) #Effective learning rate of LoRA updates
utils.set_seed(args.seed)
data = networks.Data_and_Teachers(args)
stud = networks.Student(args)

logs, OP_init = training.training(args, data, stud)
print('Done training')
utils.save_to_file(logs, args, ODES=False)
logs_ODES = ODEs_theory.solve_ODES(args, OP_init, gamma)
print("Done ODEs")
utils.save_to_file(logs_ODES, args, ODES=True)