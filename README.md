STRUCTURE OF THE CODE:\
file .json is to choose the parameters of the model\
networks is composed of the classes of teacher and students\
training shows the training loop. It returns a dictionnary containing all the possible observables of the systems\
ODEs_theory contains all the functions to compute ODEs, for LGS setting and LoRA. It also compute the solver of ODEs.\
utils contains helper functions such as import of the json, or saving of the order parameters during training/ODE solving.\


TODO

1) Fix the initialization of the OPs in the ODES. Right now, we must still run training.
2) Once last step is done, add two helpers in json to choose if we want to compute theory/sim


