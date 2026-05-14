
import torch
import torch.nn as nn
from numpy import sqrt
import utils 

#TODO : Put the wt procedure
def training(args, data, stud):
    dist = torch.distributions.MultivariateNormal(torch.zeros(args.N), torch.eye(args.N))
    P = args.alpha * args.N
    gamma = args.beta/sqrt(args.L)
    t1, t2 = data.get_Teachers()
    t1.to(args.device)
    t2.to(args.device)
    print('Empirical rho:', (t1.fc1.weight @ t2.fc1.weight.T)/args.N)
    # Order parameters after initialization of the network
    W0 = stud.fc1.weight.detach().numpy().copy()
    A0 = stud.A.weight.detach().numpy().copy()
    D0 = stud.B.weight.detach().numpy().copy()
    B_a = data.teacher_1.fc1.weight.detach().numpy().copy()
    B_b = data.teacher_2.fc1.weight.detach().numpy().copy()


    OP_init = {
    "Q0": ((W0 @ W0.T) / args.N).copy(),
    "R0": ((W0 @ B_a.T) / args.N).copy(),
    "T0": ((B_a @ B_a.T) / args.N).copy(),
    "U0": ((W0 @ B_b.T) / args.N).copy(),
    "S0": ((B_b @ B_b.T) / args.N).copy(),
    "V0": ((B_a @ B_b.T) / args.N).copy(),
    "G0": ((W0 @ A0.T) / args.N).copy(),
    "Lam0": ((B_a @ A0.T) / args.N).copy(),
    "Gam0": ((B_b @ A0.T) / args.N).copy(),
    "Phi0": ((A0 @ A0.T) / args.N).copy(),
    "Ha0": stud.head_1.weight.detach().numpy().reshape((args.K,)).copy(),
    "Hb0": stud.head_2.weight.detach().numpy().reshape((args.K,)).copy(),
    "v_a": data.teacher_1.fc2.weight.detach().numpy().copy(),
    "v_b": data.teacher_2.fc2.weight.detach().numpy().copy(),
    "D0" : stud.B.weight.detach().numpy().copy()
}

    stud.to(args.device)
    

    opt = torch.optim.SGD( [

        {"params": stud.fc1.parameters(), "lr": args.alpha_W},

        {"params": stud.A.parameters(), "lr": args.alpha_a},

        {"params": stud.B.parameters(), "lr": args.alpha_b/args.N},

        {"params": stud.head_1.parameters(), "lr": args.alpha_H/args.N},

        {"params": stud.head_2.parameters(), "lr": args.alpha_H/args.N},] )

    loss = nn.MSELoss()
    t1_first_layer = t1.fc1.weight.data
    t2_first_layer = t2.fc1.weight.data
    Tcos_t1 = torch.sqrt(torch.diag(t1_first_layer @ t1.fc1.weight.data.T / args.N )) #Norm of first teacher matrix, used to compute overlaps
    Tcos_t2 = torch.sqrt(torch.diag(t2_first_layer @ t2.fc1.weight.data.T / args.N )) #Norm of second teacher matrix, used to compute overlaps

    logs = utils.initialize_dictionary(args)
    
        #Saving metrics before training
    test = data.test_error(stud, loss, args.P_test, LoRA=False)

    Rcos_t1 = (stud.fc1.weight @ t1.fc1.weight.data.T) / args.N
    Rcos_t2 = (stud.fc1.weight @ t2.fc1.weight.data.T) / args.N
    Qcos = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
    overlap_t1 = (Rcos_t1.T / Qcos).T /Tcos_t1
    overlap_t2 = (Rcos_t2.T / Qcos).T /Tcos_t2
    
    logs = utils.save_overlaps_in_dictionnary(logs, args, overlap_t1, overlap_t2, task=1)
    logs["steps"].append(0)
    logs["test_loss1"].append(test[0])
    logs["test_loss2"].append(test[1])
    logs["forgetting"].append(0)



    print('Starting Task 1 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    # FIRST PHASE OF TRAINING:
    
    for _ in range(1,P+1):
      x, y1, y2 = data.get_data()
      opt.zero_grad()

      y_pred1, y_pred2 = stud(x.to(args.device))
      l1 = 0.5*loss(y_pred1, y1)
      l2 = 0.5*loss(y_pred2, y2)

      l1.backward()
      opt.step()

      if _ % 500 == 0: #Saving procedure
        
        test = data.test_error(stud, loss, args.P_test, LoRA=False)
        Rcos_t1 = (stud.fc1.weight @ t1_first_layer.T) / args.N
        Rcos_t2 = (stud.fc1.weight @ t2_first_layer.T) / args.N
        Qcos = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
        overlap_t1 = (Rcos_t1.T / Qcos).T /Tcos_t1
        overlap_t2 = (Rcos_t2.T / Qcos).T /Tcos_t2
        
        logs = utils.save_overlaps_in_dictionnary(logs, args, overlap_t1, overlap_t2, task=1)
        logs["steps"].append(_)
        logs["test_loss1"].append(test[0])
        logs["test_loss2"].append(test[1])
        logs["forgetting"].append(0)


    test = data.test_error(stud, loss, args.P_test, LoRA=False)
    print('Starting Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    if args.A_only:
      val,idx = torch.topk( torch.abs(stud.head_1.weight), args.K - args.L )
      with torch.no_grad():
        k = 0
        for i in range(args.K):
          if i in idx:
            stud.B.weight[i,:] = 0
          else:
            stud.B.weight[i,:] = 0
            stud.B.weight[i,k % args.L] = 1
            k += 1
    
    w1 = stud.fc1.weight.data

    W0_switch = stud.fc1.weight.detach().numpy().copy()
    B_a_switch = data.teacher_1.fc1.weight.detach().numpy().copy()
    B_b_switch = data.teacher_2.fc1.weight.detach().numpy().copy()

    OP_init.update({
    "Q0_switch": ((W0_switch @ W0_switch.T) / args.N).copy(),
    "R0_switch": ((W0_switch @ B_a_switch.T) / args.N).copy(),
    "U0_switch": ((W0_switch @ B_b_switch.T) / args.N).copy(),
    "G0_switch": ((W0_switch @ A0.T) / args.N).copy(),
    "Ha0_switch": stud.head_1.weight.detach().numpy().reshape((args.K,)).copy(),
    "D0_switch" : stud.B.weight.detach().numpy().copy()
    })
    test_1_switch = test[0] #Used to compute forgetting
    # SECOND PHASE OF TRAINING:
    for _ in range(1,P+1):
      x, y1, y2 = data.get_data()
      opt.zero_grad()


      y_pred1, y_pred2 = stud(x.to(args.device), LoRA = args.LoRA, A_only=args.A_only)
      l1 = 0.5*loss(y_pred1, y1)
      l2 = 0.5*loss(y_pred2, y2)

      l2.backward()
      
      opt.step()
      
      if _ % 500 == 0:
        
        test = data.test_error(stud, loss, args.P_test, LoRA=args.LoRA)
        #Computation of the overlaps for Ws
        Rcos_t1 = (stud.fc1.weight @ t1_first_layer.T) / args.N
        Rcos_t2 = (stud.fc1.weight @ t2_first_layer.T) / args.N
        Norm_student = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
        overlap_t1 = (Rcos_t1.T / Norm_student).T /Tcos_t1
        overlap_t2 = (Rcos_t2.T / Norm_student).T /Tcos_t2
        
        #Computation of the overlaps for LoRA
        delta_W = (stud.B.weight @ stud.A.weight) / sqrt(args.L) #Normalization due to the forward pass
        Rcos_t1_LoRA = (delta_W @ t1_first_layer.T) / args.N
        Rcos_t2_LoRA = (delta_W @ t2_first_layer.T) / args.N
        Norm_LoRA = torch.sqrt(torch.diag(delta_W @ delta_W.T / args.N))
        overlap_t1_LoRA = (Rcos_t1_LoRA.T / Norm_LoRA).T /Tcos_t1
        overlap_t2_LoRA = (Rcos_t2_LoRA.T / Norm_LoRA).T /Tcos_t2
        
        Rcos_t1_wt = (stud.wt.weight @ t1_first_layer.T) / args.N
        Rcos_t2_wt = (stud.wt.weight @ t2_first_layer.T) / args.N
        Norm_wt = torch.sqrt(torch.diag(stud.wt.weight @ stud.wt.weight.T / args.N))
        overlap_t1_wt = (Rcos_t1_wt.T / Norm_wt).T /Tcos_t1
        overlap_t2_wt = (Rcos_t2_wt.T / Norm_wt).T /Tcos_t2
        
        if args.wt == False and args.LoRA == True :
          full_student = stud.fc1.weight + delta_W
          
        elif args.wt == True and args.LoRA == False :
          full_student = stud.fc1.weight + stud.wt.weight
          
        else :
          full_student = stud.fc1.weight
        
        Rcos_t1_full = (full_student @ t1_first_layer.T) / args.N
        Rcos_t2_full = (full_student @ t2_first_layer.T) / args.N
        Norm_full = torch.sqrt(torch.diag(full_student @ full_student.T / args.N))
        overlap_t1_full = (Rcos_t1_full.T / Norm_full).T /Tcos_t1
        overlap_t2_full = (Rcos_t2_full.T / Norm_full).T /Tcos_t2
                
        logs["steps"].append(P +_)
        logs["test_loss1"].append(test[0])
        logs["test_loss2"].append(test[1])
        logs["forgetting"].append(test[0] - test_1_switch)
        logs = utils.save_overlaps_in_dictionnary(logs, args, overlap_t1, overlap_t2, task=2, 
                                 overlap_t1_LoRA=overlap_t1_LoRA, overlap_t2_LoRA=overlap_t2_LoRA,
                                 overlap_t1_wt=overlap_t1_wt, overlap_t2_wt=overlap_t2_wt, 
                                 overlap_t1_full=overlap_t1_full, overlap_t2_full=overlap_t2_full,
                                 Norm_LoRA=Norm_LoRA)
        
    test = data.test_error(stud, loss, args.P_test, LoRA=args.LoRA)

    print('End of Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')
    
    logs['N'] = [args.N] * len(logs["steps"])
    logs['M'] = [args.M] * len(logs["steps"])
    logs['K'] = [args.K] * len(logs["steps"])
    logs['rho'] = [args.rho] * len(logs["steps"])
    logs['alpha'] = [args.alpha] * len(logs["steps"])
    logs['beta'] = [args.beta] * len(logs["steps"])
    logs['L'] = [args.L] * len(logs["steps"])
    logs['A_only'] = [args.A_only] * len(logs["steps"])
    if args.LoRA:
      logs['method'] = ['LoRA'] * len(logs["steps"])
    elif args.wt:
      logs['method'] = ['LoRA_full_rank'] * len(logs["steps"])
    else:
      logs['method'] = ['standard'] * len(logs["steps"])

    return logs, OP_init