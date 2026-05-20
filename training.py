
import torch
import torch.nn as nn
from numpy import sqrt
from logger import ExperimentLogger

def training(args, data, t1, t2, stud, logs_training):
    P = args.alpha * args.N
    gamma = args.beta/sqrt(args.L)
    t1.to(args.device)
    t2.to(args.device)
    stud.to(args.device)
    print('Empirical rho:', (t1.fc1.weight @ t2.fc1.weight.T)/args.N)
    
    #Teacher first layers
    t1_first_layer = t1.fc1.weight.detach()
    t2_first_layer = t2.fc1.weight.detach()
    

    opt = torch.optim.SGD( [
        {"params": stud.fc1.parameters(), "lr": args.alpha_W},
        {"params": stud.A.parameters(), "lr": args.alpha_a},
        {"params": stud.B.parameters(), "lr": args.alpha_b/args.N},
        {"params": stud.head_1.parameters(), "lr": args.alpha_H/args.N},
        {"params": stud.head_2.parameters(), "lr": args.alpha_H/args.N},] )

    loss = nn.MSELoss()
    
    Tcos_t1 = torch.sqrt(torch.diag(t1_first_layer @ t1_first_layer.T / args.N )) #Norm of first teacher matrix, used to compute overlaps
    Tcos_t2 = torch.sqrt(torch.diag(t2_first_layer @ t2_first_layer.T / args.N )) #Norm of second teacher matrix, used to compute overlaps

    #Saving metrics before training
    test = data.test_error(stud, loss, args.P_test, LoRA=False)
    W_curr = stud.fc1.weight.detach()
    Ha_curr = stud.head_1.weight.detach().numpy().flatten().copy()

    #overlaps for initialization
    Rcos_t1 = (stud.fc1.weight @ t1_first_layer.T) / args.N
    Rcos_t2 = (stud.fc1.weight @ t2_first_layer.T) / args.N
    Qcos = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
    overlap_t1 = (Rcos_t1.T / Qcos).T /Tcos_t1
    overlap_t2 = (Rcos_t2.T / Qcos).T /Tcos_t2
    
    #Order parameters for initialization
    Q=((W_curr @ W_curr.T)/args.N) 
    R=((W_curr @ t1_first_layer.T )/args.N) 
    U=((W_curr @ t2_first_layer.T )/args.N) 
    Ha=Ha_curr
    logs_training.log_many(
    step=0,
    test_loss_1=test[0],
    test_loss_2=test[1],

    Q=Q.detach().cpu().clone(),
    R=R.detach().cpu().clone(),
    U=U.detach().cpu().clone(),
    Ha=Ha.copy(),

    overlap_t1=overlap_t1.detach().cpu().clone(),
    overlap_t2=overlap_t2.detach().cpu().clone())

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
        W_curr = stud.fc1.weight.detach()
        Ha_curr = stud.head_1.weight.detach().numpy().flatten().copy()
        
        #Overlaps
        Rcos_t1 = (stud.fc1.weight @ t1_first_layer.T) / args.N
        Rcos_t2 = (stud.fc1.weight @ t2_first_layer.T) / args.N
        Qcos = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
        overlap_t1 = (Rcos_t1.T / Qcos).T /Tcos_t1
        overlap_t2 = (Rcos_t2.T / Qcos).T /Tcos_t2
        
        #Order parameters
        Q=((W_curr @ W_curr.T)/args.N) 
        R=((W_curr @ t1_first_layer.T )/args.N) 
        U=((W_curr @ t2_first_layer.T )/args.N) 
        Ha=Ha_curr.copy()
        
        logs_training.log_many(
        step=_,
        test_loss_1=test[0],
        test_loss_2=test[1],

        Q=Q.detach().cpu().clone(),
        R=R.detach().cpu().clone(),
        U=U.detach().cpu().clone(),
        Ha=Ha_curr.copy(),

        overlap_t1=overlap_t1.detach().cpu().clone(),
        overlap_t2=overlap_t2.detach().cpu().clone())


    test = data.test_error(stud, loss, args.P_test, LoRA=False)
    print('Starting Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    #Selection rule
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
    
    W0_switch = stud.fc1.weight.detach()
    A0 = stud.A.weight.detach()
    

    OP_init_switch = {
    "Q0_switch": ((W0_switch @ W0_switch.T) / args.N).detach().cpu().numpy(),
    "R0_switch": ((W0_switch @ t1_first_layer.T) / args.N).detach().cpu().numpy(),
    "U0_switch": ((W0_switch @ t2_first_layer.T) / args.N).detach().cpu().numpy(),
    "G0_switch": ((W0_switch @ A0.T) / args.N).detach().cpu().numpy(),
    "Ha0_switch": stud.head_1.weight.detach().cpu().numpy().reshape((args.K,)),
    "D0_switch": stud.B.weight.detach().cpu().numpy()
    }
    test_1_switch = test[0] #Used to compute forgetting
    test_2_switch = test[1] #Used to compute transfer
    
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
        
        #Overlaps for Ws
        Rcos_t1 = (stud.fc1.weight @ t1_first_layer.T) / args.N
        Rcos_t2 = (stud.fc1.weight @ t2_first_layer.T) / args.N
        Norm_student = torch.sqrt(torch.diag(stud.fc1.weight @ stud.fc1.weight.T / args.N))
        overlap_t1 = (Rcos_t1.T / Norm_student).T /Tcos_t1
        overlap_t2 = (Rcos_t2.T / Norm_student).T /Tcos_t2
        
        #Overlaps for LoRA
        delta_W = (stud.B.weight @ stud.A.weight) / sqrt(args.L) #Normalization due to the forward pass
        Rcos_t1_LoRA = (delta_W @ t1_first_layer.T) / args.N
        Rcos_t2_LoRA = (delta_W @ t2_first_layer.T) / args.N
        Norm_LoRA = torch.sqrt(torch.diag(delta_W @ delta_W.T / args.N))
        overlap_t1_LoRA = (Rcos_t1_LoRA.T / Norm_LoRA).T /Tcos_t1
        overlap_t2_LoRA = (Rcos_t2_LoRA.T / Norm_LoRA).T /Tcos_t2
        
        #Overlaps for wt
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
                
        #Order parameters 
        W0 = stud.fc1.weight.detach()
        A_curr = stud.A.weight.detach()
        B_curr = stud.B.weight.detach()
        Hb_curr = stud.head_2.weight.detach().flatten()

        D=B_curr
        Hb=Hb_curr
        G=((W0 @ A_curr.T)/args.N)
        Phi=((A_curr @ A_curr.T)/args.N)
        Gamma=(( t2_first_layer @ A_curr.T)/args.N)
        Lambda=(( t1_first_layer @ A_curr.T)/args.N)
        
        logs_training.log_many(
    step=P + _,

    test_loss_1=test[0],
    test_loss_2=test[1],
    forgetting=test[0] - test_1_switch,
    transfer=test_2_switch - test[1],
    
    # Order parameters
    D=D.detach().cpu().clone(),
    G=G.detach().cpu().clone(),
    Phi=Phi.detach().cpu().clone(),
    Gamma=Gamma.detach().cpu().clone(),
    Lambda=Lambda.detach().cpu().clone(),
    Hb=Hb.detach().cpu().clone(),

    # Overlaps
    overlap_t1=overlap_t1.detach().cpu().clone(),
    overlap_t2=overlap_t2.detach().cpu().clone(),

    overlap_t1_LoRA=overlap_t1_LoRA.detach().cpu().clone(),
    overlap_t2_LoRA=overlap_t2_LoRA.detach().cpu().clone(),

    overlap_t1_wt=overlap_t1_wt.detach().cpu().clone(),
    overlap_t2_wt=overlap_t2_wt.detach().cpu().clone(),

    overlap_t1_full=overlap_t1_full.detach().cpu().clone(),
    overlap_t2_full=overlap_t2_full.detach().cpu().clone(),

    Norm_LoRA=Norm_LoRA.detach().cpu().clone(),
)
    test = data.test_error(stud, loss, args.P_test, LoRA=args.LoRA)

    print('End of Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')
    
    ExperimentLogger.append_to_file(
    f"results/run_K={args.K}_M={args.M}.npy",
    logs_training
    )
    return OP_init_switch