
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
  
    #Saving metrics before training
    test = data.test_error(stud, loss, args.P_test, LoRA=False)
    W_init = stud.fc1.weight.detach()
    Ha_init = stud.head_1.weight.detach().numpy().flatten().copy()

    
    #Order parameters for initialization
    Q=((W_init @ W_init.T)/args.N) 
    R=((W_init @ t1_first_layer.T )/args.N) 
    U=((W_init @ t2_first_layer.T )/args.N) 
    Ha=Ha_init
    
    logs_training.log_many(
    step=0,
    test_loss_1=test[0],
    test_loss_2=test[1],

    Q=Q.detach().cpu().clone(),
    R=R.detach().cpu().clone(),
    U=U.detach().cpu().clone(),
    Ha=Ha.copy(),)

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
        Ha=Ha_curr.copy())


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

        #Order parameters 
        W_curr = stud.fc1.weight.detach()
        Ha_curr = stud.head_1.weight.detach().numpy().flatten().copy()
        Hb_curr = stud.head_2.weight.detach().numpy().flatten().copy()
        A_curr = stud.A.weight.detach()
        B_curr = stud.B.weight.detach()

        
        logs_training.log_many(
          step=P + _,
          test_loss_1=test[0],
          test_loss_2=test[1],
          forgetting=test[0] - test_1_switch,
          transfer=test_2_switch - test[1],)

        if args.LoRA :
          
          G=((W_curr @ A_curr.T)/args.N)
          Phi=((A_curr @ A_curr.T)/args.N)
          Gamma=(( t2_first_layer @ A_curr.T)/args.N)
          Lambda=(( t1_first_layer @ A_curr.T)/args.N)
          
          logs_training.log_many(
          step=P + _,
          D=B_curr.detach().cpu().clone(),
          G=G.detach().cpu().clone(),
          Phi=Phi.detach().cpu().clone(),
          Gamma=Gamma.detach().cpu().clone(),
          Lambda=Lambda.detach().cpu().clone(),
          Hb=Hb_curr.copy())
          
        else :
          
          Q=((W_curr @ W_curr.T)/args.N) 
          R=((W_curr @ t1_first_layer.T )/args.N) 
          U=((W_curr @ t2_first_layer.T )/args.N)
          
          logs_training.log_many(
            step=P + _,
            Q=Q.detach().cpu().clone(),
            R=R.detach().cpu().clone(),
            U=U.detach().cpu().clone(),
            Hb=Hb_curr.copy())
        
        
    test = data.test_error(stud, loss, args.P_test, LoRA=args.LoRA)

    print('End of Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    ExperimentLogger.append_to_file(
    f"results/run_K={args.K}_M={args.M}.npy",
    logs_training
    )
    return OP_init_switch