
import torch
import torch.nn as nn
from numpy import sqrt
from logger import ExperimentLogger

def get_opt(args, stud):
    
  return torch.optim.SGD( [
      {"params": stud.fc1.parameters(), "lr": args.alpha_W},
      {"params": stud.A.parameters(), "lr": args.alpha_a},
      {"params": stud.B.parameters(), "lr": args.alpha_b/args.N},
      {"params": stud.head_1.parameters(), "lr": args.alpha_H/args.N},
      {"params": stud.head_2.parameters(), "lr": args.alpha_H/args.N},] )

def initialize_B(stud, args, loss, scoring_method="SDGM", data=None):

    if scoring_method == "SDGM":
      _, idx = torch.topk(torch.abs(stud.head_1.weight).flatten(), args.K - args.rows_to_freeze)

    elif scoring_method == "inv_SDGM":
      _, idx = torch.topk(torch.abs(stud.head_1.weight).flatten(), args.K - args.rows_to_freeze, largest=False)

    elif scoring_method == "re-use":
      x_list, y1_list = [], []
      importance_scores = torch.zeros(args.K, device=args.device)
      for _ in range(args.P_test):
        x_i, y1_i, _ = data.get_data(task=1)
        x_list.append(x_i)
        y1_list.append(y1_i)
        
      x = torch.stack(x_list).to(args.device)
      y1 = torch.stack(y1_list).to(args.device)

      with torch.no_grad():
        y_pred1, _ = stud(x)
        base_loss = 0.5 * loss(y_pred1, y1)
        original_head1 = stud.head_1.weight.clone()
        for k_node in range(args.K):
          #ablate the k-th node in head_1 and compute loss. We build it back at the end
          stud.head_1.weight[:, k_node] = 0.0
          y_pred1_ablated, _ = stud(x)
          ablated_loss = 0.5 * loss(y_pred1_ablated, y1)
          importance_scores[k_node] = ablated_loss - base_loss
          stud.head_1.weight[:, k_node] = original_head1[:, k_node]
          
      _, idx = torch.topk(importance_scores, args.K - args.rows_to_freeze)

    else:
        raise ValueError(f"Unknown scoring_method: {scoring_method}")

    if args.method in ["Sco-LoRA", "Sco-standard"]:
        idx_set = set(idx.cpu().numpy().tolist())
        with torch.no_grad():
            k = 0
            for i in range(args.K):
                stud.B.weight[i, :] = 0.0
                if i not in idx_set:
                    stud.B.weight[i, k % args.L] = 1.0
                    k += 1
    print(stud.B.weight)
def log_loss(logs_training, test_losses, task, step, test_1_switch=None, test_2_switch=None):
  if task ==1 : 
      logs_training.log_many(step=step,
      test_loss_1=test_losses[0], test_loss_2=test_losses[1],)
  else :
      logs_training.log_many(step=step,
      test_loss_1=test_losses[0], test_loss_2=test_losses[1], forgetting=test_losses[0] - test_1_switch, transfer=test_2_switch - test_losses[1],)
    
      
def log_order_parameters(args, logs_training, stud, t1_first_layer, t2_first_layer, task, step,):
  if task ==1 :
    W_curr = stud.fc1.weight.detach()
    h1_curr = stud.head_1.weight.detach().numpy().flatten().copy()
    
    #Order parameters
    Q=((W_curr @ W_curr.T)/args.N) 
    R=((W_curr @ t1_first_layer.T )/args.N) 
    U=((W_curr @ t2_first_layer.T )/args.N) 
    h1=h1_curr.copy()
    
    logs_training.log_many(step=step,
    Q=Q.detach().cpu().clone(), R=R.detach().cpu().clone(), U=U.detach().cpu().clone(), h1=h1.copy())
  
  elif task==2 :
    W_curr = stud.fc1.weight.detach()
    h1_curr = stud.head_1.weight.detach().numpy().flatten().copy()
    h2_curr = stud.head_2.weight.detach().numpy().flatten().copy()
    A_curr = stud.A.weight.detach()
    B_curr = stud.B.weight.detach()

    if args.method in {"LoRA", "Sco-LoRA", "only_LoRA", "Sco-standard"}  :
          
      Xi=((W_curr @ A_curr.T)/args.N)
      Phi=((A_curr @ A_curr.T)/args.N)
      Gamma=(( t2_first_layer @ A_curr.T)/args.N)
      Lambda=(( t1_first_layer @ A_curr.T)/args.N)
      
      logs_training.log_many(step=step,
      B=B_curr.detach().cpu().clone(), Xi=Xi.detach().cpu().clone(), Phi=Phi.detach().cpu().clone(),
      Gamma=Gamma.detach().cpu().clone(), Lambda=Lambda.detach().cpu().clone(), h2=h2_curr.copy())
          
      if args.method=="only_LoRA":
            Q=((W_curr @ W_curr.T)/args.N) 
            R=((W_curr @ t1_first_layer.T )/args.N) 
            U=((W_curr @ t2_first_layer.T )/args.N)
          
            logs_training.log_many(
              step=step,
              Q=Q.detach().cpu().clone(), R=R.detach().cpu().clone(), U=U.detach().cpu().clone(), h2=h2_curr.copy())
          
    else :
      Q=((W_curr @ W_curr.T)/args.N) 
      R=((W_curr @ t1_first_layer.T )/args.N) 
      U=((W_curr @ t2_first_layer.T )/args.N)

      logs_training.log_many(step=step,
        Q=Q.detach().cpu().clone(), R=R.detach().cpu().clone(), U=U.detach().cpu().clone(), h2=h2_curr.copy())
    
  
def training(args, data, stud, logs_training, t1, t2):
    if args.dataset == "synthetic":
      P = args.alpha * args.N
    elif args.dataset == "MNIST":
      P=30000 #max P for splitted MNIST.
      args.P_test = 5000
    elif args.dataset == "CIFAR":
      P=10000
      args.P_test = 2000
    elif args.dataset == "fMNIST":
      P=12000
      args.P_test = 2000
    
    gamma = args.gamma/sqrt(args.L)
    if args.dataset == "synthetic":
      t1.to(args.device)
      t2.to(args.device)
      print('Empirical rho:', (t1.fc1.weight @ t2.fc1.weight.T)/args.N)
    
      t1_first_layer = t1.fc1.weight.detach()
      t2_first_layer = t2.fc1.weight.detach()
      
    stud.to(args.device)
    

    opt = get_opt(args, stud)
    loss = nn.MSELoss()
    test_losses = data.test_error(stud, loss, args.P_test, args.method)
    log_loss(logs_training, test_losses, task=1, step=0)
    
    if args.dataset == "synthetic":
      log_order_parameters(args, logs_training, stud, t1_first_layer, t2_first_layer, task=1, step=0,)

    print('Starting Task 1 ----------------------')
    print('Test Loss on Task 1:', test_losses[0])
    print('Test Loss on Task 2:', test_losses[1], '\n')

    # FIRST PHASE OF TRAINING:
    
    for _ in range(1,P+1):
      x, y1, y2 = data.get_data(task=1)
      opt.zero_grad()
      y_pred1, y_pred2 = stud(x.to(args.device))
      l1 = 0.5*loss(y_pred1, y1)
      l2 = 0.5*loss(y_pred2, y2)
      l1.backward()
      opt.step()

      if _ % 500 == 0: #Saving procedure
        test_losses = data.test_error(stud, loss, args.P_test, args.method)
        log_loss(logs_training, test_losses, task=1, step=_)
        if args.dataset == "synthetic":
          log_order_parameters(args,logs_training,stud,t1_first_layer,t2_first_layer, task=1, step=_)

    test = data.test_error(stud, loss, args.P_test, args.method)
    print('Starting Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    #Selection rule    
    if args.method in["Sco-LoRA", "Sco-standard"]:
      initialize_B(stud, args, loss, scoring_method=args.scoring_method, data=data)
            
    stud.update_grad_state(method=args.method)

    if args.method == "only_LoRA":
      stud.fc1.weight.data.zero_()    

    test_1_switch = test[0] #Used to compute forgetting
    test_2_switch = test[1] #Used to compute transfer
    
    # SECOND PHASE OF TRAINING:
    
    for _ in range(1,P+1):
      x, y1, y2 = data.get_data(task=2)
      opt.zero_grad()
      y_pred1, y_pred2 = stud(x.to(args.device), method=args.method, task_id=2)
      l1 = 0.5*loss(y_pred1, y1)
      l2 = 0.5*loss(y_pred2, y2)
      l2.backward()
      opt.step()
      
      if _ % 500 == 0:
        test_losses = data.test_error(stud, loss, args.P_test, method=args.method)
        log_loss(logs_training, test_losses, task=2, step=P+_, test_1_switch = test_1_switch, test_2_switch=test_2_switch)
        if args.dataset == "synthetic":
          log_order_parameters(args, logs_training, stud, t1_first_layer, t2_first_layer, task=2, step = P + _)        
        
    test = data.test_error(stud, loss, args.P_test, args.method)

    print('End of Task 2 ----------------------')
    print('Test Loss on Task 1:', test[0])
    print('Test Loss on Task 2:', test[1], '\n')

    if args.dataset =="synthetic":
      ExperimentLogger.append_to_file(f"results/run_K={args.K}_M={args.M}.npy", logs_training)
    elif args.dataset == "MNIST":
      ExperimentLogger.append_to_file(f"results/run_MNIST.npy",logs_training)
    elif args.dataset == "CIFAR":
      ExperimentLogger.append_to_file(f"results/run_CIFAR.npy",logs_training)
    elif args.dataset == "fMNIST":
      ExperimentLogger.append_to_file(f"results/run_fMNIST.npy",logs_training)