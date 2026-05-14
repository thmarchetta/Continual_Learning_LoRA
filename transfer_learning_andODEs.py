import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from numpy import pi, sqrt, asin
import os
import pandas as pd
import ast
import re
from Odes_functions_LEE import *

torch.set_printoptions(linewidth=100000)
np.set_printoptions(linewidth=100000)

class Erf(nn.Module):
  def __init__(self):
    super(Erf,self).__init__()
  def forward(self,x):
    return torch.erf(x/sqrt(2))
  
class Committee(nn.Module):
    def __init__(self, input_size, hidden_size,J,h):
        super().__init__()

        self.N = input_size
        self.K = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(J)

        self.fc2 = nn.Linear(hidden_size,1,bias=False)
        with torch.no_grad():
            self.fc2.weight.copy_(h)

        self.activation = Erf()

    def forward(self, x):
        x = self.fc1(x)/sqrt(self.N)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class Committee_Multi(nn.Module):
    def __init__(self, input_size, hidden_size,J,hA,hB):
        super().__init__()

        self.N = input_size
        self.K = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(J)

        self.fc2A = nn.Linear(hidden_size,1,bias=False)
        with torch.no_grad():
            self.fc2A.weight.copy_(hA)

        self.fc2B = nn.Linear(hidden_size,1,bias=False)
        with torch.no_grad():
            self.fc2B.weight.copy_(hB)

        self.activation = Erf()

    def forward(self, x):
        
        x = self.fc1(x)/sqrt(self.N)
        x = self.activation(x)

        xA = self.fc2A(x)
        xB = self.fc2B(x)
        return { 'A':xA , 'B':xB}


K = 5
M = 1
N = 1000
torch.manual_seed(((K+M)*N)//M)
alpha_i = 0 # 0/N
alpha_f = 400 # P/N
alpha_s = alpha_f//2 #switch/N

switch = alpha_s * N
P = alpha_f * N

P_test = 5*N

num_metrics = 500 # collect one metric every step_metric
step_metric = int( P//num_metrics )

# LEARNING RATES FOR TRAINING PROCEDURES:
alpha_W = 0.9
alpha_H = 0.9
dist = torch.distributions.MultivariateNormal(torch.zeros(N), torch.eye(N))
MSE = nn.MSELoss()

# GENERATE TWO TEACHERS WITH SIMILARITY gamma
# SIMILARITY BETWEEN TASKS
gamma = float(torch.tensor([0.1]))

with open('network_parameters.txt','w') as f:
  f.write('V N K M alphaW alpha_H alpha_i alpha_s alpha_f P switch P_test step_metric\n')
with open('network_parameters.txt','a') as f:
  f.write(f'{gamma} {N} {K} {M} {alpha_W} {alpha_H} {alpha_i} {alpha_s} {alpha_f} {P} {switch} {P_test} {step_metric}')

#if M == 1:
#  q,r = torch.linalg.qr( torch.randn(N,N) )
#  B_A = sqrt(N) * q[0,:]
#  B_B = gamma * B_A + sqrt(1-gamma**2) * sqrt(N) * q[1,:]
#else:
B_A = dist.sample((M,))
B_A_orth = dist.sample((M,))
B_B = gamma*B_A + sqrt(1 - gamma**2)*B_A_orth
print('TeacherA - Teacher B overlap: ',(B_A @ B_B.T)/N)
print('Teacher A norm: ',(B_A @ B_A.T)/N)
print('Teacher B norm: ',(B_B @ B_B.T)/N)
input('Waiting...')
v_a = 1 + 0.0*torch.randn((M,))
v_b = -1 + 0.0*torch.randn((M,))

with open('teacher_parameters.txt','w') as f:
  f.write('V v_a v_b\n')
with open('teacher_parameters.txt','a') as f:
  f.write(f'{gamma} {v_a.detach()} {v_b.detach()}')

# DEFINE TEACHERS:
teacherA = Committee(N,M,B_A,v_a)
teacherB = Committee(N,M,B_B,v_b)
teachers = {'A':teacherA,'B':teacherB}

# GENERATE STUDENT A-B INITIAL WEIGHTS:
J = 0.001*torch.randn((K,N))
#h_a = torch.ones((K,)) # SAAD and SOLLA
#h_b = torch.ones((K,)) # SAAD and SOLLA
h_a = 0.001*torch.randn((1,K))
h_b = 0.001*torch.randn((1,K))

Q0 = (J @ J.T).detach()/N
R0 = (J @ B_A.T).detach()/N
S0 = (B_B @ B_B.T).detach()/N
T0 = (B_A @ B_A.T).detach()/N
U0 = (J @ B_B.T).detach()/N
V0 = (B_A @ B_B.T).detach()/N

with open(f"Q.txt", "w") as f:
  f.write("n Q\n")

with open(f"R.txt", "w") as f:
  f.write("n R\n")

with open(f"S.txt", "w") as f:
  f.write("n S\n")

with open(f"T.txt", "w") as f:
  f.write("n T\n")

with open(f"U.txt", "w") as f:
  f.write("n U\n")

with open(f"V.txt", "w") as f:
  f.write("n V\n")

with open(f"h_a.txt", "w") as f:
  f.write("n h_a\n")

with open(f"h_b.txt", "w") as f:
  f.write("n h_b\n")


# DEFINING STUDENT NETWORK FOR MULTITASK:
student = Committee_Multi(N,K,J,h_a,h_b)

with open(f"Aloss_function.txt", "w") as f:
  f.write("n LossA\n")

with open(f"Bloss_function.txt", "w") as f:
  f.write("n LossB\n")

with open(f"Q.txt", "a") as f:
  f.write(f"{0} {Q0.flatten()}\n")

with open(f"R.txt", "a") as f:
  f.write(f"{0} {R0.flatten()}\n")

with open(f"S.txt", "a") as f:
  f.write(f"{0} {S0.flatten()}\n")

with open(f"T.txt", "a") as f:
  f.write(f"{0} {T0.flatten()}\n")

with open(f"U.txt", "a") as f:
  f.write(f"{0} {U0.flatten()}\n")

with open(f"V.txt", "a") as f:
  f.write(f"{0} {V0.flatten()}\n")

with open(f"h_a.txt", "a") as f:
  f.write(f"{0} {h_a.flatten()}\n")

with open(f"h_b.txt", "a") as f:
  f.write(f"{0} {h_b.flatten()}\n")

with open(f"forgetting.txt","w") as f:
  f.write(f'N forgetting\n')

with open(f"transfer.txt","w") as f:
  f.write(f'N transfer\n')

# COMPUTE INITIAL LOSS:
x_test = dist.sample((P_test,))

y_student = student(x_test)
y_teacherA = teachers['A'](x_test)
y_teacherB = teachers['B'](x_test)

lossA = 0.5*MSE(y_student['A'],y_teacherA)
lossB = 0.5*MSE(y_student['B'],y_teacherB)

print(f'\nLearning at step {0} over {P}')
print(f'Student A output: {y_student['A'][0:3].tolist()}\nTeacher A output: {y_teacherA[0:3].tolist()}\nLoss A: {lossA.item()}\n')
print(f'Student B output: {y_student['B'][0:3].tolist()}\nTeacher B output: {y_teacherB[0:3].tolist()}\nLoss B: {lossB.item()}')

# CHOOSE FIRST TASK
task = 'A'

opt1 = torch.optim.SGD( [

    {"params": student.fc1.parameters(), "lr": alpha_W},

    {"params": student.fc2A.parameters(), "lr": alpha_H/N}] )

opt2 = torch.optim.SGD( [

    {"params": student.fc1.parameters(), "lr": alpha_W},

    {"params": student.fc2B.parameters(), "lr": alpha_H/N}] )


optimizers = {'A':opt1, 'B':opt2}

opt = optimizers[task]

sigma = 0.

print('STARTING TRAINING PROCEDURE')
for i in range(1,P+1):

    if i == switch:
        if task == 'A':
            task = 'B'
            opt = optimizers[task]
            endlossA = lossA.detach()
            endlossB = lossB.detach()
        else:
            task = 'A'
            opt = optimizers[task]

    x = dist.sample((1,))
    y_student = student(x)[task]
    y_teacher = teachers[task](x)

    opt.zero_grad()
    loss = 0.5*MSE(y_teacher, y_student)
    loss.backward()
    opt.step()

    if i % step_metric == 0:
        
        print(f'\nTraining step {i} over {P}. Training on task '+task)
        print(f'Student A output: {student(x)['A'].item()}\nTeacher A output: {teacherA(x).item()}')
        print(f'Student B output: {student(x)['B'].item()}\nTeacher B output: {teacherB(x).item()}')
        x_test = dist.sample((P_test,))# Sampling of P N-dimensional inputs (independent?)
        y_student = student(x_test)
        y_teacherA = teachers['A'](x_test)
        y_teacherB = teachers['B'](x_test)

        lossA = 0.5*MSE(y_student['A'],y_teacherA)
        lossB = 0.5*MSE(y_student['B'],y_teacherB)

        if i == switch:
          endlossA = lossA.detach()
          endlossB = lossB.detach()

        with open(f"Aloss_function.txt", "a") as f:
          f.write(f"{i} {lossA.item()}\n")

        with open(f"Bloss_function.txt", "a") as f:
          f.write(f"{i} {lossB.item()}\n")
        
        print(f'\nLoss A: {lossA.item()}\n')
        print(f'\nLoss B: {lossB.item()}')

        with open(f"Q.txt", "a") as f:
            f.write(f"{i} {((student.fc1.weight @ student.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"R.txt", "a") as f:
            f.write(f"{i} {((student.fc1.weight @ teacherA.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"S.txt", "a") as f:
            f.write(f"{i} {((teacherB.fc1.weight @ teacherB.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"T.txt", "a") as f:
            f.write(f"{i} {((teacherA.fc1.weight @ teacherA.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"U.txt", "a") as f:
            f.write(f"{i} {((student.fc1.weight @ teacherB.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"V.txt", "a") as f:
            f.write(f"{i} {((teacherA.fc1.weight @ teacherB.fc1.weight.T).detach()/N).flatten()}\n")

        with open(f"h_a.txt", "a") as f:
            f.write(f"{i} {student.fc2A.weight.detach().flatten()}\n")

        with open(f"h_b.txt", "a") as f:
            f.write(f"{i} {student.fc2B.weight.detach().flatten()}\n")

        if i >= switch:
          with open(f"forgetting.txt","a") as f:
              f.write(f'{i} {torch.abs(torch.log10(endlossA) - torch.log10(lossA)).item()}\n')
          
          with open(f"transfer.txt","a") as f:
              f.write(f'{i} {torch.abs(torch.log10(endlossB) - torch.log10(lossB)).item()}\n')

integration_step = 0.05
switch_step = int(alpha_s / integration_step)
num_steps = int(alpha_f / integration_step)

t_span = np.linspace(0,alpha_f*N,num_steps+1)
print(f'STARTING INTEGRATION OF THE ODEs')

Q = Q0.numpy()
Q = Q.reshape((K,K))
R = R0.numpy()
R = R.reshape((K,M))
U = U0.numpy()
U = U.reshape((K,M))
V = V0.numpy()
V = V.reshape((M,M))
T = T0.numpy()
T = T.reshape((M,M))
S = S0.numpy()
S = S.reshape((M,M))
Ha = h_a.numpy()
Ha = Ha.reshape((-1,))
Hb = h_b.numpy()
Hb = Hb.reshape((-1,))
v_a = v_a.numpy()
v_a = v_a.reshape((-1,))
v_b = v_b.numpy()
v_b = v_b.reshape((-1,))

R1 = np.concatenate([Q , R, U],axis=1)
R2 = np.concatenate([R.T,T,V],axis=1)
R3 = np.concatenate([U.T,V.T,S],axis=1)
C = np.concatenate( [R1,R2,R3],axis=0 )
errorA = ( sum( [ I2_val(C[[i,k],:][:,[i,k]])*Ha[i]*Ha[k] for i in range(K) for k in range(K) ] )   ) + (
      sum( [ I2_val(C[[K+n,K+m],:][:,[K+n,K+m]])*v_a[m]*v_a[n] for m in range(M) for n in range(M) ] )
    ) - 2*(
      sum( [ I2_val(C[[k,K+n],:][:,[k,K+n]])*Ha[k]*v_a[n] for k in range(K) for n in range(M) ] )
    )
errorB = ( sum( [ I2_val(C[[i,k],:][:,[i,k]])*Hb[i]*Hb[k] for i in range(K) for k in range(K) ] )   ) + (
      sum( [ I2_val(C[[K+M+n,K+M+m],:][:,[K+M+n,K+M+m]])*v_b[m]*v_b[n] for m in range(M) for n in range(M) ] )
    ) - 2*(
      sum( [ I2_val(C[[k,K+M+n],:][:,[k,K+M+n]])*Hb[k]*v_b[n] for k in range(K) for n in range(M) ] )
    )

with open(f"Q_num.txt", "w") as f:
  f.write(f"n Q_num\n")
with open(f"Q_num.txt", "a") as f:
  f.write(f"{0} {Q.flatten()}\n")

with open(f"R_num.txt", "w") as f:
  f.write(f"n R_num\n")
with open(f"R_num.txt", "a") as f:
  f.write(f"{0} {R.flatten()}\n")

with open(f"U_num.txt", "w") as f:
  f.write(f"n U_num\n")
with open(f"U_num.txt", "a") as f:
  f.write(f"{0} {U.flatten()}\n")

with open(f"V_num.txt", "w") as f:
  f.write(f"n V_num\n")
with open(f"V_num.txt", "a") as f:
  f.write(f"{0} {V.flatten()}\n")

with open(f"T_num.txt", "w") as f:
  f.write(f"n T_num\n")
with open(f"T_num.txt", "a") as f:
  f.write(f"{0} {T.flatten()}\n")

with open(f"S_num.txt", "w") as f:
  f.write(f"n S_num\n")
with open(f"S_num.txt", "a") as f:
  f.write(f"{0} {S.flatten()}\n")

with open(f"ha_num.txt", "w") as f:
  f.write(f"n ha_num\n")
with open(f"ha_num.txt", "a") as f:
  f.write(f"{0} {Ha.flatten()}\n")

with open(f"hb_num.txt", "w") as f:
  f.write(f"n S_num\n")
with open(f"hb_num.txt", "a") as f:
  f.write(f"{0} {Hb.flatten()}\n")

with open(f"errA_num.txt", "w") as f:
  f.write(f"n errA_num\n")
with open(f"errA_num.txt", "a") as f:
  f.write(f"{0} {errorA/2}\n")
      
with open(f"errB_num.txt", "w") as f:
  f.write(f"n errB_num\n")
with open(f"errB_num.txt", "a") as f:
  f.write(f"{0} {errorB/2}\n")

with open(f"forgetting_num.txt","w") as f:
  f.write(f'N forgetting_num\n')

with open(f"transfer_num.txt","w") as f:
  f.write(f'N transfer_num\n')

for step in range(num_steps):
    if  (step+1) % 50 ==0 :
        print(f"Integration step {step+1} over {num_steps}")
    # NUMERICAL INTEGRATION:
    
    if step <= switch_step:
        
      R = R + integration_step * update_R(C,Ha,alpha_W,v_a,K,M,'A')
      Q = Q + integration_step * update_Q(C,Ha,alpha_W,v_a,K,M,'A')
      U = U + integration_step * update_U(C,Ha,alpha_W,v_a,K,M,'A')
      Ha = Ha + integration_step * update_H(C,Ha,alpha_H,v_a,K,M,'A')
    else:
      R = R + integration_step * update_R(C,Hb,alpha_W,v_b,K,M,'B')
      Q = Q + integration_step * update_Q(C,Hb,alpha_W,v_b,K,M,'B')
      U = U + integration_step * update_U(C,Hb,alpha_W,v_b,K,M,'B')
      Hb = Hb + integration_step * update_H(C,Hb,alpha_H,v_b,K,M,'B')

    R1 = np.concatenate([Q , R, U],axis=1)
    R2 = np.concatenate([R.T,T,V],axis=1)
    R3 = np.concatenate([U.T,V.T,S],axis=1)
    C = np.concatenate( [R1,R2,R3],axis=0 )

    errorA = ( sum( [ I2_val(C[[i,k],:][:,[i,k]])*Ha[i]*Ha[k] for i in range(K) for k in range(K) ] )   ) + (
          sum( [ I2_val(C[[K+n,K+m],:][:,[K+n,K+m]])*v_a[m]*v_a[n] for m in range(M) for n in range(M) ] )
        ) - 2*(
          sum( [ I2_val(C[[k,K+n],:][:,[k,K+n]])*Ha[k]*v_a[n] for k in range(K) for n in range(M) ] )
        )

    errorB = ( sum( [ I2_val(C[[i,k],:][:,[i,k]])*Hb[i]*Hb[k] for i in range(K) for k in range(K) ] )   ) + (
          sum( [ I2_val(C[[K+M+n,K+M+m],:][:,[K+M+n,K+M+m]])*v_b[m]*v_b[n] for m in range(M) for n in range(M) ] )
        ) - 2*(
          sum( [ I2_val(C[[k,K+M+n],:][:,[k,K+M+n]])*Hb[k]*v_b[n] for k in range(K) for n in range(M) ] )
        )
    
    if step + 1 == switch_step:
       endlossA_num = errorA
       endlossB_num = errorB

    with open(f"Q_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {Q.flatten()}\n")
      
    with open(f"R_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {R.flatten()}\n")

    with open(f"U_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {U.flatten()}\n")

    with open(f"V_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {V.flatten()}\n")

    with open(f"T_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {T.flatten()}\n")

    with open(f"S_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {S.flatten()}\n")

    with open(f"ha_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {Ha.flatten()}\n")

    with open(f"hb_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {Hb.flatten()}\n")

    with open(f"errA_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {errorA/2}\n")

    with open(f"errB_num.txt", "a") as f:
      f.write(f"{float(t_span[step+1])} {errorB/2}\n")
    
    if step + 1 >= switch_step:
      with open(f"forgetting_num.txt","a") as f:
        f.write(f'{float(t_span[step+1])} {np.abs(np.log10(endlossA_num) - np.log10(errorA)).item()}\n')      
      with open(f"transfer_num.txt","a") as f:
        f.write(f'{float(t_span[step+1])} {np.abs(np.log10(endlossB_num) - np.log10(errorB)).item()}\n')

    