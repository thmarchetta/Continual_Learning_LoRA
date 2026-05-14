import torch
import torch.nn as nn
from numpy import sqrt
class Teacher(nn.Module):
    def __init__(self, args):
        super(Teacher, self).__init__()
        self.N = args.N
        self.M = args.M

        self.fc1 = nn.Linear(args.N, args.M, bias=False)

        self.fc2 = nn.Linear(args.M, 1, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=1)

    def forward(self, x):
        x = self.fc1(x)/(self.N**0.5)
        x = torch.erf(x/sqrt(2))
        x = self.fc2(x)
        return x.unsqueeze(1)

class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        self.N = args.N
        self.K = args.K
        self.L = args.L
        self.device = args.device
        self.beta = args.beta

        self.fc1 = nn.Linear(args.N, args.K, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)
        
        # case where we learn new, full rank, random init matrix
        self.wt = nn.Linear(args.N, args.K, bias=False)
        nn.init.normal_(self.wt.weight, mean=0.0, std=0.001)

        self.head_1 = nn.Linear(args.K, 1,  bias=False)
        self.head_2 = nn.Linear(args.K, 1,  bias=False)

        nn.init.normal_(self.head_1.weight, mean=0.0, std=0.0001)
        nn.init.normal_(self.head_2.weight, mean=0.0, std=0.0001)

        self.A = nn.Linear(args.N, args.L, bias = False)
        self.B = nn.Linear(args.L, args.K, bias = False)

        nn.init.normal_(self.A.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.B.weight, mean=0.0, std=0.001)


    def forward(self, x, LoRA=False, A_only=False):
        h = self.fc1(x)/(self.N**0.5)
        if LoRA:
          self.fc1.weight.detach_()
          if A_only == True :
            self.B.weight.detach_()
          l = self.A(x)/(self.N**0.5)
          l = self.B(l)
          h = h + (self.beta/sqrt(self.L)) * l

        h = torch.erf(h/sqrt(2))

        o1 = self.head_1(h)
        o2 = self.head_2(h)

        return o1, o2

class Data_and_Teachers():
    def __init__(self, args):
        print('Initializing data and teachers')
        self.N = args.N
        self.M = args.M
        self.rho = args.rho
        self.teacher_1 = Teacher(args)
        self.teacher_2 = Teacher(args)
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(self.N), torch.eye(self.N))
        self.device = args.device

    def get_Teachers(self):
        #self.target_data = self.dist.sample((1,self.input_size))

        # take random initialised teacher
        self.wt1 = self.teacher_1.fc1.weight.data


        # take a orthogonal vector
        r = torch.randn_like(self.wt1)

        # Combine w1 and r with overlap rho
        wt2 = self.rho * self.wt1 + torch.sqrt(1 - torch.tensor([self.rho])**2) * r

        # obtain w2 and update the weights
        self.teacher_2.fc1.weight.data = wt2

        # set the output layer to +1, -1
        self.teacher_1.fc2.weight.data = 1 + 0.01*torch.randn((self.M,))
        self.teacher_2.fc2.weight.data = -1 + 0.01*torch.randn((self.M,))

        return self.teacher_1, self.teacher_2

    def get_data(self):
        #self.datum = self.dist.sample((1,)).to(device)
        self.datum = torch.randn((1, self.N)).to(self.device) # sample x

        self.yt1 = self.teacher_1(self.datum).detach() # obtain y from teacher 1
        self.yt2 = self.teacher_2(self.datum).detach() # obtain y from teacher 2

        return self.datum, self.yt1, self.yt2

    def get_target(self, x):    # get only targets given x
        y1 = self.teacher_1(x).detach()
        y2 = self.teacher_2(x).detach()
        return y1, y2
    
    def sample_x(self, P_test):
        return torch.randn(P_test, self.N, device=self.device)

    def test_error(self, model, loss, P_test, LoRA=False):
        x = self.sample_x(P_test)
        y1, y2 = self.get_target(x)

        y_pred1, y_pred2 = model(x, LoRA=LoRA)

        l1 = 0.5 * loss(y_pred1, y1)
        l2 = 0.5 * loss(y_pred2, y2)

        return l1.item(), l2.item()