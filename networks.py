import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
class Teacher(nn.Module):
    def __init__(self, args):
        super(Teacher, self).__init__()
        self.N = args.N
        self.M = args.M
        self.teacher_activation_function = args.teacher_activation_function
        self.teacher_hl_initialization = args.teacher_hl_initialization

        self.fc1 = nn.Linear(args.N, self.M, bias=False)
        self.fc2 = nn.Linear(self.M, 1, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=1)

        if self.teacher_hl_initialization == "graded":
            graded_scale = torch.sqrt(torch.arange(1, self.M + 1, dtype=torch.float32, device=self.fc1.weight.device))
            with torch.no_grad():
                self.fc1.weight.data *= graded_scale.view(-1, 1)

    def forward(self, x):
        x = self.fc1(x) / (self.N**0.5)

        if self.teacher_activation_function == 'erf':
            x = torch.erf(x / math.sqrt(2))
        elif self.teacher_activation_function == 'ReLU':
            x = F.relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
            
        x = self.fc2(x)
        return x.unsqueeze(1)



def load_matching_run(path, args):
    data = np.load(path, allow_pickle=True).item()
    runs = data["runs"]

    def match(meta):
        return (
            ((int(meta["L"]) == int(args.L)) if args.method != "Sco-standard" else True) and
            float(meta["gamma"]) == float(args.gamma) and
            meta["informed_initialization"] == False and
            str(meta["method"]) == str(args.method) and
            np.isclose(float(meta["rho"]), float(args.rho)) and
            str(meta["student_activation_function"]) == str(args.student_activation_function) and
            str(meta["teacher_activation_function"]) == str(args.teacher_activation_function) and
            int(meta["M"]) == int(args.M) and
            int(meta["K"]) == int(args.K) and
            int(meta["N"]) == int(args.N) and
            float(meta["alpha"]) == float(args.alpha)
        )

    filtered = [r for r in runs if match(r["metadata"])]

    if len(filtered) != 1:
        raise ValueError(f"Expected exactly 1 matching run, found {len(filtered)}")

    return filtered[0]

def get_metric_at_step(run, metric_name, P):
    logs = run["logs"]

    if metric_name not in logs:
        raise ValueError(f"{metric_name} not found")

    steps = np.array(logs[metric_name]["steps"])
    values = np.array(logs[metric_name]["values"], dtype=object)

    idx = np.argmin(np.abs(steps - P))
    
    return values[idx]

def get_Q_from_theory(args):
    path = os.path.join(args.path_to_res_folder,f"ODES_K={args.K}_M={args.M}.npy")
    print(path)
    run = load_matching_run(path, args)
    P = args.alpha * args.N
    Q = get_metric_at_step(run, "Q", P)
    Q = torch.tensor(np.asarray(Q, dtype=np.float32))
    print("Q from theory : ", Q)

    return np.array(Q)

def initialize_with_information(Q, N):
    Q = torch.tensor(Q, dtype=torch.float32)

    Q = Q @ Q.T   # (M, M)

    eigvals, U = torch.linalg.eigh(Q)
    eigvals = torch.clamp(eigvals, min=0)

    sqrt_L = torch.diag(torch.sqrt(eigvals))

    K = Q.shape[0]

    G = torch.randn(N, N)
    Qrand, _ = torch.linalg.qr(G)

    Qk = Qrand[:K, :]

    W = (N ** 0.5) * (U @ sqrt_L @ Qk)

    return W

class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        self.args = args
        self.N, self.K, self.L = args.N, args.K, args.L
        self.gamma = args.gamma
        self.student_activation_function = args.student_activation_function
        self.informed_initialization = args.informed_initialization
        self.LoRA_prefactor = args.gamma/math.sqrt(args.L)
        self.student_initialization = args.student_initialization
        self.seed = int(args.seed)
        #self.LoRA_prefactor = args.gamma

        self.fc1 = nn.Linear(self.N, self.K, bias=False)
        self.head_1 = nn.Linear(self.K, 1, bias=False)
        self.head_2 = nn.Linear(self.K, 1, bias=False)

        for layer in [self.fc1, self.head_1, self.head_2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)

        std_head=0.01
        nn.init.normal_(self.head_1.weight, mean=0, std= std_head)
        nn.init.normal_(self.head_2.weight, mean=0, std= std_head)
        
        if self.student_initialization == "specializing":
            with torch.no_grad():
                half_k = self.K // 2
                self.head_1.weight.zero_()
                self.head_2.weight.zero_()
                self.head_1.weight[:, :half_k] = 0.01
                self.head_2.weight[:,:half_k] = 1
        
        else:
            with torch.no_grad():        
                nn.init.normal_(self.head_1.weight, mean=0.0, std=0.001)
                nn.init.normal_(self.head_2.weight, mean=0.0, std=0.001) 

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed+1) #Need to set specific seed for A and B such that learning curve of Sco-standard on first task reproduces the others.
        self.A = nn.Linear(self.N, self.L, bias=False)
        self.B = nn.Linear(self.L, self.K, bias=False)
        #Initialization of LoRA heads
        with torch.no_grad():
            self.A.weight.zero_()
            self.B.weight.zero_()
            self.B.weight[torch.arange(self.K), torch.arange(self.K) % self.L] = 1.0
            #TRIAL: RANDOM INIT FOR B
            #nn.init.normal_(self.B.weight, mean=0.0, std=1)
            
        torch.set_rng_state(rng_state)

        if self.informed_initialization:
            Q = get_Q_from_theory(args)
            fc1_weight = initialize_with_information(Q, self.N)
            print("student initialized with overlap", fc1_weight @ fc1_weight.T / self.N)

            with torch.no_grad():
                self.fc1.weight.copy_(fc1_weight)

    def update_grad_state(self, method):
        for param in self.parameters():
            param.requires_grad = True

        if method in ["LoRA", "Sco-LoRA", "only_LoRA", "Sco-standard"]:
            self.fc1.weight.requires_grad = False
            
            if method in ["Sco-LoRA", "Sco-standard"]:
                self.B.weight.requires_grad = False

    def forward(self, x, method="standard", task_id=1):
        h = self.fc1(x) / (self.N**0.5)

        if method in ["LoRA", "Sco-LoRA", "only_LoRA", "Sco-standard"]:
            l = self.A(x) / (self.N**0.5)
            l = self.B(l)
            
            if (method == "only_LoRA" and task_id == 2):
                h = (self.LoRA_prefactor) * l
            else:
                h = h + (self.LoRA_prefactor) * l

        if self.student_activation_function == 'erf':
            h = torch.erf(h / math.sqrt(2))
        elif self.student_activation_function == 'ReLU':
            h = torch.relu(h)
        elif self.student_activation_function == 'sigmoid':
            h = torch.sigmoid(h)
            
        return self.head_1(h), self.head_2(h)

class Data_and_Teachers():
    def __init__(self, args):
        print('Initializing data and teachers')
        self.N = args.N
        self.M = args.M
        self.rho = args.rho
        self.teacher_1 = Teacher(args)
        self.teacher_2 = Teacher(args)
        self.teacher_initialization = args.teacher_initialization
        self.teacher_hl_initialization = args.teacher_hl_initialization
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(self.N), torch.eye(self.N))
        self.device = args.device

    def get_Teachers(self):

        self.wt1 = self.teacher_1.fc1.weight.data
        r = torch.randn_like(self.wt1)
        if self.teacher_hl_initialization == "graded":
            graded_scale = torch.sqrt(torch.arange(1, self.M + 1, dtype=torch.float32))
            with torch.no_grad():
                r*= graded_scale.view(-1, 1)
            
        wt2 = self.rho * self.wt1 + torch.sqrt(1 - torch.tensor([self.rho])**2) * r


        self.teacher_2.fc1.weight.data = wt2

        if self.teacher_initialization == "graded":
            graded_weights = torch.arange(1, self.M + 1, dtype=torch.float32)
            self.teacher_1.fc2.weight.data = graded_weights
            self.teacher_2.fc2.weight.data = graded_weights
        elif self.teacher_initialization == "committee":
            self.teacher_1.fc2.weight.data = 1 + 0.01*torch.randn((self.M,))
            self.teacher_2.fc2.weight.data = 1 + 0.01*torch.randn((self.M,))
        else:
            raise ValueError(f"Unsupported teacher initialization: {self.teacher_initialization}")

        return self.teacher_1, self.teacher_2

    def get_data(self, task):
        self.datum = torch.randn((1, self.N)).to(self.device)

        self.yt1 = self.teacher_1(self.datum).detach()
        self.yt2 = self.teacher_2(self.datum).detach()

        return self.datum, self.yt1, self.yt2

    def get_target(self, x):
        y1 = self.teacher_1(x).detach()
        y2 = self.teacher_2(x).detach()
        return y1, y2
    
    def sample_x(self, P_test):
        return torch.randn(P_test, self.N, device=self.device)

    def test_error(self, model, loss, P_test, method="standard"):
        x = self.sample_x(P_test)
        y1, y2 = self.get_target(x)

        y_pred1, y_pred2 = model(x, method=method)

        l1 = 0.5 * loss(y_pred1, y1)
        l2 = 0.5 * loss(y_pred2, y2)

        return l1.item(), l2.item()

class MnistTask(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        x, digit = self.dataset[idx]
        #First task, <=5, second odd/even.
        y1 = torch.tensor(0.0 if digit <= 4 else 1.0,dtype=torch.float32,)
        y2 = torch.tensor(0.0 if digit % 2 == 0 else 1.0,dtype=torch.float32,)

        return x, y1, y2


class ContinualMnist:

    def __init__(self, args):

        self.device = args.device
        self.N = 784
        self.seed = int(args.seed)

        transform = transforms.ToTensor()

        train = MNIST("./data",train=True,download=True,transform=transform,)

        test = MNIST("./data",train=False,download=True,transform=transform,)
        generator = torch.Generator().manual_seed(self.seed)

        train1, train2 = random_split(
            train,
            [30000, 30000],
            generator=generator,
        )

        test1, test2 = random_split(
            test,
            [5000, 5000],
            generator=generator,
        )

        task1_train = MnistTask(train1)
        task2_train = MnistTask(train2)

        task1_test = MnistTask(test1)
        task2_test = MnistTask(test2)

        self.train_loader1 = DataLoader(task1_train,batch_size=1,shuffle=True,)

        self.train_loader2 = DataLoader(task2_train,batch_size=1,shuffle=True,)

        self.test_loader1 = DataLoader(task1_test,batch_size=256,shuffle=False,)

        self.test_loader2 = DataLoader(task2_test,batch_size=256,shuffle=False,)

        self.iter1 = iter(self.train_loader1)
        self.iter2 = iter(self.train_loader2)

    def get_Teachers(self):

        return None, None

    def get_data(self, task=1):

        if task == 1:
            try:
                x, y1, y2 = next(self.iter1)
            except StopIteration:
                # restart online stream with new shuffle
                self.iter1 = iter(self.train_loader1)
                x, y1, y2 = next(self.iter1)
        else:
            try:
                x, y1, y2 = next(self.iter2)

            except StopIteration:
                # restart online stream with new shuffle
                self.iter2 = iter(self.train_loader2)
                x, y1, y2 = next(self.iter2)
                
        x = x.view(1, -1).to(self.device)

        return (
            x,
            y1.to(self.device),
            y2.to(self.device),
        )

    def test_error(self, model, loss_fn, P_test, method):
        model.eval()
        loss1 = self.evaluate_task(
            model,
            self.test_loader1,
            P_test,
            task=1,
            loss_fn=loss_fn,
            method=method,
        )
        loss2 = self.evaluate_task(
            model,
            self.test_loader2,
            P_test,
            task=2,
            loss_fn=loss_fn,
            method=method,
        )
        model.train()

        return loss1, loss2

    def evaluate_task(self, model, loader, P_test, task, loss_fn, method):

        total_loss = 0.0
        n = 0
        
        with torch.no_grad():
            for x, y1, y2 in loader:
                x = x.view(x.size(0), -1).to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                pred1, pred2 = model(x,method=method,task_id=task,)

                if task == 1:
                    loss = loss_fn(pred1.squeeze(), y1.squeeze())
                else:
                    loss = loss_fn(pred2.squeeze(), y2.squeeze())

                total_loss += loss.item() * x.size(0)
                n += x.size(0)

                if n >= P_test:
                    break

        return total_loss / n
    
    
class CifarTask(Dataset):
    """
    Dataset wrapper filtering CIFAR-10 for specific pair of classes 
    and mapping them to binary float targets [0.0, 1.0].
    """
    def __init__(self, dataset, task_classes, task_num):
        self.dataset = dataset
        self.task_num = task_num
        self.c0, self.c1 = task_classes[0], task_classes[1]
        
        # Filter indices corresponding to selected task classes
        targets = getattr(dataset, 'targets', None)
        if targets is not None:
            self.indices = [i for i, label in enumerate(targets) if label in task_classes]
        else:
            self.indices = [i for i, (_, label) in enumerate(dataset) if label in task_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, label = self.dataset[real_idx]

        if self.task_num == 1:
            y1 = torch.tensor(0.0 if label == self.c0 else 1.0, dtype=torch.float32)
            y2 = torch.tensor(0.0, dtype=torch.float32)
        else:
            y1 = torch.tensor(0.0, dtype=torch.float32)
            y2 = torch.tensor(0.0 if label == self.c0 else 1.0, dtype=torch.float32)

        return x, y1, y2


class ContinualCifar:
    """
    Continual Learning dataloader for CIFAR-10 following ContinualMnist architecture.
    - Task 1: Classes [1, 7] -> Target [0.0, 1.0]
    - Task 2: Classes [9, 4] -> Target [0.0, 1.0]
    """
    def __init__(self, args):
        self.device = args.device
        
        # Adapt input dimension N based on args
        self.N = 784
        
        if self.N == 784:
            # Grayscale images
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])

        train = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = CIFAR10(root="./data", train=False, download=True, transform=transform)

        # Filter datasets for specific task classes
        task1_train = CifarTask(train, task_classes=[1, 7], task_num=1)
        task2_train = CifarTask(train, task_classes=[9, 4], task_num=2)
        task1_test = CifarTask(test, task_classes=[1, 7], task_num=1)
        task2_test = CifarTask(test, task_classes=[9, 4], task_num=2)
        
        task1_train = CifarTask(train, task_classes=[9, 4], task_num=1)
        task2_train = CifarTask(train, task_classes=[1, 7], task_num=2)
        task1_test = CifarTask(test, task_classes=[9, 4], task_num=1)
        task2_test = CifarTask(test, task_classes=[1, 7], task_num=2)

        self.train_loader1 = DataLoader(task1_train, batch_size=1, shuffle=True)
        self.train_loader2 = DataLoader(task2_train, batch_size=1, shuffle=True)

        self.test_loader1 = DataLoader(task1_test, batch_size=256, shuffle=False)
        self.test_loader2 = DataLoader(task2_test, batch_size=256, shuffle=False)

        self.iter1 = iter(self.train_loader1)
        self.iter2 = iter(self.train_loader2)

    def get_Teachers(self):
        return None, None

    def get_data(self, task=1):
        if task == 1:
            try:
                x, y1, y2 = next(self.iter1)
            except StopIteration:
                self.iter1 = iter(self.train_loader1)
                x, y1, y2 = next(self.iter1)
        else:
            try:
                x, y1, y2 = next(self.iter2)
            except StopIteration:
                self.iter2 = iter(self.train_loader2)
                x, y1, y2 = next(self.iter2)

        x = x.view(-1).to(self.device)

        return (
            x,
            y1.to(self.device),
            y2.to(self.device),
        )

    def test_error(self, model, loss_fn, P_test, method):
        model.eval()
        loss1 = self.evaluate_task(
            model,
            self.test_loader1,
            P_test,
            task=1,
            loss_fn=loss_fn,
            method=method,
        )
        loss2 = self.evaluate_task(
            model,
            self.test_loader2,
            P_test,
            task=2,
            loss_fn=loss_fn,
            method=method,
        )
        model.train()

        return loss1, loss2

    def evaluate_task(self, model, loader, P_test, task, loss_fn, method):
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for x, y1, y2 in loader:
                x = x.view(x.size(0), -1).to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                pred1, pred2 = model(x, method=method, task_id=task)

                if task == 1:
                    loss = loss_fn(pred1.squeeze(), y1.squeeze())
                else:
                    loss = loss_fn(pred2.squeeze(), y2.squeeze())

                total_loss += loss.item() * x.size(0)
                n += x.size(0)

                if n >= P_test:
                    break

        return total_loss / n
    
class FashionTaskDataset(Dataset):
    def __init__(self, dataset, class_map, task_id=1):
        """
        dataset: PyTorch FashionMNIST dataset
        class_map: dict mapping original class -> binary target label (e.g., {0: 0.0, 5: 1.0})
        task_id: 1 or 2 (determines whether target label goes to y1 or y2)
        """
        self.dataset = dataset
        self.class_map = class_map
        self.task_id = task_id

        # Extract targets array safely
        targets = dataset.targets if hasattr(dataset, "targets") else torch.tensor([y for _, y in dataset])
        
        # Filter indices containing only the requested classes for this task
        self.indices = [i for i, label in enumerate(targets) if label.item() in class_map]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, orig_class = self.dataset[real_idx]

        # Map original class to target binary label (0.0 or 1.0)
        target_label = torch.tensor(self.class_map[orig_class], dtype=torch.float32)

        if self.task_id == 1:
            y1 = target_label
            y2 = torch.tensor(0.0, dtype=torch.float32)  # Dummy target for unactive head
        else:
            y1 = torch.tensor(0.0, dtype=torch.float32)  # Dummy target for unactive head
            y2 = target_label

        return x, y1, y2


class ContinualFashionMNIST:
    def __init__(self, args):
        self.device = args.device
        self.N = 784  # 28x28 flattened input size

        transform = transforms.ToTensor()

        train = FashionMNIST("./data", train=True, download=True, transform=transform)
        test = FashionMNIST("./data", train=False, download=True, transform=transform)

        # Task 1: Class 0 -> 0.0, Class 5 -> 1.0
        task2_map = {0: 0.0, 5: 1.0}
        # Task 2: Class 2 -> 0.0, Class 7 -> 1.0
        task1_map = {2: 0.0, 7: 1.0}

        task1_train = FashionTaskDataset(train, task1_map, task_id=1)
        task2_train = FashionTaskDataset(train, task2_map, task_id=2)

        task1_test = FashionTaskDataset(test, task1_map, task_id=1)
        task2_test = FashionTaskDataset(test, task2_map, task_id=2)

        self.train_loader1 = DataLoader(task1_train, batch_size=1, shuffle=True)
        self.train_loader2 = DataLoader(task2_train, batch_size=1, shuffle=True)

        self.test_loader1 = DataLoader(task1_test, batch_size=256, shuffle=False)
        self.test_loader2 = DataLoader(task2_test, batch_size=256, shuffle=False)

        self.iter1 = iter(self.train_loader1)
        self.iter2 = iter(self.train_loader2)

    def get_Teachers(self):
        return None, None

    def get_data(self, task=1):
        if task == 1:
            try:
                x, y1, y2 = next(self.iter1)
            except StopIteration:
                self.iter1 = iter(self.train_loader1)
                x, y1, y2 = next(self.iter1)
        else:
            try:
                x, y1, y2 = next(self.iter2)
            except StopIteration:
                self.iter2 = iter(self.train_loader2)
                x, y1, y2 = next(self.iter2)

        x = x.view(1, -1).to(self.device)

        return (
            x,
            y1.to(self.device),
            y2.to(self.device),
        )

    def test_error(self, model, loss_fn, P_test, method):
        model.eval()
        loss1 = self.evaluate_task(
            model,
            self.test_loader1,
            P_test,
            task=1,
            loss_fn=loss_fn,
            method=method,
        )
        loss2 = self.evaluate_task(
            model,
            self.test_loader2,
            P_test,
            task=2,
            loss_fn=loss_fn,
            method=method,
        )
        model.train()

        return loss1, loss2

    def evaluate_task(self, model, loader, P_test, task, loss_fn, method):
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for x, y1, y2 in loader:
                x = x.view(x.size(0), -1).to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                pred1, pred2 = model(x, method=method, task_id=task)

                if task == 1:
                    loss = loss_fn(pred1.squeeze(), y1.squeeze())
                else:
                    loss = loss_fn(pred2.squeeze(), y2.squeeze())

                total_loss += loss.item() * x.size(0)
                n += x.size(0)

                if n >= P_test:
                    break

        return total_loss / n