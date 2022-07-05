import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_size=16, num_classes=4):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.input_size=input_size
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(self.input_size, self.num_classes)


    def forward(self, x):
        out =self.pool(x)
        out =torch.squeeze(out)
        out = self.linear(out)
        return out

class CNN(nn.Module):
#regular cnn
    def __init__(self,input_size=3, num_classes=4):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size=input_size
        self.Conv1 =  nn.Conv2d(self.input_size, 64, kernel_size=3, stride=2, padding=0)
        self.Conv1_bn = nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.Conv2_bn = nn.BatchNorm2d(128)
        self.FC1 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        #self.FC2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        #print("INPUT 2222 STUDENT DCNNS checking", x.size())
        #x = F.avg_pool2d(x, kernel_size=3, stride=2)
        #x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv1_bn(self.Conv1(x)))
        #print("output 0000 after conv checking", x.size())
        x = F.avg_pool2d(x, kernel_size=3, stride=2)
        #x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv2_bn(self.Conv2(x)))
        #print("output 0000 after conv checking", x.size())
        x = F.avg_pool2d(x, 63)
        #print("output 333333 after conv checking", x.size())
        x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), 64*63*63)
        x = self.dropout(x)

        x = self.FC1(x)
        #x = self.FC2(x)
        return x


class argmax_CNN(nn.Module):
#regular cnn
    def __init__(self,input_size=1, num_classes=4):
        super(argmax_CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size=input_size
        self.Conv1 =  nn.Conv2d(self.input_size, 64, kernel_size=3, stride=2, padding=0)
        self.Conv1_bn = nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.Conv2_bn = nn.BatchNorm2d(128)
        self.FC1 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        #self.FC2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        #print("INPUT 2222 STUDENT DCNNS checking", x.size())
        #x = F.avg_pool2d(x, kernel_size=3, stride=2)
        #x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv1_bn(self.Conv1(x)))
        #print("output 0000 after conv checking", x.size())
        x = F.avg_pool2d(x, kernel_size=3, stride=2)
        #x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv2_bn(self.Conv2(x)))
        #print("output 0000 after conv checking", x.size())
        x = F.avg_pool2d(x, 63)
        #print("output 333333 after conv checking", x.size())
        x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), 64*63*63)
        x = self.dropout(x)

        x = self.FC1(x) #change to F.relu?
        #x = self.FC2(x)
        return x


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size=15, num_classes=4):
        super(LogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , self.num_classes)

    def forward(self, x):
        x = self.linear(x)
        outputs = x
        return outputs

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=15, num_classes=4):
        super(LinearRegression, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , self.num_classes)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(4)

    def forward(self, x, temp=1):
        x = self.bn1(self.linear(x))
        x = self.dropout(x)
        outputs = x
        return outputs

class LinearRegressionCDGAI(torch.nn.Module):
    def __init__(self, input_size=22, num_classes=13):
        super(LinearRegressionCDGAI, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(self.num_classes)

    def forward(self, x, temp=1):
        x = self.bn1(self.linear(x))
        x = self.dropout(x)
        outputs = x
        return outputs

class LinearRegressionPascal(torch.nn.Module):
    def __init__(self, input_size=23, num_classes=16):
        super(LinearRegressionPascal, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(self.num_classes)

    def forward(self, x, temp=1):
        x = self.bn1(self.linear(x))
        x = self.dropout(x)
        #outputs = x
        return x

class DoubleLinearRegression(torch.nn.Module):
    def __init__(self, input_size=15, num_classes=4):
        super(DoubleLinearRegression, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , 100)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(100)
        self.linear_2 = torch.nn.Linear(100, self.num_classes)

    def forward(self, x, temp=1):
        x = F.relu(self.bn1(self.linear(x)))
        x = self.dropout(x)
        x = self.linear_2(x)
        outputs = x
        return outputs

class DoubleLinearRegressionPascal(torch.nn.Module):
    def __init__(self, input_size=44, num_classes=20):
        super(DoubleLinearRegressionPascal, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , 100)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(100)
        self.linear_2 = torch.nn.Linear(100, self.num_classes)

    def forward(self, x, temp=1):
        x = F.relu(self.bn1(self.linear(x)))
        x = self.dropout(x)
        x = self.linear_2(x)
        outputs = x
        return outputs

class TripleLinearRegression(torch.nn.Module):
    def __init__(self, input_size=15, num_classes=4):
        super(TripleLinearRegression, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size , 100)
        self.linear_2 = torch.nn.Linear(100, 100)
        self.linear_3 = torch.nn.Linear(100, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x, temp=1):
        x = F.relu(self.bn1(self.linear(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.linear_2(x)))
        x = self.dropout(x)
        x = self.linear_3(x)
        outputs = x
        return outputs

class TripleLinearRegressionPascal(torch.nn.Module):
    def __init__(self, input_size=44, num_classes=20):
        super(TripleLinearRegressionPascal, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size, 100)
        self.linear_2 = torch.nn.Linear(100, 100)
        self.linear_3 = torch.nn.Linear(100, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x, temp=1):
        x = F.relu(self.bn1(self.linear(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.linear_2(x)))
        x = self.dropout(x)
        x = self.linear_3(x)
        outputs = x
        return outputs
