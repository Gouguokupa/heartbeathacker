import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

class IEGMNet(nn.Module):
    def __init__(self):
        super(IEGMNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(4,1), padding=0),
            nn.ReLU(True),
            #nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(4,1), padding=0),
            nn.ReLU(True),
            #nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            #nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=370, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv3_output = conv3_output.view(-1,370)

        fc1_output = F.relu(self.fc1(conv3_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class IEGMNetSimple5a(nn.Module):             #number 4 solution prev year
    def __init__(self):
        super(IEGMNetSimple5a, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(2, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.linear = nn.Linear(370, 2)

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 370)

        out = self.linear(conv5_output)

        return out



class IEGMNet1(nn.Module):
    def __init__(self):
        super(IEGMNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=64, stride=10, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=10, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=550, out_features=20)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=20, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,550)

        fc1_output = F.relu(self.fc1(conv2_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output


class IEGMNet2(nn.Module):
    def __init__(self):
        super(IEGMNet2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=64, stride=10, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=16, stride=4, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=130, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,130)

        fc1_output = self.fc1(conv2_output)
        return fc1_output



class MyNet4(nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(15, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2, 1), padding=0),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        # )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=918, out_features=2),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=10, out_features=2)
        # )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.fc1(x)
        # x = self.fc2(x)


        return x

class IEGMNet3(nn.Module):
    def __init__(self):
        super(IEGMNet3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=49, stride=10, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=16, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=530, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,530)

        fc1_output = self.fc1(conv2_output)
        return fc1_output


class IEGMNet4(nn.Module):
    def __init__(self):
        super(IEGMNet4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(64, 1), stride=(10, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(16, 1), stride=(1, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=520, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,520)

        fc1_output = self.fc1(conv2_output)
        return fc1_output


class IEGMNet5(nn.Module):
    def __init__(self):
        super(IEGMNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=85, stride=5, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=12, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1115, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,1115)

        fc1_output = self.fc1(conv2_output)
        return fc1_output

class IEGMNet6(nn.Module):
    def __init__(self):
        super(IEGMNet6, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=16, stride=10, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=13, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=10, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(1, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )


        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=103, out_features=16)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv3_output = conv3_output.view(-1,103)

        fc1_output = self.fc1(conv3_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class IEGMNet2_quant(nn.Module):
    def __init__(self):
        super(IEGMNet2_quant, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=64, stride=10, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=16, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=0.001, momentum=0.99),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=520, out_features=2)
        )

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):

        x = self.quant(input)
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        conv2_output = conv2_output.view(-1,520)

        fc1_output = self.fc1(conv2_output)
        out = self.dequant(fc1_output)
        return out