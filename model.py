import torch.nn as nn
import torch.nn.functional as F

class Le_net_AMD_classifier(nn.Module):

    def __init__(self, dropout_rate):
        super(Le_net_AMD_classifier, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3,6,4)
        self.pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,12,5)
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12,24,5)
        self.batch_norm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24,48,5)
        self.batch_norm4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48,74,5)
        self.batch_norm5 = nn.BatchNorm2d(74)
        self.conv6 = nn.Conv2d(74,96,5)
        self.batch_norm6 = nn.BatchNorm2d(96)
        self.conv7 = nn.Conv2d(96,128,5)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(18432,9216)
        self.fc2 = nn.Linear(9216,4608)
        self.fc3 = nn.Linear(4608,750)
        self.fc4 = nn.Linear(750,1)

    def forward(self,x):
        
        x = self.batch_norm1(self.dropout(self.pool(F.relu(self.conv1(x)))))
        x = self.batch_norm2(self.dropout(self.pool(F.relu(self.conv2(x)))))
        x = self.batch_norm3(self.dropout(self.pool(F.relu(self.conv3(x)))))
        x = self.batch_norm4(self.dropout(self.pool(F.relu(self.conv4(x)))))
        x = self.batch_norm5(self.dropout(self.pool(F.relu(self.conv5(x)))))
        x = self.batch_norm6(self.dropout(self.pool(F.relu(self.conv6(x)))))
        x = self.batch_norm7(self.dropout(self.pool(F.relu(self.conv7(x)))))
        x = x.view(-1,128*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
